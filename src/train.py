#!/usr/bin/env python3

import argparse
from datautils import prepare_datasets
from datetime import datetime
import json
import logging
import os
from peft import get_peft_model, PrefixTuningConfig, TaskType
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

def load_model(model_name):
    start_time = datetime.now()
    model = AutoModelForCausalLM.from_pretrained(model_name)
    load_time = str(datetime.now() - start_time)
    logger.info(f"Base model loaded in {load_time}")
    return model

@torch.no_grad
def run_eval(model, val_loader, device, total_loss, train_loader):
    model.eval()
    val_loss = 0
    for step, batch in enumerate(val_loader):
        inp, att, lbl = (x.to(device) for x in batch)
        outputs = model(input_ids=inp, attention_mask=att, labels=lbl)
        loss = outputs.loss
        val_loss += loss.detach().float()

    train_epoch_loss = total_loss / len(train_loader)
    train_ppl = torch.exp(train_epoch_loss)
    val_epoch_loss = val_loss / len(val_loader)
    val_ppl = torch.exp(val_epoch_loss)

    return train_epoch_loss, train_ppl, val_epoch_loss, val_ppl

def run_training(
        device, log_steps, model, optimizer, num_epochs, train_loader,
        val_loader, writer):
    train_start = datetime.now()
    logger.info("Training started")

    # Training procedure
    for epoch in range(num_epochs):
        epoch_start = datetime.now()
        logger.info(f"Epoch {epoch + 1} started")
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            inp, att, lbl = (x.to(device) for x in batch)
            outputs = model(input_ids=inp, attention_mask=att, labels=lbl)
            loss = outputs.loss
            writer.add_scalar("Loss/train", loss, step)
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (step + 1) % log_steps == 0 or (step + 1) == len(train_loader):
                logger.info(f"{step + 1}/{len(train_loader)}")

        # Evaluate and log results
        train_loss, train_ppl, val_loss, val_ppl = run_eval(
                model, val_loader, device, total_loss, train_loader)
        epoch_time = str(datetime.now() - epoch_start)
        logger.info(f"{train_loss=:.3f}, {train_ppl=:.3f}")
        logger.info(f"{val_loss=:.3f}, {val_ppl=:.3f}")
        logger.info(f"Epoch {epoch + 1} finished in {epoch_time}")
    train_time = str(datetime.now() - train_start)
    logger.info(f"Training finished in {train_time}")

def main():
    # Configure logger
    job_id = os.environ["SLURM_JOB_ID"]
    job_name = os.environ["SLURM_JOB_NAME"]
    logging.basicConfig(
            filename=f"logs/{job_id}-{job_name}.log",
            level=logging.INFO
            )
    logger.info(f"{job_id}-{job_name}")

    # Load program configuration from file
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--task_config", required=True)
    args = parser.parse_args()

    with open(args.data_config) as data_config_file:
        data_config = json.load(data_config_file)
    with open(args.task_config) as task_config_file:
        task_config = json.load(task_config_file)
    for key, val in task_config.items():
        logger.info(f"{key}: {val}")  # log hyperparams
    config = {**data_config, **task_config}

    # Set paths
    model_save_dir = f"models/{job_id}-{job_name}"
    prompt_templates_path = "config/prompt_templates.json"
    tensorboard_log_dir = f"runs/{job_id}-{job_name}"
    
    # Instantiate base model and tokenizer
    base_model = load_model(task_config["model_name"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # Prepare data for training
    num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    pin_memory = torch.cuda.is_available()
    with open(prompt_templates_path) as prompt_templates_file:
        prompt_templates = json.load(prompt_templates_file)
    train_loader, val_loader = prepare_datasets(
            config=config,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prompt_templates=prompt_templates,
            tokenizer=tokenizer,
            )

    # Instantiate PEFT model
    peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            num_virtual_tokens=config["num_virtual_tokens"],
            )
    peft_model = get_peft_model(base_model, peft_config)

    # Configure training
    device = (torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu"))
    peft_model.to(device)
    optimizer = AdamW(peft_model.parameters(), lr=config["lr"])

    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    run_training(
            device=device,
            log_steps=config["log_steps"],
            model=peft_model,
            optimizer=optimizer,
            num_epochs=config["num_epochs"],
            train_loader=train_loader,
            val_loader=val_loader,
            writer=writer,
            )
    writer.close()

    peft_model.save_pretrained(model_save_dir)
    logger.info(f"PEFT model saved to '{model_save_dir}'")
    
if __name__ == "__main__":
    main()
