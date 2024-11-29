#!/usr/bin/env python3

import argparse
from datautils import prepare_datasets
from datetime import datetime
import json
import logging
import os
import pandas as pd
from peft import AutoPeftModelForCausalLM
import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

def load_model(model_name):
    start_time = datetime.now()
    model = AutoPeftModelForCausalLM.from_pretrained(model_name)
    load_time = str(datetime.now() - start_time)
    logger.info(f"PEFT model loaded in {load_time}")
    return model

@torch.no_grad
def run_generation(
        batch_size, device, gen_kwargs, log_steps, model, test_loader,
        tokenizer):
    gen_start = datetime.now()
    logger.info("Generation started")
    model.eval()
    preds = []
    for step, batch in enumerate(test_loader):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        prompt_texts = tokenizer.batch_decode(
                batch[0], skip_special_tokens=True)
        prompt_lengths = [len(text) for text in prompt_texts]
        inp = batch[0].to(device)
        att = batch[1].to(device)
        outputs = model.generate(
                attention_mask=att, input_ids=inp, **gen_kwargs)
        decoded_outputs = tokenizer.batch_decode(
            outputs.detach().cpu().numpy(),
            skip_special_tokens=True
            )
        for i in range(len(decoded_outputs)):
            # Exclude prompt from stored output
            p_len = prompt_lengths[i]
            preds.append(decoded_outputs[i][p_len:])
        if (step + 1) % log_steps == 0 or (step + 1) == len(test_loader):
                logger.info(f"{step + 1}/{len(test_loader)}")
    gen_time = str(datetime.now() - gen_start)
    logger.info(f"Generation finished in {gen_time}")
    return preds

def main():
    # Configure logger
    job_id = os.environ["SLURM_JOB_ID"]
    job_name = os.environ["SLURM_JOB_NAME"]
    logging.basicConfig(
            filename=f"logs/{job_id}-{job_name}.log",
            level=logging.INFO
            )
    logger.info(f"{job_id}-{job_name}")

    # Load program configuration and model from files
    parser = argparse.ArgumentParser()
    parser.add_argument( "--data_config", required=True)
    parser.add_argument( "--task_config", required=True)
    parser.add_argument("--peft_model", required=True)
    args = parser.parse_args()

    with open(args.data_config) as data_config_file:
        data_config = json.load(data_config_file)
    with open(args.task_config) as task_config_file:
        task_config = json.load(task_config_file)
    for key, val in task_config.items():
        logger.info(f"{key}: {val}")
    config = {**data_config, **task_config}

    preds_path=f"preds/{job_id}-{job_name}.csv"
    prompt_templates_path = "config/prompt_templates.json"

    # Instantiate PEFT model and tokenizer
    model = load_model(args.peft_model)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(
            config["base_model"], padding_side="left")

    # Prepare data for generation
    num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    pin_memory = torch.cuda.is_available()
    with open(prompt_templates_path) as prompt_templates_file:
        prompt_templates = json.load(prompt_templates_file)
    test_data, test_loader = prepare_datasets(
            config=config,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prompt_templates=prompt_templates,
            tokenizer=tokenizer,
            )

    # Configure generation
    device = (torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu"))
    model.to(device)
    gen_kwargs = {
            "do_sample": True if config["do_sample"] else False,
            "early_stopping": True if config["early_stopping"] else False,
            "low_memory": True if config["low_memory"] else False,
            "max_new_tokens": config["max_new_tokens"],
            "num_beams": config["num_beams"],
            "repetition_penalty": config["repetition_penalty"],
            "temperature": config["temperature"],
            }
    preds = run_generation(
            batch_size=config["batch_size"],
            device=device,
            gen_kwargs=gen_kwargs,
            log_steps=config["log_steps"],
            model=model,
            test_loader=test_loader,
            tokenizer=tokenizer,
            )

    # Write predictions to file
    df_preds = pd.DataFrame(
            {
                "word": test_data[config["colname_word"]],
                "example": test_data[config["colname_ex"]],
                "target": test_data[config["colname_def"]],
                "prediction": preds
                }
            )
    df_preds.to_csv(preds_path, index=False)
    logger.info(f"Predictions saved to '{preds_path}'")
 
if __name__ == "__main__":
    main()
