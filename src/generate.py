#!/usr/bin/env python3

import argparse
from datautils import prepare_datasets
from datetime import datetime
import json
import logging
import os
from peft import AutoPeftModelForCausalLM
import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

def load_model(model_name):
    start_time = datetime.now()
    model = AutoPeftModelForCausalLM.from_pretrained(model_name)
    logger.info(
            "PEFT model loaded in" + " "
            + str(datetime.now() - start_time))
    return model

@torch.no_grad
def run_generation(
        device, do_sample, early_stopping, log_steps, low_memory,
        max_new_tokens, model, num_beams, repetition_penalty, temperature,
        test_loader, tokenizer):
    gen_start = datetime.now()
    logger.info("Generation started")
    model.eval()
    preds = []
    for step, batch in enumerate(test_loader):
        inp = batch[0].to(device)
        att = batch[1].to(device)
        outputs = model.generate(
            attention_mask=att,
            do_sample=do_sample,
            early_stopping=early_stopping,
            input_ids=inp,
            low_memory=low_memory,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            )
        decoded_outputs = tokenizer.batch_decode(
            outputs.detach().cpu().numpy(),
            skip_special_tokens=True,
            )
        preds += decoded_outputs
        if (step + 1) % log_steps == 0 or (step + 1) == len(test_loader):
                logger.info(f"{step + 1}/{len(test_loader)}")
    logger.info("Generation finished in" + " "
            + str(datetime.now() - gen_start))
    return preds

def main():
    # Configure logger
    job_id = os.environ["SLURM_JOB_ID"]
    job_name = os.environ["SLURM_JOB_NAME"]
    logging.basicConfig(
            filename=f"logs/{job_id}-{job_name}.log",
            level=logging.INFO)
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

    # Set paths
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

    preds = run_generation(
        device=device,
        do_sample=True if config["do_sample"] else False,
        early_stopping=True if config["early_stopping"] else False,
        log_steps=config["log_steps"],
        low_memory=True if config["low_memory"] else False,
        max_new_tokens=config["max_new_tokens"],
        model=model,
        num_beams=config["num_beams"],
        repetition_penalty=config["repetition_penalty"],
        temperature=config["temperature"],
        test_loader=test_loader,
        tokenizer=tokenizer,
        )

    # Write predictions to file
    test_data[config["colname_pred"]] = preds
    test_data.to_csv(preds_path)
    logger.info(f"Predictions saved to '{preds_path}'")
 
if __name__ == "__main__":
    main()
