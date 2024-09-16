import argparse
from datatools import prepare_test_set
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
        device, do_sample, length_penalty, max_new_tokens, model,
        repetition_penalty, temperature, test_loader, tokenizer, top_k,
        top_p):
    model.eval()
    preds = []
    for batch in test_loader:
        inp, att = (x.to(device) for x in batch)
        outputs = model.generate(
            do_sample=do_sample,
            input_ids=inp,
            length_penalty=length_penalty,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            )
        decoded_outputs = tokenizer.batch_decode(
            outputs.detach().cpu().numpy(),
            skip_special_tokens=True)
        preds.append(decoded_outputs)
    return preds

def main():
    # Configure logger
    job_id = os.environ["SLURM_JOB_ID"]
    job_name = os.environ["SLURM_JOB_NAME"]
    logging.basicConfig(
            filename=f"logs/{job_id}_{job_name}.log",
            level=logging.INFO)
    logger.info(f"{job_id}_{job_name}")

    # Load program configuration from file
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--config", help="path to configuration file", required=True)
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = json.load(config_file)
    
    # Instantiate PEFT model and tokenizer
    model = load_model(config["model_path"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # Prepare data for generation
    num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    pin_memory = torch.cuda.is_available()
    test_loader, test_data = prepare_test_set(
            config=config,
            num_workers=num_workers,
            pin_memory=pin_memory,
            tokenizer=tokenizer,
            )

    # Configure generation
    device = (torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu"))
    model.to(device)

    preds = run_generation(
        device=device,
        do_sample=True if config["do_sample"] else False,
        length_penalty=config["length_penalty"],
        max_new_tokens=config["max_new_tokens"],
        model=model,
        repetition_penalty=config["repetition_penalty"],
        temperature=config["temperature"],
        test_loader=test_loader,
        tokenizer=tokenizer,
        top_k=config["top_k"],
        top_p=config["top_p"],
        )

    # Write predictions to file
    test_data["PREDICTION"] = preds
    outfile=f"./preds/{job_id}_{job_name}.csv"
    test_data.to_csv(outfile)
    logger.info(f"Predictions saved to '{outfile}'")
 
if __name__ == "__main__":
    main()
