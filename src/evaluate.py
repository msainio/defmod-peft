#!/usr/bin/env python3

import argparse
import evaluate

def main():
    # Configure logger
    job_id = os.environ["SLURM_JOB_ID"]
    job_name = os.environ["SLURM_JOB_NAME"]
    logging.basicConfig(
            filename=f"logs/{job_id}-{job_name}.log",
            level=logging.INFO)
    logger.info(f"{job_id}-{job_name}")

    # Load configuration and data from files
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", "-d", required=True)
    parser.add_argument("--predictions", "-p", required=True)
    args = parser.parse_args()

    with open(args.data_config) as data_config_file:
        data_config = json.load(data_config_file)

    preds_file = args.predictions
    logger.info(f"{preds_file}")

    data = pd.read_csv(preds_file)
    preds = data[data_config["colname_pred"]]
    refs = data[data_config["colname_def"]]

    sacrebleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    results_sacrebleu = sacrebleu.compute(
            predictions=preds, references=refs)
    results_rouge = rouge.compute(
            predictions=preds, references=refs)
    results_bertscore = bertscore.compute(
            predictions=preds, references=refs)

    bleu = results_sacrebleu["score"]
    rougeL = results_rouge["rougeL"]
    bertscore_f1 = results_bertscore["f1"]

    logger.info(f"{model=}")
    logger.info(f"{bleu=}")
    logger.info(f"{rougeL=}")
    logger.info(f"{bertscore_f1=}")

if __name__ == "__main__":
    main()
