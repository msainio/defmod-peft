#!/usr/bin/env python3

import argparse
import bert_score
from bert_score import BERTScorer
import json
import logging
import numpy as np
import os
import pandas as pd
from rouge_score.rouge_scorer import RougeScorer
from sacrebleu.metrics import BLEU, CHRF

logger = logging.getLogger(__name__)

def main():
    job_id = os.environ["SLURM_JOB_ID"]
    job_name = os.environ["SLURM_JOB_NAME"]
    logging.basicConfig(
            filename=f"logs/{job_id}-{job_name}.log",
            level=logging.INFO
            )
    logger.info(f"{job_id}-{job_name}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", "-d", required=True)
    parser.add_argument("--predictions", "-p", required=True)
    args = parser.parse_args()

    with open(args.data_config) as data_config_file:
        data_config = json.load(data_config_file)

    colname_def = "target"
    colname_ex = "example"
    colname_pred = "prediction"
    colname_word = "word"
    data_lang = data_config["data_lang"]
    preds_path = args.predictions
    scores_path = f"scores/{job_id}-{job_name}.csv"

    logger.info(f"predictions: {preds_path}")

    data = pd.read_csv(preds_path, lineterminator="\n")
    data = data.map(lambda x: str(x))
    preds = data[colname_pred].to_list()
    refs = data[colname_def].to_list()

    scores = {
            "word": data[colname_word],
            "example": data[colname_ex],
            "target": data[colname_def],
            "prediction": data[colname_pred],
            "chrF++": [],
            "bleu": [],
            "rougeL": [],
            "bertF1": [],
            }

    if data_lang == "en":
        model_type = "google-bert/bert-base-uncased"
    elif data_lang == "fi":
        model_type = "TurkuNLP/bert-base-finnish-uncased-v1"
    num_layers = bert_score.utils.model2layers["bert-base-uncased"]

    chrf_scorer = CHRF(word_order=2)
    bleu_scorer = BLEU(effective_order=True)
    rouge_scorer = RougeScorer(["rougeL"])
    bert_scorer = BERTScorer(model_type=model_type, num_layers=num_layers)

    logger.info("Evaluation started")
    logger.info("Computing character- and token-based metrics")
    for i in range(len(preds)):
        pred = preds[i]
        ref = refs[i]
        scores["chrF++"].append(
                chrf_scorer.sentence_score(
                    hypothesis=pred, references=[ref]).score
                )
        scores["bleu"].append(
                bleu_scorer.sentence_score(
                    hypothesis=pred, references=[ref]).score
                )
        scores["rougeL"].append(
                rouge_scorer.score(
                    prediction=pred, target=ref)["rougeL"].fmeasure
                )
    logger.info("Computing BERTScore")
    scores["bertF1"] = bert_scorer.score(preds, refs)[-1]

    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(scores_path, index=False)
    logger.info(f"Scores saved to '{scores_path}'")

if __name__ == "__main__":
    main()
