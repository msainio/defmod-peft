#!/usr/bin/env python3

import argparse
from bert_score import BERTScorer
import json
import numpy as np
import os
import pandas as pd
from rouge_score.rouge_scorer import RougeScorer
from sacrebleu.metrics import BLEU, CHRF

def main():
    job_id = os.environ["SLURM_JOB_ID"]
    job_name = os.environ["SLURM_JOB_NAME"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", "-d", required=True)
    parser.add_argument("--predictions", "-p", required=True)
    args = parser.parse_args()

    with open(args.data_config) as data_config_file:
        data_config = json.load(data_config_file)

    colname_def = data_config["colname_def"]
    colname_ex = data_config["colname_ex"]
    colname_pred = "prediction"
    colname_word = data_config["colname_word"]
    data_lang = data_config["data_lang"]
    preds_path = args.predictions
    scores_path = f"scores/{job_id}-{job_name}.csv"

    data = pd.read_csv(preds_path, lineterminator="\n")
    preds = data[colname_pred].to_list()
    refs = data[data_config["colname_def"]].to_list()

    scores = {
            "word": data[colname_word],
            "example": data[colname_ex],
            "gloss": data[colname_def],
            "prediction": data[colname_pred],
            "chrF++": [],
            "bleu": [],
            "rougeL": [],
            "bert_score_F1": [],
            }

    if data_lang == "en":
        model_type = "google-bert/bert-base-uncased"
    elif data_lang == "fi":
        model_type = "TurkuNLP/bert-base-finnish-uncased-v1"
    num_layers = bert_score.utils.model2layers["bert-base-uncased"]

    chrf = CHRF(word_order=2)
    bleu = BLEU()
    rouge = RougeScorer(["rougeL"])
    bert_score = BERTScorer(model_type=model_type, num_layers=num_layers)

    for i in range(len(preds)):
        pred = preds[i]
        ref = refs[i]
        scores["chrF++"].append(
                chrf.sentence_score(hypothesis=pred, references=ref))
        scores["bleu"].append(
                bleu.sentence_score(hypothesis=pred, references=ref))
        scores["rougeL"].append(
                rouge.score(prediction=pred, target=ref))
    scores["bert_score_F1"] = bert_score.score(preds, refs)[-1]

    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(scores_path, index=False)

if __name__ == "__main__":
    main()
