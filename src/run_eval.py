#!/usr/bin/env python3

import argparse
import bert_score
import evaluate
import json
import numpy as np
import os
import pandas as pd
import re

def remove_prompt(row, colname_pred, colname_word, data_lang):
    if data_lang == "en":
        pattern = rf"What is the definition of {row[colname_word]}\??(.*)"
    elif data_lang == "fi":
        pattern = rf"Mikä on sanan {row[colname_word]} määritelmä\??(.*)"
    after_prompt = re.findall(pattern, row[colname_pred])
    # Some preds are so long that they do not contain the prompt
    if after_prompt:
        return after_prompt[0]
    else:
        return row[colname_pred]

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
    colname_pred = data_config["colname_pred"]
    colname_word = data_config["colname_word"]
    data_lang = data_config["data_lang"]
    preds_path = args.predictions
    scores_path = f"scores/{job_id}-{job_name}.csv"

    data = pd.read_csv(preds_path, lineterminator="\n")
    preds = data.apply(
            remove_prompt, axis=1,
            args=(colname_pred, colname_word, data_lang)
            ).to_list()
    refs = data[data_config["colname_def"]].to_list()

    scores = {
            "word": data[colname_word],
            "example": data[colname_ex],
            "gloss": data[colname_def],
            "prediction": data[colname_pred],
            "sacrebleu": [],
            "rouge_l": [],
            "bertscore_f1": [],
            }

    sacrebleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    if data_lang == "en":
        model_type = "google-bert/bert-base-uncased"
    elif data_lang == "fi":
        model_type = "TurkuNLP/bert-base-finnish-uncased-v1"
    num_layers = bert_score.utils.model2layers["bert-base-uncased"]

    for i in range(len(preds)):
        res = sacrebleu.compute(predictions=[preds[i]], references=[refs[i]])
        scores["sacrebleu"].append(res["score"])
    scores["rouge_l"] = rouge.compute(
            predictions=preds, references=refs,
            use_aggregator=False)["rougeL"]
    scores["bertscore_f1"] = bertscore.compute(
            predictions=preds, references=refs,
            model_type=model_type, num_layers=num_layers)["f1"]

    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(scores_path, index=False)

if __name__ == "__main__":
    main()
