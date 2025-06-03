#!/usr/bin/env python3

import pandas as pd
from pathlib import Path


def main():
    annotation_dir = Path("annotation")

    data = {}
    keys = {
            str(file.name).split("_", 2)[0].rstrip(".csv")
            for file in annotation_dir.glob("*.csv")
            }

    for key in keys:
        only_text = pd.read_csv(f"annotation/{key}_only_text.csv", lineterminator="\n")
        scores = pd.read_csv(
                f"annotation/{key}_scores.csv",
                lineterminator="\n",
                ).drop(columns=["fluent", "correct"])
        df = pd.concat((scores, only_text), axis=1)
        data[key] = df

    df_all = pd.concat(data)
    df_mean = df_all.groupby(level=0).mean(numeric_only=True).round(2)
    df_std = df_all.groupby(level=0).std(numeric_only=True).round(2)

    df_table = (
            df_mean.map(lambda x: str(x))
            + "±"
            + df_std.map(lambda x: str(x))
            )
    df_table.columns = ["Fluent", "Correct", "chrF++", "BLEU", "ROUGE", "BERTScore"]

    print(df_table.to_latex())


if __name__ == "__main__":
    main()
