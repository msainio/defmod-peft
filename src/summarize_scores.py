#!/usr/bin/env python3

import pandas as pd
from pathlib import Path


def main():
    scores_dir = Path("scores")

    data = {}
    for file in scores_dir.glob("*.csv"):
        key = str(file.name).split("-", 2)[-1].rstrip(".csv")
        df = pd.read_csv(file, lineterminator="\n")
        df["rougeL"] = df["rougeL"].map(lambda x: 100 * x)
        df["bertF1"] = df["bertF1"].map(lambda x: 100 * x)
        data[key] = df

    df_all = pd.concat(data)
    df_mean = df_all.groupby(level=0).mean(numeric_only=True).round(2)
    df_std = df_all.groupby(level=0).std(numeric_only=True).round(2)

    df_table = (
            df_mean.map(lambda x: str(x))
            + "±"
            + df_std.map(lambda x: str(x))
            )
    df_table.columns = ["chrF++", "BLEU", "ROUGE", "BERTScore"]

    print(df_table.to_latex())


if __name__ == "__main__":
    main()
