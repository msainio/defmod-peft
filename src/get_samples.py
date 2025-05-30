#!/usr/bin/env python3

import pandas as pd
from pathlib import Path


def main():
    scores_dir = Path("scores")
    num_items = 5  # Items to sample per quantile
    random_state = 42
    
    for file in scores_dir.glob("*.csv"):
        key = str(file.name).split("-", 2)[-1].rstrip(".csv")
        df = pd.read_csv(file, lineterminator="\n")
        df["rougeL"] = df["rougeL"].map(lambda x: 100 * x)
        df["bertF1"] = df["bertF1"].map(lambda x: 100 * x)

        sub_dfs = []
        for metric in df.select_dtypes("number").columns:
            quantiles = df[metric].quantile(
                    [0.0, 0.25, 0.5, 0.75, 1.0]
                    ).to_list()
            for low, high in zip(quantiles[:-1], quantiles[1:]):
                sub_df = df[(df[metric] >= low) & (df[metric] < high)]
                if sub_df.shape[0] < num_items:
                    sub_dfs.append(
                            sub_df.sample(
                                sub_df.shape[0],
                                random_state=random_state,
                                )
                            )
                elif sub_df.shape[0] == 0:
                    sub_dfs.append(
                            df.sample(num_items, random_state=random_state)
                            )
                else:
                    sub_dfs.append(
                            sub_df.sample(
                                num_items,
                                random_state=random_state,
                                )
                            )

        sample_to_annotate = pd.concat(sub_dfs).drop_duplicates()
        desired_size = (
                num_items
                * 4  # quantiles
                * len(df.select_dtypes("number").columns)  # metrics
                )

        # Sample more items until desired number of unique items is reached
        iter_state = 42
        while sample_to_annotate.shape[0] < desired_size:
            difference = desired_size - sample_to_annotate.shape[0]
            addl_sample = df.sample(difference, random_state=iter_state)
            sample_to_annotate = pd.concat([sample_to_annotate, addl_sample])
            iter_state += 1

        sample_to_annotate[
                ["word", "example", "target", "prediction"]
                ].to_csv(f"annotation/{key}_only_text.csv", index=False)
        sample_to_annotate.to_csv(f"annotation/{key}_scores.csv", index=False)


if __name__ == "__main__":
    main()
