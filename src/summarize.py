#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys

def main():
    results = pd.read_csv(sys.argv[1], lineterminator="\n")
    summary = {
            "metric": ["sacrebleu", "rouge_l", "bertscore_f1"],
            "mean": [],
            "std": []
            }
    for m in summary["metric"]:
        summary["mean"].append(np.mean(results[m]))
        summary["std"].append(np.std(results[m]))
    print(pd.DataFrame(summary).to_string(index=False))

if __name__ == "__main__":
    main()
