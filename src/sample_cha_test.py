#!/usr/bin/env python3

import pandas as pd

def main():
    cha_test_path = "data/cha/CHA_test.csv"
    sample_path = "data/cha/CHA_test_sample.csv"
    df = pd.read_csv(cha_test_path, lineterminator="\n")
    sample_df = df.sample(frac=0.02, random_state=1)
    sample_df.to_csv(sample_path, index=False)

if __name__ == "__main__":
    main()
