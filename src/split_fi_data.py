#!/usr/bin/env python3

import numpy as np
import pandas as pd

def main():
    df = pd.read_csv("data/dbnary/dbnary_fi.csv")

    total_size = df.shape[0]
    train_size = int(total_size * 0.8)
    test_size = (total_size - train_size) // 2
    train_size += (total_size - train_size) % 2

    train_dataset, val_dataset, test_dataset = np.split(
            df.sample(frac=1, random_state=1),
            [train_size, train_size + test_size])

    train_dataset.to_csv("data/dbnary/dbnary_fi_train.csv")
    val_dataset.to_csv("data/dbnary/dbnary_fi_val.csv")
    test_dataset.to_csv("data/dbnary/dbnary_fi_test.csv")

if __name__ == "__main__":
    main()
