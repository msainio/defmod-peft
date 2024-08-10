#!/bin/bash

for file in CHA_train.csv CHA_valid.csv CHA_test.csv
do
    head -n $(expr $(wc -l data/$file | awk '{print $1}') / 100) \
        data/$file > data/$(echo $file | awk -F '.' '{ print $1 "_mini." $2 }')
done
