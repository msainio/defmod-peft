#!/bin/bash

datapath=data/cha

for file in CHA_train.csv CHA_valid.csv CHA_test.csv
do
    head -n $(expr $(wc -l $datapath/$file | awk '{print $1}') / 100) \
        $datapath/$file > $datapath/$(echo $file | awk -F '.' '{ print $1 "_mini." $2 }')
done
