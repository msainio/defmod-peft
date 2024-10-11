#!/bin/bash

datapath=data/codwoe

for file in train.csv val.csv test.csv
do
    head -n $(expr $(wc -l $datapath/$file | awk '{print $1}') / 100) \
        $datapath/$file > $datapath/$(echo $file | awk -F '.' '{ print $1 "_mini." $2 }')
done
