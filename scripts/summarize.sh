#!/bin/bash

module purge
module load python-data

scores_dir="scores"
summary_file="results/summary.txt"

if [[ -f $summary_file ]]
then
    rm $summary_file
fi

for file in $(ls $scores_dir)
do
    echo "$scores_dir/$file" >> $summary_file
    python3 src/summarize.py "$scores_dir/$file" >> $summary_file
    echo >> $summary_file
done
