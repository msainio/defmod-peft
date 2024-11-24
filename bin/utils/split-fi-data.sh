#!/bin/bash

module load python-data
python3 src/split_data.py
module unload python-data
