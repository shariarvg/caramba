#!/bin/sh

arg1=$1
arg2=$2
arg3=$3
arg4=$4

python3 inference.py $arg1 $arg2 $arg4
python3 train_predictor.py $arg4  
python3 evaluate_perplexity.py $arg2 $arg3 $arg4