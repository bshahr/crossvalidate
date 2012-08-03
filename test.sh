#!/bin/bash

K=5
NJOBS=1
DATASET=data/digits.npy

for J in 0
do
    python crossvalidate.py $DATASET --n-jobs $NJOBS -j$J -k$K
done
