#!/bin/bash

K=5
NJOBS=2
DATASET=data/digits.npy

for J in {0..1}
do
    python crossvalidate.py $DATASET --n-jobs $NJOBS -j$J -k$K
done
