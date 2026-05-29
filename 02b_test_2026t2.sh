#!/bin/bash

dev_eval=$1

echo "dev_eval = $dev_eval"

if [ "$dev_eval" != "-d" ] && [ "$dev_eval" != "--dev" ]; then
    echo "Usage: bash test.sh -d"
    exit 1
fi

python train.py \
    --dataset AutoTrash \
    --test_only \
    --mono False \
    --score MAHALA