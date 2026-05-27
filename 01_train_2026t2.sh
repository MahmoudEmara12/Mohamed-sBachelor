#!/bin/bash

echo "Training AutoTrash dataset"
echo "--------------------------"

DATASET="DCASE2026T2AutoTrash"

BASE_DIR="D:/MohamedBach/dcase2023_task2_baseline_ae/data/dcase2026t2/dev_data/raw/AutoTrash"

TRAIN_DIR="${BASE_DIR}/train"
TEST_DIR="${BASE_DIR}/test"
ATTR_FILE="${BASE_DIR}/attributes_00.csv"

echo "Train: $TRAIN_DIR"
echo "Test : $TEST_DIR"
echo "Attr : $ATTR_FILE"

DEV_EVAL="-d"

# IMPORTANT: pass boolean as lowercase string (safer for argparse)
MONO="True"

bash train_ae.sh \
    "${DATASET}" \
    "${DEV_EVAL}" \
    "${MONO}" \
    0 