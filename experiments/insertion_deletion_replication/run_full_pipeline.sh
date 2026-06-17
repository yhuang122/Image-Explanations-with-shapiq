#!/bin/bash
set -e

RESULTS=XXXX/Image-Explanations-with-shapiq/results
MODEL=openai/clip-vit-base-patch16
ROOT=XXXX/Image-Explanations-with-shapiq

echo "=== Step 0: copy mscoco_predictions.csv ===" && date
mkdir -p $RESULTS/$MODEL
cp $ROOT/results/$MODEL/mscoco_predictions.csv $RESULTS/$MODEL/

echo "=== Step 1a: explain shapley ===" && date
cd $ROOT/experiments
python explain_mscoco.py --model_name $MODEL \
    --path_input $RESULTS/$MODEL \
    --path_output $RESULTS/$MODEL/shapley \
    --start 0 --stop 1000 --mode shapley --p_sampler 0.5 --budget 256 --batch_size 64 --random_state 0

echo "=== Step 1b: explain banzhaf 0.3 ===" && date
python explain_mscoco.py --model_name $MODEL \
    --path_input $RESULTS/$MODEL \
    --path_output $RESULTS/$MODEL/banzhaf/0.3 \
    --start 0 --stop 1000 --mode banzhaf --p_sampler 0.3 --budget 256 --batch_size 64 --random_state 0

echo "=== Step 1c: explain banzhaf 0.5 ===" && date
python explain_mscoco.py --model_name $MODEL \
    --path_input $RESULTS/$MODEL \
    --path_output $RESULTS/$MODEL/banzhaf/0.5 \
    --start 0 --stop 1000 --mode banzhaf --p_sampler 0.5 --budget 256 --batch_size 64 --random_state 0

echo "=== Step 1d: explain banzhaf 0.7 ===" && date
python explain_mscoco.py --model_name $MODEL \
    --path_input $RESULTS/$MODEL \
    --path_output $RESULTS/$MODEL/banzhaf/0.7 \
    --start 0 --stop 1000 --mode banzhaf --p_sampler 0.7 --budget 256 --batch_size 64 --random_state 0

echo "=== Step 2: insertion/deletion ===" && date
cd $ROOT
python experiments/migrated/insertion_deletion.py \
    --model_name $MODEL \
    --path_input $RESULTS \
    --path_output $RESULTS \
    --start 0 --stop 1000

echo "=== Done ===" && date
