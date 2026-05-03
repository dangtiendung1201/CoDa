#!/usr/bin/env sh

DATASET="$1"
EVAL_SPLIT="${2:-test}"

TEST_RUN=0
if [ "$EVAL_SPLIT" = "test" ]; then
  TEST_RUN=1
fi

python src/run.py  \
   --eval_on "$DATASET" \
   --test_run "$TEST_RUN"\
  --load_from "./output/$DATASET/pcfg/" \
   --model_type pcfg \
   --preterminals 64 \
   --nonterminals 32 \
   --eval_batch_size 1 \
  --output_dir "output/$DATASET/" \
   --max_length 128 \
   --max_eval_length 128 \
   --cache "";

   # --load_from output/$1/pcfg/ \
