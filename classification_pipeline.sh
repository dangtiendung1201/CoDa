#!/usr/bin/env sh
set -eu

if [ "$#" -lt 4 ]; then
        echo "Usage: sh classification_pipeline.sh <dataset_name> <dataset_split> <debug_mode> <dataset_split_for_shortcut_grammar>"
        exit 1
fi

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if command -v module >/dev/null 2>&1; then
        module load cuda || true
fi

if ! "$PYTHON_BIN" -c "import nltk" >/dev/null 2>&1; then
        echo "Missing Python dependency: nltk"
        echo "Install it in your active environment with: python -m pip install nltk"
        exit 1
fi

mkdir -p "$SCRIPT_DIR/generation_data" "$SCRIPT_DIR/tsv_data/out_data/$1"

cd "$SCRIPT_DIR/ShortcutGrammar"

cp "$SCRIPT_DIR/tsv_data/inp_data/$1_$2.tsv" "$SCRIPT_DIR/ShortcutGrammar/data/$1/test.tsv"

"$PYTHON_BIN" conv_label.py -d $1 -s test

CKPT_DIR="$SCRIPT_DIR/ShortcutGrammar/output/$1/pcfg"
if [ ! -f "$CKPT_DIR/tokenizer.json" ] || [ ! -f "$CKPT_DIR/model.pt" ]; then
        echo "Missing ShortcutGrammar checkpoint for dataset '$1' at $CKPT_DIR"
        echo "Bootstrapping a small PCFG checkpoint so the pipeline can proceed..."
        "$PYTHON_BIN" src/run.py \
                --train_on "$1" \
                --model_type pcfg \
                --max_length 128 \
                --max_eval_length 128 \
                --min_length 1 \
                --use_product_length 0 \
                --steps "${PCFG_BOOTSTRAP_STEPS:-300}" \
                --patience 2 \
                --eval_every 1024 \
                --eval_on "$1" \
                --eval_on_train_datasets "$1" \
                --train_batch_size 2 \
                --eval_batch_size 2 \
                --criterion loss \
                --learning_rate 1e-3 \
                --gradient_accumulation_steps 1 \
                --save \
                --preterminals 64 \
                --nonterminals 32 \
                --output_dir "output/$1/pcfg"
fi

sh eval.sh $1 $4

"$PYTHON_BIN" sst2_features_final.py -d $1 -s $2 -k 3

cd "$SCRIPT_DIR"

"$PYTHON_BIN" "$SCRIPT_DIR/generate_data_pipe.py" -m meta-llama/Llama-2-13b-chat-hf -c $1_$2_concept_constraint -nr 1

"$PYTHON_BIN" "$SCRIPT_DIR/Lexical-Constraints/lexical_constraints_exemplars_without_pos_and_concepts.py" \
                                -i "$SCRIPT_DIR/tsv_data/inp_data/$1_$2.tsv" \
                                -p "$SCRIPT_DIR/tsv_data/out_data" \
                                -j "$SCRIPT_DIR/generation_data/$1_$2_concept_constraint/generated_predictions.jsonl" \
        -ds $1 \
        -o $1_$2_final_constraint\
        -d $3

"$PYTHON_BIN" "$SCRIPT_DIR/add_json_meta_data.py" -d $1 -s $2
echo "Added metadata for: $1_$2"

"$PYTHON_BIN" "$SCRIPT_DIR/tsv_to_jsonl.py" -d $1 -s $2
echo "Solo json created for: $1_$2"

cp "$SCRIPT_DIR/tsv_data/out_data/$1/$1_$2_solo_constraint.json" "$SCRIPT_DIR/generation_data/"
cp "$SCRIPT_DIR/tsv_data/out_data/$1/$1_$2_final_constraint_abs.json" "$SCRIPT_DIR/generation_data/"

"$PYTHON_BIN" "$SCRIPT_DIR/generate_data_pipe.py" -m meta-llama/Llama-2-13b-chat-hf -c $1_$2_final_constraint_abs -nr 1

"$PYTHON_BIN" "$SCRIPT_DIR/generate_data_pipe.py" -m meta-llama/Llama-2-13b-chat-hf -c $1_$2_solo_constraint -nr 3

"$PYTHON_BIN" "$SCRIPT_DIR/abs_format.py" -d $1 -s $2

echo "$1"
"$PYTHON_BIN" "$SCRIPT_DIR/generate_data_pipe.py" -m meta-llama/Llama-2-13b-chat-hf -c $1_$2_abs_prompt_1 -nr 1

echo "$2"
"$PYTHON_BIN" "$SCRIPT_DIR/generate_data_pipe.py" -m meta-llama/Llama-2-13b-chat-hf -c $1_$2_abs_prompt_2 -nr 1

echo "Creating final aug file."
"$PYTHON_BIN" "$SCRIPT_DIR/json_to_tsv.py" -d $1 -s $2