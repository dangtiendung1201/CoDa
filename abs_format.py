import pandas as pd
import json
import re
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='A simple script with command-line arguments.')
parser.add_argument('--dataset', '-d', type=str, help='Path to the input file')
parser.add_argument('--split', '-s', type=str, help='Path to the output file')
args = parser.parse_args()
base_dir = Path(__file__).resolve().parent

df = pd.read_csv(
    base_dir / "tsv_data" / "out_data" / args.dataset / f"{args.dataset}_{args.split}_final_constraint.tsv",
    sep="\t",
    header=0,
)

mix_prompts_1 = []
mix_prompts_2 = []

json_data = [[],[]]

with open(base_dir / "generation_data" / f"{args.dataset}_{args.split}_final_constraint_abs" / "generated_predictions.jsonl", "r") as file:
    i = 0

    for line in file:
        obj = json.loads(line)
        if len(obj["predict"])>0:
            predict_split = obj["predict"].split(":")
            if len(predict_split) > 1:
                predict_split = predict_split[1].split("\n\n")
                for text in predict_split:
                    if len(text.strip())>0:
                        predict_split = text
                        break
            else:
                predict_split = ""       
            abstract = str(predict_split)
            if len(abstract) > 0:
                for k in range(1,3):
                    json_entry = {
                        "instruction": df.at[i, "mix_const_" + str(k)].replace("$abs$", abstract),
                        "input": df.at[i, "label"],
                        "output": df.at[i, "label"]
                    }
                    json_data[k-1].append(json_entry)
        i = i + 1

for k in range(1,3):
    with open(base_dir / "generation_data" / f"{args.dataset}_{args.split}_abs_prompt_{k}.json", "w") as json_file:
        json.dump(json_data[k-1], json_file, indent=2)