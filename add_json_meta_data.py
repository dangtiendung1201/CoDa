import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='A simple script with command-line arguments.')
parser.add_argument('--dataset', '-d', type=str, help='Path to the input file')
parser.add_argument('--split', '-s', type=str, help='Path to the output file')
args = parser.parse_args()

base_dir = Path(__file__).resolve().parent
dataset_info_path = base_dir / "generation_data" / "dataset_info.json"

with open(dataset_info_path, "r") as json_file:
    data = json.load(json_file)

for suffix in ["_concept_constraint", "_final_constraint_abs", "_solo_constraint", "_abs_prompt_1", "_abs_prompt_2"]:
    file_name = f"{args.dataset}_{args.split}{suffix}"
    data[f"{file_name}"] = {"file_name": f"{file_name}.json"}

with open(dataset_info_path, "w") as f:
    json.dump(data, f, indent=2)