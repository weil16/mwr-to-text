# Standard library imports
from pathlib import Path
import pandas as pd
import sys

# Third-party imports
import yaml
from transformers import T5TokenizerFast

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.dataset import MWRDataset
from utils.data_processing import load_and_preprocess

# Load config
config_path = project_root / "config" / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Load tokenizer
tokenizer = T5TokenizerFast.from_pretrained(config["model"]["tokenizer_path"], legacy=True)


# Load and preprocess data
df = load_and_preprocess(config["data"]["paths"]["input"])

# Initialize dataset
dataset = MWRDataset(df, tokenizer)

# Analyze token counts
over_limit = 0
token_counts = []

print("\nAnalyzing token counts for all input texts:")
for i in range(len(dataset)):
    input_text = dataset.input_texts[i]
    tokens = tokenizer.tokenize(input_text)
    token_count = len(tokens)
    token_counts.append(token_count)

    if token_count > 512:
        over_limit += 1
        print(f"Sample {i}: Token count = {token_count + 2}")

print(f"\nSummary:")
print(f"Total samples: {len(dataset)}")
print(f"Samples over 512 tokens: {over_limit}")
print(f"Max token count: {max(token_counts) if token_counts else 0}")
print(f"Min token count: {min(token_counts) if token_counts else 0}")
print(f"Average token count: {sum(token_counts)/len(token_counts) if token_counts else 0:.2f}")
