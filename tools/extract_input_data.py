import sys
from pathlib import Path
import yaml
from transformers import T5Tokenizer

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.dataset import MWRDataset
from utils.data_processing import load_and_preprocess


def main():
    # Load config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize tokenizer
    tokenizer = T5Tokenizer.from_pretrained(config["model"]["name"], legacy=True)

    # Load and preprocess data
    df = load_and_preprocess(config["data"]["paths"]["input"])

    # Create dataset
    dataset = MWRDataset(
        df=df,
        tokenizer=tokenizer,
        max_length=config["data"]["preprocessing"]["max_seq_length"],
        truncation_strategy=config["data"]["preprocessing"]["truncation_strategy"],
    )

    # Extract input_texts
    input_texts = dataset.input_texts
    print(f"Extracted {len(input_texts)} input texts")
    for i, text in enumerate(input_texts[:5]):  # Print first 5 as example
        print(f"\nSample {i+1}:")
        print(text)


if __name__ == "__main__":
    main()
