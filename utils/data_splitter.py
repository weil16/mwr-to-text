# Standard library imports
import os
from pathlib import Path

# Third-party imports
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Load configuration from yaml file
config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

def split_data() -> None:
    """
    Split data into train/val/test sets according to config.yaml specifications
    """
    # Load raw data from config path
    df = pd.read_csv(config["data"]["paths"]["input"])
    
    # First split: train + (val + test)
    train_df, remaining_df = train_test_split(
        df,
        train_size=config["data"]["split"]["train_size"],
        random_state=config["data"]["split"]["random_state"]
    )
    
    # Second split: val and test
    val_ratio = config["data"]["split"]["val_size"] / (config["data"]["split"]["val_size"] + config["data"]["split"]["test_size"])
    val_df, test_df = train_test_split(
        remaining_df,
        train_size=val_ratio,
        random_state=config["data"]["split"]["random_state"]
    )
    
    # Save splits
    os.makedirs(os.path.dirname(config["data"]["paths"]["train"]), exist_ok=True)
    train_df.to_csv(config["data"]["paths"]["train"], index=False)
    val_df.to_csv(config["data"]["paths"]["val"], index=False)
    test_df.to_csv(config["data"]["paths"]["test"], index=False)

if __name__ == '__main__':
    split_data()
