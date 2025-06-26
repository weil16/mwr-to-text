# Standard library imports
import argparse
import os
from datetime import datetime

# Third-party imports
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer


def run_inference(
    checkpoint_path,
    config,
    MWRDataset,
    MWRModel,
    get_lora_config,
    evaluate,
    load_and_preprocess,
    dataset_name="test",
):
    """Main inference function

    Args:
        config_path: Path to config file
        checkpoint_path: Path to model checkpoint
        dataset_name: Name of dataset to evaluate on ('train', 'val' or 'test')
    """

    # Validate dataset name
    if dataset_name not in ["train", "val", "test"]:
        raise ValueError("dataset_name must be one of: 'train', 'val', 'test'")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize customized tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(config["model"]["tokenizer_path"], legacy=True)
    lora_config = get_lora_config(config)
    model = MWRModel(config=config, lora_config=lora_config).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from {checkpoint_path}")

    # Get dataset path from config
    dataset_path = config["data"]["paths"][dataset_name]

    # Load and preprocess data
    df = load_and_preprocess(dataset_path)

    # Create dataset
    dataset_config = {
        "max_length": config["data"]["preprocessing"]["max_seq_length"],
        "truncation_strategy": config["data"]["preprocessing"]["truncation_strategy"],
    }
    dataset = MWRDataset(df, tokenizer, **dataset_config)

    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    # Setup directories
    os.makedirs(config["training"]["log_dir"], exist_ok=True)
    log_file = os.path.join(
        config["training"]["inference_log_dir"],
        f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    # Run evaluation

    print(f"\n🔍 Running inference on {dataset_name} set ({len(dataset)} samples)")
    results = evaluate(model, data_loader, device, tokenizer)
    print(f"Loss: {results['loss']:.4f}")
    log_message += f", Accuracy: {results['classification']['accuracy']:.4f}"
    log_message += f", F1: {results['classification']['f1']:.4f}"
    log_message += f", Sensitivity: {results['classification']['sensitivity']:.4f}"
    log_message += f", Specificity: {results['classification']['specificity']:.4f}"
    log_message += f", AUC: {results['classification']['auc']:.4f}"
    log_message += f", BERTScore F1: {results['bertscore']['f1']:.4f}"
    log_message += f", METEOR Score: {results['meteor']:.4f}"
    print(log_message)
    with open(log_file, "a") as f:
        f.write(log_message + "\n")

    return results


if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--dataset",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset to evaluate on",
    )
    args = parser.parse_args()

    results = run_inference(args.checkpoint, args.dataset)
