#!/usr/bin/env python3
"""
Main entry point for MWR project
Provides unified interface for training and inference
"""
# Standard library imports
import argparse
import sys
from pathlib import Path

# Third-party imports
import yaml
import torch

# Local application imports
from src.dataset import MWRDataset
from src.inference import run_inference
from src.model import MWRModel, get_lora_config
from src.train import train, evaluate
from utils.data_processing import load_and_preprocess
from utils.data_splitter import split_data
from utils.test_pipeline import run_full_test_suite, print_success

def load_config():
    """Load configuration from yaml file"""
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Ensure numeric values are properly typed
    config["training"]["learning_rate"] = float(config["training"]["learning_rate"])
    config["training"]["batch_size"] = int(config["training"]["batch_size"])
    config["training"]["epochs"] = int(config["training"]["epochs"])
    config["training"]["gradient_accumulation_steps"] = int(config["training"]["gradient_accumulation_steps"])
    
    return config

def main():
    # Load config
    config = load_config()

    # Set up argument parser
    parser = argparse.ArgumentParser(description='MWR Project Main Entry')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Training subcommand
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                              help="Path to checkpoint file to resume training from")
    
    # Inference subcommand
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    infer_parser.add_argument("--dataset", type=str, default='val', 
                              choices=['train', 'val', 'test'], help="Dataset to evaluate on")
    
    # Test subcommand
    test_parser = subparsers.add_parser('debug', help='Run full test suite')

    args = parser.parse_args()

    # Execute the requested command
    # Training mode
    if args.command == 'train':
        # Ensure numeric values are properly typed
        config["training"]["learning_rate"] = float(config["training"]["learning_rate"])
        config["training"]["batch_size"] = int(config["training"]["batch_size"])
        config["training"]["epochs"] = int(config["training"]["epochs"])
        config["training"]["gradient_accumulation_steps"] = int(config["training"]["gradient_accumulation_steps"])

        if args.resume_from_checkpoint:
            checkpoint = torch.load(args.resume_from_checkpoint)
            config = checkpoint["config"]
            print(f"\n🔁 从检查点恢复训练: {args.resume_from_checkpoint}")
            print(f"   已训练epoch数: {checkpoint['epoch'] - 1}")

        train(args, config, MWRDataset, MWRModel, get_lora_config, load_and_preprocess, split_data)   
    # Inference mode
    elif args.command == 'infer':
        run_inference(args.checkpoint, config, MWRDataset, MWRModel, get_lora_config, evaluate, load_and_preprocess, args.dataset)
    # Debug mode
    elif args.command == 'debug':
        
        from src.model import MWRModel, get_lora_config
        from src.train import evaluate
        print("\nStarting comprehensive pipeline test...")
        success = run_full_test_suite(MWRDataset, MWRModel, get_lora_config, evaluate, load_and_preprocess)

        if success:
            print_success("All components are working correctly")
        else:
            print("[WARNING] Some tests failed, please check the output")
            
        print("\nTest completed.")
    # Invalid command
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)

if __name__ == '__main__':
    main()
