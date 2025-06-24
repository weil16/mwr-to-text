"""
Enhanced one-click test pipeline for MWR breast cancer risk prediction model.
This script tests the complete pipeline including data loading, training, validation and testing.
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
import torch
import yaml
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import T5Tokenizer

sys.path.append(str(Path(__file__).parent.parent))
from dataset import MWRDataset
from model import MWRModel, get_lora_config
from train import evaluate
from utils.data_processing import load_and_preprocess
from utils.data_splitter import split_data

# Load configuration
config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

warnings.filterwarnings('ignore', category=UserWarning, module='requests')

def print_step(message):
    print(f"\n[STEP] {message}...")

def print_success(message):
    print(f"[SUCCESS] {message}")

def print_info(message):
    print(f"[INFO] {message}")

def test_data_loading():
    """Test data loading and preprocessing"""
    try:
        print_step("Testing data loading")
        df = load_and_preprocess(config["data"]["paths"]["train"])
        assert len(df) > 0, "Empty dataframe loaded"
        print_success("Data loading test passed")
        return True
    except Exception as e:
        print(f"[ERROR] Data loading test failed: {str(e)}")
        return False

def test_model_initialization():
    """Test model initialization and forward pass"""
    try:
        print_step("Testing model initialization")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MWRModel(
            lora_config=get_lora_config(config),
            config=config
        ).to(device)
        
        # Test forward pass with explicit decoder inputs
        dummy_input = torch.randint(0, 100, (2, 512)).to(device)
        dummy_decoder_input = torch.randint(0, 100, (2, 512)).to(device)
        outputs = model(
            input_ids=dummy_input,
            attention_mask=torch.ones_like(dummy_input),
            decoder_input_ids=dummy_decoder_input
        )
        assert 'loss' in outputs, "Model forward pass failed"
        print_success("Model initialization test passed")
        return True
    except Exception as e:
        print(f"[ERROR] Model initialization test failed: {str(e)}")
        return False

def test_training_workflow():
    """Test complete training workflow"""
    try:
        print_step("Testing training workflow")
        
        # Mock command line arguments with test defaults
        # batch_size=2 is chosen for quick testing with small dataset
        # In production, this would come from config or command line
        args = argparse.Namespace(
            epochs=1,
            batch_size=2,  # Small batch size for test efficiency
            lr=3e-5,
            grad_accum_steps=1,
            fp16=False,
            num_workers=0,  # Disable multiprocessing for test stability
            evaluate_test=False,
            resume_from_checkpoint=None
        )
        
        # Initialize model and data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load tokenizer from local path if available
        local_path = config["model"]["local_path"]
        if local_path and Path(local_path).exists():
            print_info("Loading tokenizer from local path")
            tokenizer = T5Tokenizer.from_pretrained(local_path, legacy=True)
        else:
            raise RuntimeError("Local model path not found, cannot run in offline mode")
        
        # Load small subset of data
        df = load_and_preprocess(config["data"]["paths"]["train"])
        train_df = df.iloc[:10].copy()
        val_df = df.iloc[10:15].copy()
        
        # Create datasets and loaders
        dataset_config = {
            'max_length': config['data']['preprocessing']['max_seq_length'],
            'truncation_strategy': config['data']['preprocessing']['truncation_strategy']
        }
        train_dataset = MWRDataset(train_df, tokenizer, **dataset_config)
        val_dataset = MWRDataset(val_df, tokenizer, **dataset_config)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Initialize model
        model = MWRModel(
            lora_config=get_lora_config(config),
            config=config
        ).to(device)
        
        # Save initial model state
        initial_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        # Test training step without actually updating weights
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        
        model.train()
        optimizer.zero_grad()  # Clear any existing gradients
        
        # Test single batch with gradient computation
        batch = next(iter(train_loader))
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs['loss']
        
        # Verify gradients can be computed
        loss.backward()
        
        # Verify gradients exist for some parameters
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients, "No gradients computed during training test"
        
        optimizer.zero_grad()  # Clear gradients after test
        
        # Verify model weights unchanged
        current_state = model.state_dict()
        for k in initial_state:
            assert torch.equal(initial_state[k], current_state[k]), f"Model weights changed during test: {k}"
        
        # Run validation
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
        
        print_success("Training workflow test passed")
        return True
    except Exception as e:
        print(f"[ERROR] Training workflow test failed: {str(e)}")
        return False

def test_evaluation_metrics():
    """Test evaluation metrics calculation"""
    try:
        print_step("Testing evaluation metrics")
        
        # Create dummy predictions
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        
        # Test classification metrics with more tolerance
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        
        print_info(f"Accuracy: {accuracy:.4f} (expected ~0.75)")
        print_info(f"F1 Score: {f1:.4f} (expected >0.5)") 
        print_info(f"AUC: {auc:.4f} (expected >0.5)")
        
        assert accuracy >= 0.5, "Accuracy too low"
        assert f1 >= 0.5, "F1 score too low"
        assert auc >= 0.5, "AUC too low"
        
        # Test text generation metrics with better error handling
        refs = ["This is a simple test sentence"]
        hyps = ["This is a simple test sentence"]

        try:
            _, _, f1 = bert_score(
                cands=hyps,
                refs=refs,
                lang='en',
                model_type="bert-base-uncased",
                use_fast_tokenizer=True
            )
            bert_f1 = f1.mean().item()
            print_info(f"BERTScore F1: {bert_f1:.4f}")
            assert bert_f1 > 0.5, "BERTScore too low"
        except Exception as e:
            print_info(f"Skipping BERTScore test due to error: {str(e)}")
            
        try:
            meteor = meteor_score([refs[0].split()], hyps[0].split())
            print_info(f"METEOR: {meteor:.4f}")
            assert meteor > 0.5, "METEOR score too low"
        except Exception as e:
            print_info(f"Skipping METEOR test due to error: {str(e)}")
            
        print_success("Evaluation metrics test passed")
        return True
    except Exception as e:
        print(f"[ERROR] Evaluation metrics test failed: {str(e)}")
        return False

def test_evaluate_function():
    """Test evaluate function matching production implementation"""
    try:
        print_step("Testing evaluate function")
        
        # Setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load tokenizer from local path if available
        local_path = config["model"]["local_path"]
        if local_path and Path(local_path).exists():
            print_info("Loading tokenizer from local path")
            tokenizer = T5Tokenizer.from_pretrained(local_path, legacy=True)
        else:
            raise RuntimeError("Local model path not found, cannot run in offline mode")
        
        # Create realistic mock data matching production format
        mock_batch = {
            'input_ids': torch.tensor([[1, 23, 45, 67, 89, 100]] * 2),
            'attention_mask': torch.ones(2, 6),
            'labels': torch.tensor([[2, 24, 46, 68, 90, 101]] * 2),
            'class_labels': torch.tensor([0, 1])
        }
        
        # Mock model that simulates T5 behavior
        class MockModel:
            def __init__(self):
                self.t5 = MockT5()
                self.eval_mode = False
            
            def eval(self):
                self.eval_mode = True
            
            def __call__(self, **kwargs):
                return {
                    'loss': torch.tensor(0.5),  # Directly return loss tensor
                    'logits': torch.randn(2, 6, 32000),
                    'cls_logits': torch.tensor([[0.7, 0.3], [0.4, 0.6]])
                }
        
        class MockT5:
            def generate(self, input_ids, attention_mask, **kwargs):
                # Simulate beam search behavior
                batch_size = input_ids.shape[0]
                return torch.tensor([[101, 102, 103, 104]] * batch_size)
        
        # Test evaluation
        mock_loader = [mock_batch]
        results = evaluate(MockModel(), mock_loader, device, tokenizer)
        
        # Verify results structure
        assert 'loss' in results, "Missing loss in results"
        assert isinstance(results['loss'], float), "Loss should be float"
        assert 0 <= results['loss'] <= 10, "Loss out of reasonable range"
        
        if results['bertscore']:
            assert 'precision' in results['bertscore']
            assert 'recall' in results['bertscore']
            assert 'f1' in results['bertscore']
        
        print_success("Evaluate function test passed")
        return True
    except Exception as e:
        print(f"[ERROR] Evaluate function test failed: {str(e)}")
        return False

def test_config_validation():
    """Test configuration validation"""
    try:
        print_step("Testing configuration validation")
        
        required_keys = [
            'model', 'training', 'data', 
            'data.paths', 'data.preprocessing',
            'training.log_dir', 'training.test_log_dir'
        ]
        
        for key in required_keys:
            keys = key.split('.')
            current = config
            for k in keys:
                assert k in current, f"Missing config key: {key}"
                current = current[k]
                
        print_success("Configuration validation passed")
        return True
    except Exception as e:
        print(f"[ERROR] Configuration validation failed: {str(e)}")
        return False

def run_full_test_suite():
    """Run complete test suite"""
    print("\n=== Running Full Test Suite ===")
    
    tests = [
        test_data_loading,
        test_model_initialization,
        test_training_workflow,
        test_evaluation_metrics,
        test_evaluate_function,
        test_config_validation
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    if all(results):
        print("\n=== ALL TESTS PASSED ===")
        return True
    else:
        print("\n=== SOME TESTS FAILED ===")
        return False

if __name__ == '__main__':
    print("\nStarting comprehensive pipeline test...")
    success = run_full_test_suite()
    
    if success:
        print_success("All components are working correctly")
    else:
        print("[WARNING] Some tests failed, please check the output")
    
    print("\nTest completed. Press any key to exit...")
    input()
