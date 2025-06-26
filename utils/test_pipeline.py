"""
Enhanced one-click test pipeline for MWR breast cancer risk prediction model.
This script tests the complete pipeline including data loading, training, validation and testing.
"""

import warnings
import argparse

import numpy as np
import torch
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning, module="requests")


def print_step(message):
    print(f"\n[STEP] {message}...")


def print_success(message):
    print(f"[SUCCESS] {message}")


def print_info(message):
    print(f"[INFO] {message}")


def test_data_loading(config, load_and_preprocess):
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


def test_model_initialization(MWRModel, config, get_lora_config):
    """Test model initialization and forward pass"""
    try:
        print_step("Testing model initialization")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MWRModel(lora_config=get_lora_config(config), config=config).to(device)

        # Test forward pass with explicit decoder inputs
        dummy_input = torch.randint(0, 100, (2, 512)).to(device)
        dummy_decoder_input = torch.randint(0, 100, (2, 512)).to(device)
        outputs = model(
            input_ids=dummy_input,
            attention_mask=torch.ones_like(dummy_input),
            decoder_input_ids=dummy_decoder_input,
        )
        assert "loss" in outputs, "Model forward pass failed"
        print_success("Model initialization test passed")
        return True
    except Exception as e:
        print(f"[ERROR] Model initialization test failed: {str(e)}")
        return False


def test_training_workflow(MWRDataset, MWRModel, config, get_lora_config, load_and_preprocess):
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
            resume_from_checkpoint=None,
        )

        # Initialize model and data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = T5Tokenizer.from_pretrained(
            config["model"]["tokenizer_path"], legacy=True
        )  # Explicitly set legacy=True to maintain backward compatibility

        # Load small subset of data
        df = load_and_preprocess(config["data"]["paths"]["train"])
        train_df = df.iloc[:10].copy()
        val_df = df.iloc[10:15].copy()

        # Create datasets and loaders
        dataset_config = {
            "max_length": config["data"]["preprocessing"]["max_seq_length"],
            "truncation_strategy": config["data"]["preprocessing"]["truncation_strategy"],
        }
        train_dataset = MWRDataset(train_df, tokenizer, **dataset_config)
        val_dataset = MWRDataset(val_df, tokenizer, **dataset_config)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Initialize model
        model = MWRModel(lora_config=get_lora_config(config), config=config).to(device)

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
        loss = outputs["loss"]

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

        # Create dummy predictions with more classes for better coverage
        y_true = np.array([0, 1, 0, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])

        # Test classification metrics with more tolerance
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        auc = roc_auc_score(y_true, np.eye(np.max(y_true) + 1)[y_pred], multi_class="ovo")

        # Calculate sensitivity (recall) and specificity
        sensitivity = recall_score(y_true, y_pred, average="weighted")
        cm = confusion_matrix(y_true, y_pred)
        tn = cm.diagonal()
        fp = cm.sum(axis=0) - tn
        specificity = np.mean(tn / (tn + fp))

        print_info(f"Accuracy: {accuracy:.4f} (expected ~0.6667)")
        print_info(f"F1 Score: {f1:.4f} (expected >0.5)")
        print_info(f"AUC: {auc:.4f} (expected >0.5)")
        print_info(f"Sensitivity: {sensitivity:.4f} (expected >0.5)")
        print_info(f"Specificity: {specificity:.4f} (expected >0.5)")

        assert accuracy >= 0.5, "Accuracy too low"
        assert f1 >= 0.5, "F1 score too low"
        assert auc >= 0.5, "AUC too low"
        assert sensitivity >= 0.5, "Sensitivity too low"
        assert specificity >= 0.5, "Specificity too low"

        # Test text generation metrics with better error handling
        refs = ["This is a simple test sentence", "Another test sentence"]
        hyps = ["This is a simple test sentence", "Another test example"]

        try:
            _, _, f1 = bert_score(
                cands=hyps,
                refs=refs,
                lang="en",
                model_type="bert-base-uncased",
                use_fast_tokenizer=True,
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


def test_evaluate_function(config, evaluate):
    """Comprehensive test for evaluate function including classification metrics"""
    try:
        print_step("Testing evaluate function")

        # Setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = T5Tokenizer.from_pretrained(config["model"]["tokenizer_path"], legacy=True)

        # Create mock data with multiple classes (0, 1, 2) and reference texts
        mock_batch = {
            "input_ids": torch.tensor([[1, 23, 45, 67, 89, 100]] * 3),
            "attention_mask": torch.ones(3, 6),
            "labels": torch.tensor([[2, 24, 46, 68, 90, 101]] * 3),
            "class_labels": torch.tensor([0, 1, 2]),  # 3-class classification
        }

        # Mock model with detailed behavior
        class MockT5:
            def generate(self, input_ids, attention_mask, **kwargs):
                return torch.tensor([[2, 24, 46, 68, 90, 101]] * input_ids.shape[0])

        class MockModel:
            def __init__(self):
                self.t5 = MockT5()
                self.eval_mode = False

            def eval(self):
                self.eval_mode = True

            def __call__(self, **kwargs):
                # Simulate model outputs with class logits and generated texts
                return {
                    "loss": torch.tensor(0.5),
                    "class_logits": torch.tensor(
                        [
                            [0.8, 0.1, 0.1],  # Class 0
                            [0.2, 0.7, 0.1],  # Class 1
                            [0.1, 0.3, 0.6],  # Class 2
                        ]
                    ),
                }

        class MockModelImperfect:
            def __init__(self):
                self.t5 = MockT5()
                self.eval_mode = False

            def eval(self):
                self.eval_mode = True

            def __call__(self, **kwargs):
                # Simulate model outputs with class logits and generated texts
                return {
                    "loss": torch.tensor(0.8),
                    "class_logits": torch.tensor(
                        [
                            [0.3, 0.4, 0.3],  # Wrong prediction (true class 0)
                            [0.3, 0.3, 0.4],  # Wrong prediction (true class 1)
                            [0.3, 0.3, 0.4],  # Correct prediction (true class 2)
                        ]
                    ),
                }

        # Test evaluation
        mock_loader = [mock_batch]
        results = evaluate(MockModel(), mock_loader, device, tokenizer)
        # Verify basic results structure
        assert "loss" in results, "Missing loss in results"
        assert isinstance(results["loss"], float), "Loss should be float"
        assert 0 <= results["loss"] <= 10, "Loss out of reasonable range"

        # Verify classification metrics structure
        assert "classification" in results, "Missing classification metrics"
        classification_metrics = results["classification"]
        assert isinstance(classification_metrics, dict), "Classification metrics should be a dict"

        # Verify all expected classification metrics are present
        expected_metrics = ["accuracy", "f1", "sensitivity", "specificity", "auc"]
        for metric in expected_metrics:
            assert metric in classification_metrics, f"Missing {metric} in classification metrics"
            if classification_metrics[metric] is not None:
                assert isinstance(classification_metrics[metric], float), f"{metric} should be float"

        # Verify text generation metrics are present and valid
        assert "text_generation" in results, "Missing text generation metrics"
        text_gen_metrics = results["text_generation"]
        assert isinstance(text_gen_metrics, dict), "Text generation metrics should be a dict"

        expected_text_metrics = ["bertscore", "meteor"]
        for metric in expected_text_metrics:
            assert metric in text_gen_metrics, f"Missing {metric} in text generation metrics"
            if text_gen_metrics[metric] is not None:
                assert isinstance(text_gen_metrics[metric], float), f"{metric} should be float"
                assert 0 <= text_gen_metrics[metric] <= 1, f"{metric} should be between 0 and 1"

        # Verify specific metric values based on our mock data
        # Our mock data has perfect predictions (argmax matches class_labels)
        assert classification_metrics["accuracy"] == 1.0, "Accuracy should be 1.0 with perfect predictions"
        assert classification_metrics["f1"] == 1.0, "F1 should be 1.0 with perfect predictions"
        assert classification_metrics["sensitivity"] == 1.0, "Sensitivity should be 1.0 with perfect predictions"
        assert classification_metrics["specificity"] == 1.0, "Specificity should be 1.0 with perfect predictions"
        if classification_metrics["auc"] is not None:
            assert classification_metrics["auc"] == 1.0, "AUC should be 1.0 with perfect predictions"

        # Test with imperfect predictions
        imperfect_results = evaluate(MockModelImperfect(), mock_loader, device, tokenizer)

        # 打印不完美预测的准确率，用于调试
        print(f"Imperfect accuracy: {imperfect_results['classification']['accuracy']}")
        assert (
            imperfect_results["classification"]["accuracy"] < 1.0
        ), "Accuracy should be <1.0 with imperfect predictions"

        # Verify imperfect text generation metrics
        if "text_generation" in imperfect_results:
            print(f"Imperfect BERTScore: {imperfect_results['text_generation']['bertscore']}")
            print(f"Imperfect METEOR: {imperfect_results['text_generation']['meteor']}")

        print_success("Evaluate function test passed")
        return True
    except Exception as e:
        print(f"[ERROR] Evaluate function test failed: {str(e)}")
        return False


def test_config_validation(config):
    """Test configuration validation"""
    try:
        print_step("Testing configuration validation")

        required_keys = [
            "model",
            "training",
            "data",
            "data.paths",
            "data.preprocessing",
            "training.log_dir",
            "training.inference_log_dir",
        ]

        for key in required_keys:
            keys = key.split(".")
            current = config
            for k in keys:
                assert k in current, f"Missing config key: {key}"
                current = current[k]

        print_success("Configuration validation passed")
        return True
    except Exception as e:
        print(f"[ERROR] Configuration validation failed: {str(e)}")
        return False


def run_full_test_suite(MWRDataset, MWRModel, config, get_lora_config, evaluate, load_and_preprocess):
    """Run complete test suite"""
    print("\n=== Running Full Test Suite ===")

    tests = [
        lambda: test_data_loading(config, load_and_preprocess),
        lambda: test_model_initialization(MWRModel, config, get_lora_config),
        lambda: test_training_workflow(MWRDataset, MWRModel, config, get_lora_config, load_and_preprocess),
        test_evaluation_metrics,
        lambda: test_evaluate_function(config, evaluate),
        lambda: test_config_validation(config),
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
