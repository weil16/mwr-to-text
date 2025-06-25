# Standard library imports
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Third-party imports
import numpy as np
import torch
import yaml
import nltk
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from bert_score import score as bert_score
from tqdm import tqdm
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train(args, config, MWRDataset, MWRModel, get_lora_config, load_and_preprocess, split_data):
    """Main training function"""
    # Setup directories
    os.makedirs(config["training"]["log_dir"], exist_ok=True)
    log_file = os.path.join(
        config["training"]["log_dir"],
        f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(config["model"]["name"], legacy=True)
    lora_config = get_lora_config(config)
    model = MWRModel(config=config, lora_config=lora_config).to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config.get("training", {}).get("lr_factor", 0.5),
        patience=config.get("training", {}).get("lr_patience", 3),
    )
    
    current_epoch = 0

    # Load from checkpoint if specified
    if args.resume_from_checkpoint:
        checkpoint = torch.load(args.resume_from_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        current_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        print(f"   已加载模型状态、优化器状态和学习率调度器状态")
    
    # Data preparation
    if not all(os.path.exists(p) for p in [
        config["data"]["paths"]["train"],
        config["data"]["paths"]["val"],
        config["data"]["paths"]["test"]
    ]):
        split_data()
    
    # Load and preprocess data
    train_df = load_and_preprocess(config["data"]["paths"]["train"])
    val_df = load_and_preprocess(config["data"]["paths"]["val"])
    
    # Create datasets
    dataset_config = {
        "max_length": config["data"]["preprocessing"]["max_seq_length"],
        "truncation_strategy": config["data"]["preprocessing"]["truncation_strategy"]
    }
    train_dataset = MWRDataset(train_df, tokenizer, **dataset_config)
    val_dataset = MWRDataset(val_df, tokenizer, **dataset_config)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"]
    )
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    total_epochs = config["training"]["epochs"]
    early_stopping_patience = config["training"]["early_stopping"]["patience"]
    patience_counter = 0
    
    print(f"\n开始训练 - 总共 {total_epochs} 个epoch")
    print(f"训练集大小: {len(train_dataset)} 样本")
    print(f"验证集大小: {len(val_dataset)} 样本")
    print(f"批次大小: {config['training']['batch_size']}")
    print(f"每个epoch的批次数: {len(train_loader)}")
    print("-" * 60)
    
    training_start_time = time.time()
    
    for epoch in range(current_epoch, total_epochs):
        model.train()
        total_loss = 0
        epoch_start_time = time.time()
        
        # 创建进度条
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {current_epoch + epoch + 1}/{total_epochs - current_epoch}",
            ncols=100,
            leave=True
        )
        
        for batch_idx, batch in pbar:
            try:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    class_labels=batch["class_labels"]
                )
                
                # Backward pass
                loss = outputs["loss"]
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % config["training"]["gradient_accumulation_steps"] == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item()
                
                # 更新进度条
                current_avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'Loss': f'{current_avg_loss:.4f}',
                    'Batch': f'{batch_idx + 1}/{len(train_loader)}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
                
            except Exception as e:
                print(f"\n训练批次 {batch_idx + 1} 发生错误: {e}")
                continue
        
        # 处理最后不完整的梯度累积批次
        if len(train_loader) % config["training"]["gradient_accumulation_steps"] != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # 计算epoch统计信息
        avg_train_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        total_elapsed = time.time() - training_start_time
        
        # 估算剩余时间
        avg_epoch_time = total_elapsed / (epoch + 1)
        remaining_epochs = total_epochs - (current_epoch + epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs
        
        # 格式化时间
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            if hours > 0:
                return f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{seconds}s"
        
        # 打印详细的epoch总结
        print(f"\n Epoch {current_epoch + epoch + 1}/{total_epochs} 完成!")
        print(f"   训练损失: {avg_train_loss:.4f}")
        print(f"   本epoch用时: {format_time(epoch_time)}")
        print(f"   总用时: {format_time(total_elapsed)}")
        if remaining_epochs > 0:
            print(f"   预计剩余时间: {format_time(estimated_remaining)}")
        
        # 记录日志
        log_message = f"Epoch {current_epoch + epoch + 1}/{total_epochs}, Train Loss: {avg_train_loss:.4f}, Time: {format_time(epoch_time)}, Total Time: {format_time(total_elapsed)}"
        with open(log_file, "a") as f:
            f.write(log_message + "\n")
        
        # Validation
        if (epoch + 1) % config["training"]["eval_interval"] == 0:
            print("\n🔍 开始验证...")
            eval_start_time = time.time()
            eval_results = evaluate(model, val_loader, device, tokenizer)
            eval_time = time.time() - eval_start_time
            
            print(f"   验证损失: {eval_results['loss']:.4f}")
            if eval_results['bertscore']:
                print(f"   BERTScore - P: {eval_results['bertscore']['precision']:.4f}, R: {eval_results['bertscore']['recall']:.4f}, F1: {eval_results['bertscore']['f1']:.4f}")
            if eval_results['meteor']:
                print(f"   METEOR Score: {eval_results['meteor']:.4f}")
            print(f"   验证用时: {format_time(eval_time)}")
            
            # 记录验证日志
            log_message = f"Epoch {epoch + 1}, Val Loss: {eval_results['loss']:.4f}, Val Time: {format_time(eval_time)}"
            if eval_results['bertscore']:
                log_message += f", BERTScore F1: {eval_results['bertscore']['f1']:.4f}"
            if eval_results['meteor']:
                log_message += f", METEOR Score: {eval_results['meteor']:.4f}"
            with open(log_file, "a") as f:
                f.write(log_message + "\n")
            
            # Learning rate scheduling
            scheduler.step(eval_results['loss'])
            
            # Save best model and early stopping
            if eval_results['loss'] < best_val_loss:
                best_val_loss = eval_results['loss']
                patience_counter = 0
                print(f"   🎉 新的最佳模型! (验证损失: {best_val_loss:.4f})")
                save_model(model, config, epoch, optimizer, scheduler, best_val_loss, "best_model")
            else:
                patience_counter += 1
                print(f"   当前最佳验证损失: {best_val_loss:.4f}")
                print(f"   早停计数器: {patience_counter}/{early_stopping_patience}")
                
                # Early stopping check
                if patience_counter >= early_stopping_patience:
                    print(f"\n⏹️ 早停触发! 验证损失在 {early_stopping_patience} 个epoch内没有改善")
                    print(f"最佳验证损失: {best_val_loss:.4f}")
                    break
        
        # 定期保存检查点（在验证之后）
        if (epoch + 1) % config["training"]["save_interval"] == 0:
            print(f"\n💾 保存检查点到 {config['training']['checkpoint_dir']}")
            save_model(
                model, 
                config, 
                epoch, 
                optimizer, 
                scheduler, 
                best_val_loss, 
                "checkpoint"
            )
        
        print("-" * 60)
    
    # Return best validation loss for hyperparameter optimization
    return best_val_loss

def evaluate(model, data_loader, device, tokenizer):
    """Evaluate model on validation/test set with generation quality metrics"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_refs = []
    batch_count = 0
    
    # 创建验证进度条
    eval_pbar = tqdm(
        data_loader,
        desc="验证中",
        ncols=80,
        leave=False
    )
    
    with torch.no_grad():
        for batch in eval_pbar:
            try:
                batch = {k: v.to(device) for k, v in batch.items()}
                batch_count += 1
                
                # Forward pass
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    class_labels=batch["class_labels"]
                )
                total_loss += outputs["loss"].item()
                
                # Generate predictions
                generator = model.t5
                    
                generated_ids = generator.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=128,
                    num_beams=4,
                    early_stopping=True
                )
                preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                refs = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
                
                all_preds.extend(preds)
                all_refs.extend(refs)
                
                # 更新验证进度条 - 修正计算方式
                current_avg_loss = total_loss / batch_count
                eval_pbar.set_postfix({'Val Loss': f'{current_avg_loss:.4f}'})
                
            except Exception as e:
                print(f"\n验证批次发生错误: {e}")
                continue
    
    # Calculate metrics with strict filtering and detailed logging
    filtered_preds = []
    filtered_refs = []
    empty_samples = []
    
    for i, (pred, ref) in enumerate(zip(all_preds, all_refs)):
        # Strict filtering: non-empty strings with length > 0 after stripping
        if isinstance(pred, str) and isinstance(ref, str) and len(pred.strip()) > 0 and len(ref.strip()) > 0:
            filtered_preds.append(pred.strip())
            filtered_refs.append(ref.strip())
        else:
            empty_samples.append({
                'index': i,
                'prediction': pred,
                'reference': ref
            })
    
    # Log detailed filtering results
    if empty_samples:
        print(f"\n   ⚠️ Filtered {len(empty_samples)} empty/invalid samples:")
        for sample in empty_samples[:5]:  # 最多打印5个样本避免日志过长
            print(f"      Sample {sample['index']}:")
            print(f"        Input: {data_loader.dataset.input_texts[sample['index']]}")
            print(f"        Prediction: {repr(sample['prediction'])}")
            print(f"        Reference: {repr(sample['reference'])}")
        if len(empty_samples) > 5:
            print(f"      ... and {len(empty_samples)-5} more empty samples")
    
    metrics = {
        "loss": total_loss / len(data_loader),
        "bertscore": None,
        "meteor": None
    }
    
    if len(filtered_preds) > 0 and len(filtered_refs) > 0:
        # Calculate BERTScore
        try:
            P, R, F1 = bert_score(
                filtered_preds, 
                filtered_refs, 
                lang="en",
                model_type="bert-base-uncased",
                device=device,
                verbose=False  # Disable BERTScore internal warnings
            )
            metrics["bertscore"] = {
                "precision": P.mean().item(),
                "recall": R.mean().item(),
                "f1": F1.mean().item()
            }
        except Exception as e:
            print(f"   ⚠️ BERTScore calculation error: {str(e)}")
        
        # Calculate METEOR score
        try:
            meteor_scores = [
                nltk.translate.meteor_score.meteor_score(
                    [ref.split()], 
                    pred.split()
                )
                for pred, ref in zip(filtered_preds, filtered_refs)
            ]
            metrics["meteor"] = sum(meteor_scores) / len(meteor_scores)
        except Exception as e:
            print(f"   ⚠️ METEOR score calculation error: {str(e)}")
    else:
        print("   ⚠️ No valid samples for metric calculations")
    
    return metrics

def save_model(model, config, epoch, optimizer, scheduler, best_val_loss, model_type="checkpoint"):
    """Save model checkpoint"""
    save_dir = config["training"]["best_model_dir"] if model_type == "best_model" \
               else config["training"]["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(
        save_dir,
        "best_model.pt" if model_type == "best_model" else f"{model_type}_epoch_{epoch + 1}.pt"
    )
    
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "config": config
    }, model_path)