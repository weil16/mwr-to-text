"""Hyperparameter optimization using Optuna for MWR model"""
# Standard library imports
import datetime
import sys
from pathlib import Path
import argparse
import shutil

# Third-party imports
import optuna
import torch
import yaml

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from src.train import train
from src.model import MWRModel, get_lora_config
from transformers import T5Tokenizer

def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    print(f"\n=== Starting Trial {trial.number} ===")
    
    # Load base config
    config_path = Path(__file__).parent.parent/ "config" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print("Loaded base configuration")
    
    # Define search space
    config["training"]["learning_rate"] = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    config["training"]["batch_size"] = trial.suggest_categorical("batch_size", [8, 16, 32])
    
    # LoRA parameters
    config["model"]["lora"]["rank"] = trial.suggest_int("lora_rank", 4, 16)
    config["model"]["lora"]["alpha"] = trial.suggest_int("lora_alpha", 16, 64)
    config["model"]["lora"]["dropout"] = trial.suggest_float("lora_dropout", 0.0, 0.5)
    
    # Loss weights
    config["model"]["loss_weights"]["generation"] = trial.suggest_float("gen_weight", 0.1, 0.9)
    config["model"]["loss_weights"]["classification"] = 1.0 - config["model"]["loss_weights"]["generation"]
    
    print(f"Trial parameters:")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  LoRA rank: {config['model']['lora']['rank']}")
    print(f"  LoRA alpha: {config['model']['lora']['alpha']}")
    print(f"  LoRA dropout: {config['model']['lora']['dropout']}")
    print(f"  Generation weight: {config['model']['loss_weights']['generation']}")
    
    # Setup directories - only logs and params
    optuna_dir = Path("optuna")
    trial_dir = optuna_dir / "trials" / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trial parameters
    params_path = trial_dir / "params.yaml"
    with open(params_path, "w") as f:
        yaml.dump(trial.params, f)
    
    config["training"]["log_dir"] = str(trial_dir)
    print(f"Created trial directory at: {trial_dir}")
    print(f"Saved trial parameters to: {params_path}")
    
    # Mock args
    class Args:
        evaluate_test = False
        resume_from_checkpoint = None
    
    # Run training and get best validation loss
    print("\nStarting training...")
    best_val_loss = train(Args(), config)
    print(f"Training completed with best validation loss: {best_val_loss:.4f}")
    
    print("Trial completed - logs and params saved")
    print(f"=== Completed Trial {trial.number} ===\n")
    return best_val_loss

def generate_visualizations(study):
    """Generate and save visualization plots for the study"""
    # Create visualizations directory
    vis_dir = Path(__file__).parent.parent/ "optuna" / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_slice
        )
        
        # Optimization history
        fig = plot_optimization_history(study)
        fig.write_image(str(vis_dir / "optimization_history.png"))
        
        # Parameter importance
        fig = plot_param_importances(study)
        fig.write_image(str(vis_dir / "param_importances.png"))
        
        # Parallel coordinate plot
        fig = plot_parallel_coordinate(study)
        fig.write_image(str(vis_dir / "parallel_coordinate.png"))
        
        # Slice plot
        fig = plot_slice(study)
        fig.write_image(str(vis_dir / "slice_plot.png"))
        
        # Show plots if in notebook
        try:
            from IPython.display import display
            display(fig)
        except ImportError:
            pass
            
    except ImportError as e:
        print(f"Visualization dependencies not available: {e}")

def optimize_hyperparameters(n_trials=50, visualize=True, study_name=None):
    print("\n=== Starting Hyperparameter Optimization ===")
    print(f"Number of trials: {n_trials}")
    """Run hyperparameter optimization
    
    Args:
        n_trials (int): Number of optimization trials
        visualize (bool): Whether to generate visualization plots
        study_name (str): Name for the study. If None, uses default name.
    """
    if study_name is None:
        try:
            study_name = f"MWR_optimization_{datetime.datetime.now().strftime('%Y%m%d')}"
        except AttributeError:
            # Fallback in case datetime module is not properly imported
            study_name = f"MWR_optimization"
    
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(),
        study_name=study_name
    )
    
    study.optimize(objective, n_trials=n_trials)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save only the single best configuration
    best_config_path = Path("optuna") / "best_config.yaml"
    with open(best_config_path, "w") as f:
        yaml.dump(study.best_trial.params, f)
    print(f"\nSaved best configuration to: {best_config_path}")
    
    if visualize:
        generate_visualizations(study)
    
    return study

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=20, help="Number of optimization trials")
    parser.add_argument("--no-visualize", action="store_false", dest="visualize", 
                       help="Disable visualization generation")
    parser.add_argument("--study-name", type=str, default=None,
                       help="Custom name for the study")
    args = parser.parse_args()
    
    optimize_hyperparameters(args.n_trials, args.visualize, args.study_name)

