# Standard library imports
from pathlib import Path

# Third-party imports
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import T5Config, T5ForConditionalGeneration

class MWRModel(nn.Module):
    def __init__(self, model_name=None, lora_config=None, config=None):
        if config is None:
            raise ValueError("Config must be provided")
        # Model name must be specified in config
        model_name = model_name or config['model']['name']
        """
        Multi-task T5 model for classification and text generation
        
        Args:
            model_name: Pretrained T5 model name
            lora_config: Configuration for LoRA adaptation
            config: Model configuration containing loss weights
        """
        super().__init__()
        
        # Store custom config
        self.config = config
        # Try to load from local path first
        local_path = self.config["model"]["local_path"]
        if local_path and Path(local_path).exists():
            print(f"Loading model from local path: {local_path}")
            t5_config = T5Config.from_pretrained(local_path)
            t5_config.use_cache = False
            self.t5 = T5ForConditionalGeneration.from_pretrained(local_path, config=t5_config)
        else:
            print(f"Loading model from Hugging Face: {model_name}")
            t5_config = T5Config.from_pretrained(model_name)
            t5_config.use_cache = False
            self.t5 = T5ForConditionalGeneration.from_pretrained(model_name, config=t5_config)
        
        # Add classification head (get d_model from T5 config)
        self.classifier = nn.Linear(self.t5.config.d_model, 2)
        
        # Apply LoRA if configured
        if lora_config:
            self.t5 = get_peft_model(self.t5, lora_config)
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None, class_labels=None, decoder_input_ids=None):
        """
        Forward pass with both classification and generation outputs
        
        Args:
            input_ids: Encoder input token IDs
            attention_mask: Encoder attention mask
            labels: Decoder labels (used when decoder_input_ids not provided)
            class_labels: Classification labels
            decoder_input_ids: Explicit decoder input token IDs
        """
        # Handle decoder inputs - prefer explicit decoder_input_ids
        decoder_kwargs = {}
        if decoder_input_ids is not None:
            decoder_kwargs['decoder_input_ids'] = decoder_input_ids
        elif labels is not None:
            decoder_kwargs['labels'] = labels
        
        # Base T5 outputs
        outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **decoder_kwargs
        )
        
        # Classification using encoder outputs
        encoder_hidden_states = outputs.encoder_last_hidden_state  # [batch, seq_len, hidden_size]
        
        # Pooling strategy
        if self.config["model"]["pooling_strategy"] == "cls":
            # Use [CLS] token (first token)
            pooled_output = encoder_hidden_states[:, 0, :]
        else:
            # Default to mean pooling
            pooled_output = torch.mean(encoder_hidden_states, dim=1)
        
        # Classification logits
        cls_logits = self.classifier(pooled_output)
        
        # Calculate losses
        gen_loss = outputs.loss if labels is not None else None
        cls_loss = self.ce_loss(cls_logits, class_labels) if class_labels is not None else None
        
        # Combine losses based on task type
        if labels is not None and class_labels is not None:
            loss_weights = self.config["model"]["loss_weights"]
            total_loss = (gen_loss * loss_weights["generation"] + 
                         cls_loss * loss_weights["classification"])
        elif labels is not None:
            total_loss = gen_loss
        elif class_labels is not None:
            total_loss = cls_loss
        else:
            total_loss = None
            
        return {
            'loss': total_loss,
            'logits': outputs.logits,          # Generation logits
            'cls_logits': cls_logits,          # Classification logits
            'cls_loss': cls_loss,
            'gen_loss': gen_loss,
            'decoder_hidden_states': outputs.decoder_hidden_states
        }

def get_lora_config(config):
    """Get LoRA configuration from provided config"""
    if config is None:
        raise ValueError("Config must be provided")
        
    lora_config = config["model"]['lora']
    return LoraConfig(
        r=lora_config['rank'],
        lora_alpha=lora_config['alpha'],
        target_modules=lora_config['target_modules'],
        lora_dropout=lora_config['dropout'],
        bias=lora_config['bias'],
        task_type="SEQ_2_SEQ_LM"
    )
