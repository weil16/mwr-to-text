# Standard library imports
import warnings

# Third-party imports
import pandas as pd
import torch
from torch.utils.data import Dataset

class MWRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length=512, truncation_strategy='right'):
        """
        Custom dataset for MWR classification and text generation
        
        Args:
            df: Preprocessed DataFrame containing temperature readings and labels
            tokenizer: T5 tokenizer
            max_length: Maximum sequence length
            truncation_strategy: How to handle long sequences ('right', 'left', 'middle')
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy

        
        # Prepare input text by converting numerical data to text sequence
        self.input_texts = self._prepare_input_texts()
        
        # Classification labels
        self.class_labels = torch.tensor(df['cancer_risk'].values, dtype=torch.long)
        
        # Text generation targets
        self.target_texts = df['Conclusion (Tr)'].fillna('').astype(str).tolist()

    def _prepare_input_texts(self):
        """Convert numerical data to text sequence for T5 input"""
        texts = []
        for _, row in self.df.iterrows():
            # Temperature readings (highest priority)
            temp_text = []
            for side in ['R', 'L']:
                for i in range(10):
                    for t_type in ['int', 'sk']:
                        col = f"{side}{i} {t_type}"
                        if col in row:
                            temp_text.append(f"{col}: {row[col]:.2f}")
            
            # Medical history features (medium priority)
            medical_features = [
                f"Cancer history: {row['Cancer family history']}",
                f"Breast ops: {row['Breast operations']}"
            ]
            
            # Demographic features (lowest priority)
            demographic_features = [
                f"Age: {row['r:AgeInYears']}",
                f"Cycle: {row['Cycle']}",
                f"Day: {row['Day from the first day']}",
                f"Pregnancies: {row['Num of pregnancies']}",
                f"Hormonal meds: {row['Hormonal medications']}"
            ]
            
            # Combine based on truncation strategy
            if self.truncation_strategy == 'right':
                full_text = "MWR data: " + "; ".join(temp_text + medical_features + demographic_features)
            elif self.truncation_strategy == 'left':
                full_text = "MWR data: " + "; ".join(demographic_features + medical_features + temp_text)
            else:  # middle
                full_text = "MWR data: " + "; ".join(
                    medical_features + 
                    temp_text + 
                    demographic_features
                )
            texts.append(full_text)
        
        return texts

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        class_label = self.class_labels[idx]
        target_text = self.target_texts[idx]
        
        # Tokenize input and target with new behavior
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
            
        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze(),
            'class_labels': class_label
        }
