# Standard library imports
import warnings

# Third-party imports
import pandas as pd
import torch
from torch.utils.data import Dataset


class MWRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length=512, truncation_strategy="right"):
        """
        Custom dataset for MWR classification and text generation

        Args:
            df: Preprocessed DataFrame containing temperature readings and labels
            tokenizer: T5 tokenizer (customized)
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
        self.class_labels = torch.tensor(df["Cancer_risk"].values, dtype=torch.long)

        # Text generation targets
        self.target_texts = df["Conclusion (Tr)"].fillna("").astype(str).tolist()

    def _prepare_input_texts(self):
        """Convert numerical data to text sequence for T5 input"""
        texts = []
        for _, row in self.df.iterrows():
            left_temps = []
            right_temps = []

            for i in range(10):
                for t_type in ["int", "sk"]:
                    type_full = "internal" if t_type == "int" else "skin"

                    # Left breast
                    left_col = f"L{i} {t_type}"
                    if left_col in row:
                        left_temps.append(f"{type_full} temperature {i}: [TEMP]{row[left_col]:.1f}°C[/TEMP]")

                    # Right breast
                    right_col = f"R{i} {t_type}"
                    if right_col in row:
                        right_temps.append(f"{type_full} temperature {i}: [TEMP]{row[right_col]:.1f}°C[/TEMP]")

            # Combine temperature texts
            left_text = f"Left breast temps: {'; '.join(left_temps)}" if left_temps else ""
            right_text = f"Right breast temps: {'; '.join(right_temps)}" if right_temps else ""
            temp_text = [left_text, right_text] if left_text and right_text else [left_text or right_text]

            # Medical history features (medium priority)
            medical_features = [
                f"Cancer history: {row['Cancer family history']}",
                f"Breast ops: {row['Breast operations']}",
            ]

            # Demographic features (lowest priority)
            demographic_features = [
                f"Age: [AGE]{row['r:AgeInYears']}[/AGE]",
                f"Cycle: {row['Cycle_explain']}",
                f"Day: {row['Day_explain']}",
                f"Pregnancies: [COUNT]{row['Num of pregnancies']}[/COUNT]",
                f"Hormonal meds: {row['Hormonal medications']}",
            ]

            # Combine based on truncation strategy
            if self.truncation_strategy == "right":
                full_text = "MWR data: " + "; ".join(temp_text + medical_features + demographic_features)
            elif self.truncation_strategy == "left":
                full_text = "MWR data: " + "; ".join(demographic_features + medical_features + temp_text)
            else:  # middle
                full_text = "MWR data: " + "; ".join(medical_features + temp_text + demographic_features)
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
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze(),
            "class_labels": class_label,
        }
