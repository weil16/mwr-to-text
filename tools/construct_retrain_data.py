import pandas as pd
from pathlib import Path

# Add project root to path and get absolute paths
project_root = Path(__file__).parent.parent


def construct_texts_from_csv(input_csv, output_file):
    """Process CSV file and construct new text sequences"""
    df = pd.read_csv(input_csv)
    texts = []

    for _, row in df.iterrows():
        # Generate individual temperature entries
        for i in range(10):
            for t_type in ["int", "sk"]:
                type_full = "internal" if t_type == "int" else "skin"

                # Left breast
                left_col = f"L{i} {t_type}"
                if left_col in row:
                    temp = row[left_col]
                    text = f"The {type_full} temperature {i} of the left breast is [TEMP]{temp:.1f}°C[/TEMP]."
                    texts.append(text)

                # Right breast
                right_col = f"R{i} {t_type}"
                if right_col in row:
                    temp = row[right_col]
                    text = f"The {type_full} temperature {i} of the right breast is [TEMP]{temp:.1f}°C[/TEMP]."
                    texts.append(text)

    # Save to output file
    with open(output_file, "w") as f:
        for text in texts:
            f.write(text + "\n")


if __name__ == "__main__":
    # Hardcoded file paths with project root
    input_csv = project_root / "data/processed_data_th_scale.csv"  # Absolute input path
    output_file = project_root / "tokenizer_retrain_data.txt"  # Absolute output path

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    construct_texts_from_csv(str(input_csv), str(output_file))
    print(f"Processed data from {input_csv} and saved results to {output_file}")
