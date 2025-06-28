# Standard library imports
from pathlib import Path
import pandas as pd

data_path = Path(__file__).parent.parent / "data" / "data_th_scale.csv"
df = pd.read_csv(data_path)

# Analyze temperature values
temp_values = []
for col in df.columns:
    if ("int" in col or "sk" in col) and ("L" in col or "R" in col):
        temp_values.extend(df[col].dropna().values)

print(f"Max temperature: {max(temp_values) if temp_values else 0:.1f}°C")
print(f"Min temperature: {min(temp_values) if temp_values else 0:.1f}°C")
