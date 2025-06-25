# Standard library imports
import os

# Third-party imports
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def add_cancer_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Add cancer risk column based on r:Th values"""
    df['cancer_risk'] = df['r:Th'].apply(lambda x: 0 if x == 0 else 1)
    return df

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize temperature data according to project specifications"""
    # Use new normalization method
    df = _normalize_new(df, 'R', 'T1 int')
    df = _normalize_new(df, 'L', 'T1 int')
    df = _normalize_new(df, 'R', 'T1 sk')
    df = _normalize_new(df, 'L', 'T1 sk')

    # Use old normalization method
    # df = _normalize(df, 'R', 'T1 int')
    # df = _normalize(df, 'L', 'T1 int')
    # df = _normalize(df, 'R', 'T1 sk')
    # df = _normalize(df, 'L', 'T1 sk')
    
    return df

def _normalize(df: pd.DataFrame, label_tag: str, ref_label: str) -> pd.DataFrame:
    """Helper function for temperature normalization"""
    def line_function(x, A, B):
        return A * x + B

    def transform(temperature, A, refAvg, ref):
        return round(temperature + A * (refAvg - ref), 1)
    
    ref_mean = df[ref_label].mean(axis=0)
    
    for i in range(10):
        label = f"{label_tag}{i} int"
        if label in df.columns:
            A, B = curve_fit(line_function, df[ref_label].values, df[label].values)[0]
            df[label] = np.vectorize(transform)(df[label], A, ref_mean, df[ref_label])
    return df

def _normalize_new(df: pd.DataFrame, label_tag: str, ref_label: str) -> pd.DataFrame:
    """New normalization method using control point linear regression"""
    def fit_and_transform(temp_col: str, control_col: str) -> pd.Series:
        # Fit linear model: temp = k * control + b
        k, b = curve_fit(lambda x, k, b: k * x + b,
                        df[control_col], df[temp_col])[0]
        # Apply normalization: temp' = k * control + b
        return df[temp_col] - (k * df[control_col] + b)
    
    for i in range(10):
        label = f"{label_tag}{i} int"
        if label in df.columns:
            df[label] = fit_and_transform(label, ref_label).round(1)
    return df

def load_and_preprocess(data_path: str = 'data/data_th_scale.csv') -> pd.DataFrame:
    """Load and preprocess the raw data
    
    Args:
        data_path: Path to the raw data file. Defaults to 'data/data_th_scale.csv'
    
    Returns:
        Processed DataFrame
    
    Raises:
        FileNotFoundError: If input file does not exist
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input file not found at: {data_path}")

    # Check if processed file already exists
    dir_path = os.path.dirname(data_path)
    file_name = os.path.basename(data_path)
    output_path = os.path.join(dir_path, f"processed_{file_name}")

    if os.path.exists(output_path):
        print(f"🎯 检测到已处理的{file_name}缓存文件，直接使用缓存数据~")
        return pd.read_csv(output_path)

    df = pd.read_csv(data_path)
    df = add_cancer_risk(df)
    processed_df = normalize_data(df)
    processed_df.to_csv(output_path, index=False)
    
    return processed_df
