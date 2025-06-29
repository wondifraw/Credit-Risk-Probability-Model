"""
Data Preprocessing Module for Credit Risk Dataset
- Loading raw data
- Cleaning and normalization (including Amharic-specific if needed)
- Handling missing values
- Outlier detection and removal
- Feature engineering (e.g., transaction size bins)
- Tokenization (if text fields exist)
- Saving processed data
"""
import os
import pandas as pd
import numpy as np
import re

# If Amharic-specific normalization is needed, import or define here
# Example: from amharic_nlp import normalize_amharic_text

def load_raw_data(filename=None, raw_data_dir='../data/raw/'):
    """
    Load raw CSV data from the specified directory.
    If filename is None, loads the first CSV file found.
    """
    files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError('No CSV files found in data/raw/')
    if filename is None:
        filename = files[0]
    df = pd.read_csv(os.path.join(raw_data_dir, filename))
    return df

def clean_text(text):
    """
    Basic text cleaning: remove special characters, extra spaces, and lowercasing.
    Extend this function for Amharic-specific normalization if needed.
    """
    if pd.isnull(text):
        return ''
    # Remove Amharic punctuation (።፣፤፥፦፧፨)
    text = re.sub(r'[።፣፤፥፦፧፨]', '', str(text))
    # Remove non-letter characters (keep Amharic Unicode range if needed)
    text = re.sub(r'[^\u1200-\u137F\w\s]', '', text)  # Amharic + basic Latin
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    # Optionally: text = normalize_amharic_text(text)
    return text

def handle_missing_values(df, strategy='mean', fill_value=None):
    """
    Handle missing values in the DataFrame.
    strategy: 'mean', 'median', 'mode', or 'constant'.
    fill_value: used if strategy is 'constant'.
    """
    for col in df.columns:
        if df[col].isnull().any():
            if strategy == 'mean' and df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median' and df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif strategy == 'constant':
                df[col].fillna(fill_value, inplace=True)
    return df

def detect_outliers_iqr(df, columns, factor=1.5):
    """
    Detect outliers in specified columns using the IQR method.
    Returns a boolean mask where True indicates a non-outlier row.
    """
    mask = pd.Series([True] * len(df))
    for col in columns:
        if df[col].dtype in [np.float64, np.int64]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            mask &= df[col].between(lower, upper)
    return mask

def remove_outliers(df, columns, factor=1.5):
    """
    Remove outliers from specified columns using the IQR method.
    """
    mask = detect_outliers_iqr(df, columns, factor)
    return df[mask].reset_index(drop=True)

def add_transaction_size_bins(df, amount_column='Amount', bins=[-np.inf, 100, 1000, 10000, np.inf], labels=['small', 'medium', 'large', 'very_large']):
    """
    Add a categorical column for transaction size based on amount bins.
    """
    df['TransactionSizeCategory'] = pd.cut(df[amount_column], bins=bins, labels=labels)
    return df

def format_datetime_column(df, column, fmt=None, errors='coerce'):
    """
    Convert a column in the DataFrame to pandas datetime.
    Optionally specify a format string (fmt) and error handling.
    """
    df[column] = pd.to_datetime(df[column], format=fmt, errors=errors)
    return df

def preprocess_dataframe(df, text_columns=None, datetime_columns=None, datetime_format=None, missing_strategy='mean', outlier_columns=None, outlier_factor=1.5, add_bins=False, amount_column='Amount'):
    """
    Comprehensive preprocessing: clean text, handle missing values, format datetimes, remove outliers, and add feature engineering.
    """
    # Handle missing values
    df = handle_missing_values(df, strategy=missing_strategy)
    # Clean text columns
    if text_columns is None:
        text_columns = df.select_dtypes(include='object').columns.tolist()
    for col in text_columns:
        df[col] = df[col].apply(clean_text)
    # Format datetime columns
    if datetime_columns is not None:
        for col in datetime_columns:
            df = format_datetime_column(df, col, fmt=datetime_format)
    # Remove outliers
    if outlier_columns is not None:
        df = remove_outliers(df, outlier_columns, factor=outlier_factor)
    # Feature engineering: add transaction size bins
    if add_bins:
        df = add_transaction_size_bins(df, amount_column=amount_column)
    return df

def add_tokenized_column(df, text_column, new_column='tokens'):
    """
    Add a column with tokenized text.
    """
    df[new_column] = df[text_column].apply(lambda x: x.split())
    return df

def save_processed_data(df, filename='processed.csv', processed_dir='../data/processed/'):
    """
    Save the processed DataFrame to the processed directory.
    """
    os.makedirs(processed_dir, exist_ok=True)
    df.to_csv(os.path.join(processed_dir, filename), index=False)

def date_formatter(date_input, fmt=None, errors='coerce'):
    """
    Format a date string or pandas Series to pandas datetime.
    Args:
        date_input: str, pandas Series, or array-like
        fmt: Optional format string for datetime parsing
        errors: How to handle errors ('raise', 'coerce', 'ignore')
    Returns:
        pandas.Timestamp, pandas.Series, or pandas.DatetimeIndex
    """
    return pd.to_datetime(date_input, format=fmt, errors=errors)

# Example usage (uncomment to run as script):
# if __name__ == '__main__':
#     df = load_raw_data()
#     df = preprocess_dataframe(df, outlier_columns=['Amount', 'Value'], add_bins=True)
#     # If you want to tokenize a specific column, e.g., 'message':
#     # df = add_tokenized_column(df, 'message')
#     save_processed_data(df)
