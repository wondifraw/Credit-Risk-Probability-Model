"""
Data Preprocessing Module for Credit Risk Dataset
- Loading raw data
- Cleaning and normalization (including Amharic-specific if needed)
- Tokenization (if text fields exist)
- Saving processed data
"""
import os
import pandas as pd
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

def format_datetime_column(df, column, fmt=None, errors='coerce'):
    """
    Convert a column in the DataFrame to pandas datetime.
    Optionally specify a format string (fmt) and error handling.
    """
    df[column] = pd.to_datetime(df[column], format=fmt, errors=errors)
    return df

def preprocess_dataframe(df, text_columns=None, datetime_columns=None, datetime_format=None):
    """
    Apply cleaning and normalization to specified text columns.
    Optionally format specified datetime columns.
    """
    if text_columns is None:
        # Auto-detect object columns
        text_columns = df.select_dtypes(include='object').columns.tolist()
    for col in text_columns:
        df[col] = df[col].apply(clean_text)
    if datetime_columns is not None:
        for col in datetime_columns:
            df = format_datetime_column(df, col, fmt=datetime_format)
    return df

def tokenize_text(text):
    """
    Simple whitespace tokenization. Extend for Amharic or advanced tokenization if needed.
    """
    return text.split()

def add_tokenized_column(df, text_column, new_column='tokens'):
    """
    Add a column with tokenized text.
    """
    df[new_column] = df[text_column].apply(tokenize_text)
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
#     df = preprocess_dataframe(df)
#     # If you want to tokenize a specific column, e.g., 'message':
#     # df = add_tokenized_column(df, 'message')
#     save_processed_data(df)
