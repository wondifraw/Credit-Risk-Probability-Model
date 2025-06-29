import os
import pandas as pd
import pytest
import tempfile
import shutil
import sys
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import data_processing
import numpy as np

def test_clean_text():
    assert data_processing.clean_text('Hello, World!') == 'hello world'
    assert data_processing.clean_text(None) == ''
    assert data_processing.clean_text('  ሰላም።  ') == 'ሰላም'

def test_tokenize_text():
    assert data_processing.tokenize_text('hello world') == ['hello', 'world']
    assert data_processing.tokenize_text('') == []
    print("test_tokenize_text passed.")

def test_preprocess_dataframe():
    df = pd.DataFrame({'text': ['Hello, World!', '  ሰላም።  ']})
    df_clean = data_processing.preprocess_dataframe(df, text_columns=['text'])
    assert df_clean['text'][0] == 'hello world'
    assert df_clean['text'][1] == 'ሰላም'

def test_add_tokenized_column():
    df = pd.DataFrame({'text': ['hello world']})
    df = data_processing.add_tokenized_column(df, 'text', new_column='tokens')
    assert df['tokens'][0] == ['hello', 'world']

def test_save_and_load_processed_data():
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, 'test.csv')
        data_processing.save_processed_data(df, filename='test.csv', processed_dir=tmpdir)
        loaded = pd.read_csv(file_path)
        assert loaded.equals(df)

def test_load_raw_data(tmp_path):
    # Create a temporary CSV file
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    file_path = raw_dir / "test.csv"
    df.to_csv(file_path, index=False)
    # Test loading
    loaded = data_processing.load_raw_data(filename="test.csv", raw_data_dir=str(raw_dir))
    assert loaded.equals(df)

def test_handle_missing_values():
    df = pd.DataFrame({'a': [1, np.nan, 3], 'b': ['x', None, 'z']})
    df_filled = data_processing.handle_missing_values(df.copy(), strategy='mean')
    assert df_filled['a'].isnull().sum() == 0
    df_filled = data_processing.handle_missing_values(df.copy(), strategy='mode')
    assert df_filled['b'].isnull().sum() == 0
    df_filled = data_processing.handle_missing_values(df.copy(), strategy='constant', fill_value=0)
    assert (df_filled['a'] == 0).sum() == 1
    print('test_handle_missing_values passed.')

def test_detect_and_remove_outliers():
    df = pd.DataFrame({'x': [1, 2, 3, 100], 'y': [10, 20, 30, -999]})
    mask = data_processing.detect_outliers_iqr(df, ['x', 'y'])
    assert mask.sum() < len(df)  # At least one outlier detected
    df_no_outliers = data_processing.remove_outliers(df, ['x', 'y'])
    assert df_no_outliers.shape[0] < df.shape[0]
    print('test_detect_and_remove_outliers passed.')

def test_add_transaction_size_bins():
    df = pd.DataFrame({'Amount': [50, 500, 5000, 50000]})
    df = data_processing.add_transaction_size_bins(df)
    assert set(df['TransactionSizeCategory'].unique()) <= {'small', 'medium', 'large', 'very_large'}
    print('test_add_transaction_size_bins passed.')

test_handle_missing_values()
test_detect_and_remove_outliers()
test_add_transaction_size_bins()
