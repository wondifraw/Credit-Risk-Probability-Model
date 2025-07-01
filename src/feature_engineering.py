import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
# from xverse.transformer import WOETransformer  # Uncomment if xverse is installed

# Custom transformer for aggregate features per customer
class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, customer_id_col='CustomerId', amount_col='Amount'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Group by customer and calculate aggregates
        agg_df = X.groupby(self.customer_id_col)[self.amount_col].agg([
            ('total_transaction_amount', 'sum'),
            ('avg_transaction_amount', 'mean'),
            ('transaction_count', 'count'),
            ('std_transaction_amount', 'std')
        ]).reset_index()
        # Merge back to original data
        X = X.merge(agg_df, on=self.customer_id_col, how='left')
        return X

# Custom transformer for extracting datetime features
class DateTimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        try:
            X[self.datetime_col] = pd.to_datetime(X[self.datetime_col], errors='coerce')
            X['transaction_hour'] = X[self.datetime_col].dt.hour
            X['transaction_day'] = X[self.datetime_col].dt.day
            X['transaction_month'] = X[self.datetime_col].dt.month
            X['transaction_year'] = X[self.datetime_col].dt.year
        except Exception as e:
            print(f"Error extracting datetime features: {e}")
        return X

# Utility function to determine which columns to one-hot encode and which to label encode
def get_categorical_encoding_strategy(df, max_onehot=50):
    onehot_cols = []
    label_cols = []
    for col in df.select_dtypes(include='object').columns:
        n_unique = df[col].nunique()
        if n_unique <= max_onehot:
            onehot_cols.append(col)
        else:
            label_cols.append(col)
    return onehot_cols, label_cols

# Modified CategoricalEncoder to handle both one-hot and label encoding
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, onehot_cols=None, label_cols=None):
        self.onehot_cols = onehot_cols or []
        self.label_cols = label_cols or []
        self.oh_encoder = None
        self.encoders = {}

    def fit(self, X, y=None):
        if self.onehot_cols:
            self.oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            self.oh_encoder.fit(X[self.onehot_cols])
        for col in self.label_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        # One-hot encode low-cardinality columns
        if self.onehot_cols and self.oh_encoder is not None:
            try:
                oh = self.oh_encoder.transform(X[self.onehot_cols])
                oh_df = pd.DataFrame(oh, columns=self.oh_encoder.get_feature_names_out(self.onehot_cols), index=X.index)
                X = X.drop(columns=self.onehot_cols)
                X = pd.concat([X, oh_df], axis=1)
            except Exception as e:
                print(f"Error in OneHotEncoding: {e}")
        # Label encode high-cardinality columns
        for col in self.label_cols:
            try:
                X[col] = self.encoders[col].transform(X[col].astype(str))
            except Exception as e:
                print(f"Error in LabelEncoding column {col}: {e}")
        return X

# Custom transformer for missing value handling
class MissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean', fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.imputer = None

    def fit(self, X, y=None):
        num_cols = X.select_dtypes(include=[np.number]).columns
        if self.strategy in ['mean', 'median', 'most_frequent', 'constant']:
            self.imputer = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value)
            self.imputer.fit(X[num_cols])
        return self

    def transform(self, X):
        X = X.copy()
        num_cols = X.select_dtypes(include=[np.number]).columns
        if self.imputer is not None:
            try:
                X[num_cols] = self.imputer.transform(X[num_cols])
            except Exception as e:
                print(f"Error in missing value imputation: {e}")
        return X

# Custom transformer for normalization/standardization
class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, method='standard'):
        self.method = method
        self.scaler = None

    def fit(self, X, y=None):
        num_cols = X.select_dtypes(include=[np.number]).columns
        if self.method == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        self.scaler.fit(X[num_cols])
        self.num_cols = num_cols
        return self

    def transform(self, X):
        X = X.copy()
        try:
            X[self.num_cols] = self.scaler.transform(X[self.num_cols])
        except Exception as e:
            print(f"Error in scaling: {e}")
        return X

# Update the pipeline to determine encoding strategy at runtime

def process_data_pipeline(raw_data_path='data/raw', filename=None, max_onehot=50):
    """
    Loads raw data from the specified path, applies the feature engineering pipeline, and returns processed DataFrame.
    Automatically chooses one-hot or label encoding based on cardinality.
    """
    try:
        print(f"Loading data from: {os.path.abspath(raw_data_path)}")
        files = [f for f in os.listdir(raw_data_path) if f.endswith('.csv')]
        if not files:
            raise FileNotFoundError('No CSV files found in raw data directory.')
        if filename is None:
            filename = files[0]
        df = pd.read_csv(os.path.join(raw_data_path, filename))
        onehot_cols, label_cols = get_categorical_encoding_strategy(df, max_onehot=max_onehot)
        print(f"One-hot encoding columns: {onehot_cols}")
        print(f"Label encoding columns: {label_cols}")
        # Build the pipeline with the determined columns
        feature_pipeline = Pipeline([
            ("aggregate", AggregateFeatures()),
            ("datetime", DateTimeFeatures()),
            ("missing", MissingValueHandler(strategy='mean')),
            ("categorical", CategoricalEncoder(onehot_cols=onehot_cols, label_cols=label_cols)),
            ("scaler", Scaler(method='standard'))
        ])
        processed = feature_pipeline.fit_transform(df)
        return processed
    except Exception as e:
        print(f"Error in processing data pipeline: {e}")
        return None

# Example usage
if __name__ == "__main__":
    processed_df = process_data_pipeline(filename='data.csv')
    if processed_df is not None:
        print(processed_df.head())
# Note: To enable WOE/IV encoding, install xverse and uncomment the relevant code above. 