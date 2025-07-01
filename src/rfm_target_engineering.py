import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------------
# RFM Calculation
# -----------------------------
def calculate_rfm(df, customer_id_col='CustomerId', date_col='TransactionStartTime', amount_col='Amount', snapshot_date=None):
    """
    Calculate Recency, Frequency, and Monetary (RFM) metrics for each customer.
    Args:
        df: DataFrame with transaction data
        customer_id_col: column name for customer ID
        date_col: column name for transaction date
        amount_col: column name for transaction amount
        snapshot_date: reference date for recency calculation (string or pd.Timestamp)
    Returns:
        rfm_df: DataFrame with RFM metrics per customer
    """
    try:
        # Ensure date column is datetime
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        if snapshot_date is None:
            snapshot_date = df[date_col].max() + pd.Timedelta(days=1)
        else:
            snapshot_date = pd.to_datetime(snapshot_date)
        # Group by customer and calculate RFM
        rfm = df.groupby(customer_id_col).agg({
            date_col: lambda x: (snapshot_date - x.max()).days,  # Recency
            amount_col: ['count', 'sum']                        # Frequency, Monetary
        })
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        rfm = rfm.reset_index()
        return rfm
    except Exception as e:
        print(f"Error in RFM calculation: {e}")
        return None

# -----------------------------
# RFM Clustering
# -----------------------------
def cluster_rfm(rfm_df, n_clusters=3, random_state=42):
    """
    Scale RFM features and cluster customers using KMeans.
    Args:
        rfm_df: DataFrame with RFM metrics
        n_clusters: number of clusters
        random_state: random state for reproducibility
    Returns:
        rfm_df: DataFrame with added 'cluster' column
    """
    try:
        features = ['Recency', 'Frequency', 'Monetary']
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_df[features])
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        rfm_df['cluster'] = kmeans.fit_predict(rfm_scaled)
        return rfm_df
    except Exception as e:
        print(f"Error in RFM clustering: {e}")
        return None

# -----------------------------
# High-Risk Label Assignment
# -----------------------------
def assign_high_risk_label(rfm_df, cluster_col='cluster'):
    """
    Assign is_high_risk=1 to the cluster with lowest engagement (high recency, low frequency, low monetary).
    Args:
        rfm_df: DataFrame with RFM and cluster columns
        cluster_col: name of the cluster column
    Returns:
        rfm_df: DataFrame with added 'is_high_risk' column
    """
    try:
        # Compute mean RFM for each cluster
        cluster_stats = rfm_df.groupby(cluster_col)[['Recency', 'Frequency', 'Monetary']].mean()
        # High risk: highest recency, lowest frequency & monetary
        high_risk_cluster = cluster_stats.sort_values(['Recency', 'Frequency', 'Monetary'], ascending=[False, True, True]).index[0]
        rfm_df['is_high_risk'] = (rfm_df[cluster_col] == high_risk_cluster).astype(int)
        return rfm_df
    except Exception as e:
        print(f"Error in assigning high-risk label: {e}")
        return None

# -----------------------------
# Merge High-Risk Label
# -----------------------------
def merge_high_risk(df, rfm_df, customer_id_col='CustomerId'):
    """
    Merge is_high_risk label back to the main DataFrame.
    Args:
        df: main DataFrame
        rfm_df: DataFrame with customer_id_col and is_high_risk
        customer_id_col: column name for customer ID
    Returns:
        merged DataFrame
    """
    try:
        result = df.merge(rfm_df[[customer_id_col, 'is_high_risk']], on=customer_id_col, how='left')
        # If a customer is missing from RFM, treat as not high risk
        result['is_high_risk'] = result['is_high_risk'].fillna(0).astype(int)
        return result
    except Exception as e:
        print(f"Error in merging high-risk label: {e}")
        return df 