"""
Exploratory Data Analysis (EDA) Module for Credit Risk Dataset
- Comprehensive data analysis functions
- Visualization utilities
- Statistical analysis tools
- Feature engineering insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Tuple, List, Dict, Optional
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

# Configure pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)


class CreditRiskEDA:
    """
    Comprehensive EDA class for credit risk analysis
    """
    
    def __init__(self, data_path: str = None, target_col: str = 'FraudResult'):
        """
        Initialize EDA with data path and target column
        
        Args:
            data_path: Path to the CSV data file (if None, will try to find data.csv)
            target_col: Name of the target variable column
        """
        if data_path is None:
            # Try to find data.csv in common locations
            possible_paths = [
                'data/raw/data.csv',
                '../data/raw/data.csv',
                '../../data/raw/data.csv',
                'data.csv'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    data_path = path
                    break
            
            if data_path is None:
                raise FileNotFoundError("Could not find data.csv in common locations")
        
        self.data_path = data_path
        self.target_col = target_col
        self.df = None
        self.numerical_cols = []
        self.categorical_cols = []
        
    def load_data(self, use_preprocessing: bool = True) -> pd.DataFrame:
        """
        Load and optionally preprocess the dataset
        
        Args:
            use_preprocessing: Whether to use data_processing module for preprocessing
            
        Returns:
            Loaded DataFrame
        """
        print("=== LOADING AND EXPLORING DATA ===")
        print(f"Loading data from: {self.data_path}")
        
        if use_preprocessing:
            try:
                # Import data_processing module
                from . import data_processing
                self.df = data_processing.load_raw_data(filename=os.path.basename(self.data_path))
                self.df = data_processing.preprocess_dataframe(self.df)
                print("✅ Data loaded and preprocessed using data_processing module")
            except ImportError:
                print("⚠️  data_processing module not found, loading raw data")
                self.df = pd.read_csv(self.data_path)
            except Exception as e:
                print(f"⚠️  Error with data_processing module: {e}")
                print("Loading raw data instead")
                self.df = pd.read_csv(self.data_path)
        else:
            self.df = pd.read_csv(self.data_path)
        
        # Identify column types
        self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in self.numerical_cols:
            self.numerical_cols.remove(self.target_col)
        
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Print basic information
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print(f"Data types:\n{self.df.dtypes}")
        print(f"\nTotal records: {len(self.df):,}")
        print(f"Total features: {len(self.df.columns)}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return self.df
    
    def analyze_missing_values(self) -> pd.DataFrame:
        """
        Analyze missing values in the dataset
        
        Returns:
            DataFrame with missing values summary
        """
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Values': missing_data,
            'Percentage': missing_percent
        }).sort_values('Missing Values', ascending=False)
        
        print("\nMissing values:")
        missing_cols = missing_df[missing_df['Missing Values'] > 0]
        if len(missing_cols) > 0:
            print(missing_cols)
        else:
            print("✅ No missing values found")
        
        return missing_df
    
    def analyze_target_variable(self, show_plots: bool = True) -> Dict:
        """
        Analyze the target variable distribution
        
        Args:
            show_plots: Whether to display visualizations
            
        Returns:
            Dictionary with target variable statistics
        """
        print(f"\n=== TARGET VARIABLE ANALYSIS ===")
        
        if self.target_col not in self.df.columns:
            print(f"⚠️  Target column '{self.target_col}' not found in dataset")
            return {}
        
        fraud_counts = self.df[self.target_col].value_counts()
        fraud_percentages = self.df[self.target_col].value_counts(normalize=True) * 100
        
        print(f"Value counts: {fraud_counts.to_dict()}")
        print(f"Percentages: {fraud_percentages.to_dict()}")
        print(f"Class imbalance ratio: {fraud_counts[0]/fraud_counts[1]:.2f}:1")
        
        if show_plots:
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Count plot
            sns.countplot(data=self.df, x=self.target_col, ax=ax1)
            ax1.set_title('Fraud Result Distribution')
            ax1.set_xlabel('Fraud Result (0=No, 1=Yes)')
            ax1.set_ylabel('Count')
            
            # Pie chart
            ax2.pie(fraud_counts.values, labels=['No Fraud', 'Fraud'], autopct='%1.1f%%', startangle=90)
            ax2.set_title('Fraud Result Distribution (%)')
            
            plt.tight_layout()
            plt.show()
        
        return {
            'counts': fraud_counts,
            'percentages': fraud_percentages,
            'imbalance_ratio': fraud_counts[0]/fraud_counts[1]
        }
    
    def analyze_numerical_features(self, show_plots: bool = True) -> List[str]:
        """
        Analyze numerical features
        
        Args:
            show_plots: Whether to display visualizations
            
        Returns:
            List of numerical column names
        """
        print(f"\n=== NUMERICAL FEATURES ANALYSIS ===")
        
        print(f"Numerical columns: {self.numerical_cols}")
        print(f"\nStatistical summary:")
        print(self.df[self.numerical_cols].describe())
        
        if show_plots and len(self.numerical_cols) > 0:
            # Distribution plots
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            axes = axes.ravel()
            
            for i, col in enumerate(self.numerical_cols[:6]):
                sns.histplot(data=self.df, x=col, hue=self.target_col if self.target_col in self.df.columns else None, 
                           bins=30, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'{col} Distribution')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Count')
            
            plt.tight_layout()
            plt.show()
            
            # Box plots
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            axes = axes.ravel()
            
            for i, col in enumerate(self.numerical_cols[:6]):
                if self.target_col in self.df.columns:
                    sns.boxplot(data=self.df, x=self.target_col, y=col, ax=axes[i])
                    axes[i].set_title(f'{col} by {self.target_col}')
                    axes[i].set_xlabel(self.target_col)
                else:
                    sns.boxplot(data=self.df, y=col, ax=axes[i])
                    axes[i].set_title(f'{col} Distribution')
                axes[i].set_ylabel(col)
            
            plt.tight_layout()
            plt.show()
        
        return self.numerical_cols
    
    def analyze_categorical_features(self, show_plots: bool = True) -> List[str]:
        """
        Analyze categorical features
        
        Args:
            show_plots: Whether to display visualizations
            
        Returns:
            List of categorical column names
        """
        print(f"\n=== CATEGORICAL FEATURES ANALYSIS ===")
        
        print(f"Categorical columns: {self.categorical_cols}")
        
        for col in self.categorical_cols:
            print(f"\n{col}:")
            print(f"Unique values: {self.df[col].nunique()}")
            print(f"Most common values:")
            print(self.df[col].value_counts().head())
            
            if self.target_col in self.df.columns:
                # Calculate fraud rate by category
                fraud_rate = self.df.groupby(col)[self.target_col].mean().sort_values(ascending=False)
                print(f"Fraud rate by {col} (top 5):")
                print(fraud_rate.head())
        
        if show_plots and len(self.categorical_cols) > 0 and self.target_col in self.df.columns:
            # Visualize categorical features vs fraud
            fig, axes = plt.subplots(2, 2, figsize=(20, 12))
            axes = axes.ravel()
            
            for i, col in enumerate(self.categorical_cols[:4]):
                fraud_by_cat = self.df.groupby(col)[self.target_col].sum().sort_values(ascending=False).head(10)
                fraud_by_cat.plot(kind='bar', ax=axes[i])
                axes[i].set_title(f'Fraud Count by {col} (Top 10)')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Fraud Count')
                axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.show()
        
        return self.categorical_cols
    
    def correlation_analysis(self, show_plots: bool = True) -> pd.Series:
        """
        Perform correlation analysis
        
        Args:
            show_plots: Whether to display visualizations
            
        Returns:
            Series with correlations to target variable
        """
        print(f"\n=== CORRELATION ANALYSIS ===")
        
        if self.target_col not in self.df.columns:
            print(f"⚠️  Target column '{self.target_col}' not found in dataset")
            return pd.Series()
        
        correlation_matrix = self.df[self.numerical_cols + [self.target_col]].corr()
        
        if show_plots:
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5)
            plt.title('Correlation Matrix of Numerical Features')
            plt.tight_layout()
            plt.show()
        
        # Show correlations with target variable
        target_correlations = correlation_matrix[self.target_col].sort_values(ascending=False)
        print(f"Correlations with target variable:")
        print(target_correlations)
        
        return target_correlations
    
    def time_series_analysis(self, time_col: str = 'TransactionStartTime', show_plots: bool = True) -> Tuple:
        """
        Analyze time-based patterns
        
        Args:
            time_col: Name of the time column
            show_plots: Whether to display visualizations
            
        Returns:
            Tuple of (fraud_by_hour, fraud_by_day, fraud_by_month)
        """
        print(f"\n=== TIME SERIES ANALYSIS ===")
        
        if time_col not in self.df.columns:
            print(f"⚠️  Time column '{time_col}' not found in dataset")
            return None, None, None
        
        # Convert to datetime
        self.df[time_col] = pd.to_datetime(self.df[time_col])
        
        # Extract time-based features
        self.df['hour'] = self.df[time_col].dt.hour
        self.df['day_of_week'] = self.df[time_col].dt.dayofweek
        self.df['month'] = self.df[time_col].dt.month
        self.df['day'] = self.df[time_col].dt.day
        
        if self.target_col in self.df.columns:
            # Analyze fraud patterns over time
            fraud_by_hour = self.df.groupby('hour')[self.target_col].mean()
            fraud_by_day = self.df.groupby('day_of_week')[self.target_col].mean()
            fraud_by_month = self.df.groupby('month')[self.target_col].mean()
            
            if show_plots:
                fig, axes = plt.subplots(2, 2, figsize=(20, 12))
                
                # Hour of day
                axes[0,0].plot(fraud_by_hour.index, fraud_by_hour.values, marker='o')
                axes[0,0].set_title('Fraud Rate by Hour of Day')
                axes[0,0].set_xlabel('Hour')
                axes[0,0].set_ylabel('Fraud Rate')
                
                # Day of week
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                axes[0,1].bar(range(7), fraud_by_day.values)
                axes[0,1].set_title('Fraud Rate by Day of Week')
                axes[0,1].set_xlabel('Day of Week')
                axes[0,1].set_ylabel('Fraud Rate')
                axes[0,1].set_xticks(range(7))
                axes[0,1].set_xticklabels(day_names, rotation=45)
                
                # Month
                axes[1,0].plot(fraud_by_month.index, fraud_by_month.values, marker='o')
                axes[1,0].set_title('Fraud Rate by Month')
                axes[1,0].set_xlabel('Month')
                axes[1,0].set_ylabel('Fraud Rate')
                
                # Day of month
                fraud_by_day_month = self.df.groupby('day')[self.target_col].mean()
                axes[1,1].plot(fraud_by_day_month.index, fraud_by_day_month.values, marker='o')
                axes[1,1].set_title('Fraud Rate by Day of Month')
                axes[1,1].set_xlabel('Day')
                axes[1,1].set_ylabel('Fraud Rate')
                
                plt.tight_layout()
                plt.show()
            
            return fraud_by_hour, fraud_by_day, fraud_by_month
        
        return None, None, None
    
    def outlier_analysis(self, show_plots: bool = True) -> Dict:
        """
        Detect and analyze outliers
        
        Args:
            show_plots: Whether to display visualizations
            
        Returns:
            Dictionary with outlier statistics
        """
        print(f"\n=== OUTLIER ANALYSIS ===")
        
        def detect_outliers(df, column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            return outliers, lower_bound, upper_bound
        
        outlier_stats = {}
        
        for col in self.numerical_cols:
            outliers, lower, upper = detect_outliers(self.df, col)
            outlier_percentage = (len(outliers) / len(self.df)) * 100
            print(f"{col}: {len(outliers)} outliers ({outlier_percentage:.2f}%)")
            print(f"  Range: [{lower:.2f}, {upper:.2f}]")
            
            outlier_stats[col] = {
                'count': len(outliers),
                'percentage': outlier_percentage,
                'lower_bound': lower,
                'upper_bound': upper
            }
        
        if show_plots:
            # Visualize outliers for key numerical features
            fig, axes = plt.subplots(2, 2, figsize=(20, 12))
            axes = axes.ravel()
            
            for i, col in enumerate(['Amount', 'Value', 'CountryCode', 'PricingStrategy'][:4]):
                if col in self.df.columns:
                    sns.boxplot(data=self.df, y=col, ax=axes[i])
                    axes[i].set_title(f'{col} - Outlier Detection')
            
            plt.tight_layout()
            plt.show()
        
        return outlier_stats
    
    def feature_engineering_insights(self, show_plots: bool = True) -> pd.DataFrame:
        """
        Create and analyze engineered features
        
        Args:
            show_plots: Whether to display visualizations
            
        Returns:
            DataFrame with engineered features
        """
        print(f"\n=== FEATURE ENGINEERING INSIGHTS ===")
        
        # Create some engineered features
        self.df['amount_abs'] = abs(self.df['Amount'])
        self.df['is_credit'] = (self.df['Amount'] < 0).astype(int)
        self.df['transaction_size_category'] = pd.cut(self.df['amount_abs'], 
                                                    bins=[0, 100, 500, 1000, 5000, float('inf')], 
                                                    labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])
        
        if self.target_col in self.df.columns:
            # Analyze engineered features
            fraud_by_size = self.df.groupby('transaction_size_category')[self.target_col].mean()
            fraud_by_type = self.df.groupby('is_credit')[self.target_col].mean()
            
            print("Fraud rate by transaction size:")
            print(fraud_by_size)
            print("\nFraud rate by transaction type:")
            print(fraud_by_type)
            
            if show_plots:
                # Visualize
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                fraud_by_size.plot(kind='bar', ax=ax1)
                ax1.set_title('Fraud Rate by Transaction Size')
                ax1.set_xlabel('Transaction Size Category')
                ax1.set_ylabel('Fraud Rate')
                ax1.tick_params(axis='x', rotation=45)
                
                fraud_by_type.plot(kind='bar', ax=ax2)
                ax2.set_title('Fraud Rate by Transaction Type')
                ax2.set_xlabel('Transaction Type (0=Debit, 1=Credit)')
                ax2.set_ylabel('Fraud Rate')
                
                plt.tight_layout()
                plt.show()
        
        return self.df
    
    def generate_summary_insights(self, fraud_counts: Dict, target_correlations: pd.Series, 
                                fraud_by_hour: pd.Series, fraud_by_day: pd.Series) -> Dict:
        """
        Generate comprehensive summary and insights
        
        Args:
            fraud_counts: Target variable statistics
            target_correlations: Correlations with target variable
            fraud_by_hour: Fraud rates by hour
            fraud_by_day: Fraud rates by day
            
        Returns:
            Dictionary with summary insights
        """
        print(f"\n=== KEY INSIGHTS SUMMARY ===")
        
        # Missing data analysis
        missing_df = self.analyze_missing_values()
        
        insights = {
            'dataset_overview': {
                'total_transactions': len(self.df),
                'fraud_rate': (self.df[self.target_col].mean()*100) if self.target_col in self.df.columns else None,
                'class_imbalance': fraud_counts.get('imbalance_ratio') if fraud_counts is not None else None
            },
            'data_quality': {
                'missing_columns': len(missing_df[missing_df['Missing Values'] > 0]),
                'missing_details': missing_df[missing_df['Missing Values'] > 0].to_dict('index')
            },
            'key_features': target_correlations.head(5).to_dict() if target_correlations is not None and not target_correlations.empty else {},
            'time_patterns': {
                'peak_hour': fraud_by_hour.idxmax() if fraud_by_hour is not None and not fraud_by_hour.empty else None,
                'peak_day': fraud_by_day.idxmax() if fraud_by_day is not None and not fraud_by_day.empty else None
            }
        }
        
        print("1. DATASET OVERVIEW:")
        print(f"   - Total transactions: {insights['dataset_overview']['total_transactions']:,}")
        if insights['dataset_overview']['fraud_rate'] is not None:
            print(f"   - Fraud rate: {insights['dataset_overview']['fraud_rate']:.2f}%")
        if insights['dataset_overview']['class_imbalance'] is not None:
            print(f"   - Class imbalance: {insights['dataset_overview']['class_imbalance']:.1f}:1")
        
        print("\n2. DATA QUALITY:")
        if insights['data_quality']['missing_columns'] > 0:
            print(f"   - Missing data in {insights['data_quality']['missing_columns']} columns")
            for col, details in insights['data_quality']['missing_details'].items():
                print(f"     * {col}: {details['Percentage']:.2f}%")
        else:
            print("   - No missing data found")
        
        print("\n3. KEY FEATURES FOR MODELING:")
        print("   - High correlation with target:")
        for col, corr in insights['key_features'].items():
            if col != self.target_col:
                print(f"     * {col}: {corr:.3f}")
        
        print("\n4. TIME PATTERNS:")
        if (
            insights['time_patterns']['peak_hour'] is not None and 
            isinstance(fraud_by_hour, pd.Series) and 
            not fraud_by_hour.empty
        ):
            print(f"   - Peak fraud hour: {insights['time_patterns']['peak_hour']} ({(fraud_by_hour.max()*100):.2f}%)")
        if (
            insights['time_patterns']['peak_day'] is not None and 
            isinstance(fraud_by_day, pd.Series) and 
            not fraud_by_day.empty
        ):
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            print(f"   - Peak fraud day: {day_names[insights['time_patterns']['peak_day']]} ({(fraud_by_day.max()*100):.2f}%)")
        
        print("\n5. RECOMMENDATIONS FOR MODEL DEVELOPMENT:")
        print("   - Handle class imbalance using techniques like SMOTE or class weights")
        print("   - Feature engineering: create time-based features, transaction categories")
        print("   - Consider ensemble methods for better performance")
        print("   - Implement cross-validation with stratification")
        print("   - Monitor model performance on different time periods")
        
        return insights
    
    def run_comprehensive_eda(self, use_preprocessing: bool = True, show_plots: bool = True) -> Dict:
        """
        Run comprehensive EDA analysis
        
        Args:
            use_preprocessing: Whether to use data_processing module
            show_plots: Whether to display visualizations
            
        Returns:
            Dictionary with all analysis results
        """
        print("CREDIT RISK PROBABILITY MODEL - COMPREHENSIVE EDA")
        print("=" * 60)
        
        # Load and explore data
        self.load_data(use_preprocessing=use_preprocessing)
        
        # Analyze target variable
        fraud_counts = self.analyze_target_variable(show_plots=show_plots)
        
        # Analyze numerical features
        numerical_cols = self.analyze_numerical_features(show_plots=show_plots)
        
        # Analyze categorical features
        categorical_cols = self.analyze_categorical_features(show_plots=show_plots)
        
        # Correlation analysis
        target_correlations = self.correlation_analysis(show_plots=show_plots)
        
        # Time series analysis
        fraud_by_hour, fraud_by_day, fraud_by_month = self.time_series_analysis(show_plots=show_plots)
        
        # Outlier analysis
        outlier_stats = self.outlier_analysis(show_plots=show_plots)
        
        # Feature engineering insights
        self.feature_engineering_insights(show_plots=show_plots)
        
        # Generate summary insights
        insights = self.generate_summary_insights(fraud_counts, target_correlations, fraud_by_hour, fraud_by_day)
        
        print("\n✅ EDA COMPLETED SUCCESSFULLY!")
        
        return {
            'fraud_counts': fraud_counts,
            'numerical_cols': numerical_cols,
            'categorical_cols': categorical_cols,
            'target_correlations': target_correlations,
            'time_patterns': {
                'fraud_by_hour': fraud_by_hour,
                'fraud_by_day': fraud_by_day,
                'fraud_by_month': fraud_by_month
            },
            'outlier_stats': outlier_stats,
            'insights': insights
        }


# Convenience functions for backward compatibility
def load_and_explore_data(data_path: str = '../data/raw/data.csv') -> pd.DataFrame:
    """Load and explore data (legacy function)"""
    eda = CreditRiskEDA(data_path)
    return eda.load_data()


def analyze_target_variable(df: pd.DataFrame, target_col: str = 'FraudResult') -> Dict:
    """Analyze target variable (legacy function)"""
    eda = CreditRiskEDA()
    eda.df = df
    eda.target_col = target_col
    return eda.analyze_target_variable()


def run_comprehensive_eda(data_path: str = '../data/raw/data.csv', 
                         target_col: str = 'FraudResult',
                         use_preprocessing: bool = True,
                         show_plots: bool = True) -> Dict:
    """Run comprehensive EDA (legacy function)"""
    eda = CreditRiskEDA(data_path, target_col)
    return eda.run_comprehensive_eda(use_preprocessing=use_preprocessing, show_plots=show_plots) 