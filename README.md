## Credit Scoring Business Understanding

### 1. Basel II Accord: Why Interpretability and Documentation Matter
The Basel II Accord is a global regulatory framework that sets standards for risk measurement and capital adequacy in financial institutions. It requires banks to quantify their credit risk exposure and hold sufficient capital reserves to cover potential losses. This regulatory environment places a premium on the use of interpretable and well-documented models for several reasons:

- **Regulatory Scrutiny:** Supervisors and auditors must be able to understand, validate, and challenge the risk models used by banks. Black-box models that cannot be explained are less likely to be accepted by regulators.
- **Transparency and Accountability:** Interpretable models allow institutions to trace how individual features contribute to risk scores, making it easier to justify lending decisions to both regulators and customers.
- **Auditability:** Well-documented models facilitate internal and external audits, ensuring that the model development process, assumptions, and limitations are clearly recorded and reproducible.
- **Business Trust:** Transparent models build trust with stakeholders, including management, investors, and customers, by demonstrating that risk is being measured and managed responsibly.

For example, if a bank uses a logistic regression model with Weight of Evidence (WoE) encoding, it can clearly show how each customer attribute (such as income, age, or payment history) affects the probability of default. This level of clarity is essential for regulatory compliance and for defending model decisions in case of disputes. ([Statistica Sinica](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A28n535.pdf), [CFI](https://corporatefinanceinstitute.com/resources/commercial-lending/credit-risk/))

### 2. Proxy Variables for Default: Necessity and Business Risks
In many real-world datasets, a direct indicator of default (such as a "defaulted" flag) may be missing due to data limitations, privacy concerns, or reporting practices. In such cases, practitioners must create a **proxy variable**—for example, defining default as being 90 days past due, or using charge-off status as a substitute. This approach is necessary to enable supervised learning, but it introduces several business and regulatory risks:

- **Imperfect Representation:** The proxy may not fully capture the true economic event of default. For instance, some customers may be 90 days past due but eventually repay, while others may default without ever reaching that threshold.
- **Bias and Misclassification:** If the proxy is not well-aligned with actual default behavior, the model may systematically over- or under-predict risk for certain segments, leading to poor lending decisions.
- **Regulatory Risk:** Regulators may question the validity of the proxy, especially if it is not well-justified or if it diverges from industry standards. This can result in model rejection or the need for additional capital buffers.
- **Business Impact:** Decisions based on a flawed proxy can lead to financial losses, customer dissatisfaction, or reputational damage if creditworthy customers are denied or risky customers are approved.

**Example:** Suppose a lender uses "60+ days overdue" as a proxy for default. If economic conditions change and more customers temporarily fall behind but recover, the model may overestimate risk, leading to unnecessarily tight lending standards. Careful validation, documentation, and periodic review of the proxy are essential to mitigate these risks. ([World Bank Guidelines](https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf))

### 3. Model Choice: Interpretability vs. Predictive Power in Regulated Finance
Financial institutions face a critical decision when choosing between simple, interpretable models and complex, high-performance models:

- **Simple Models (e.g., Logistic Regression with WoE):**
  - **Advantages:**
    - Easy to explain to regulators, auditors, and business users.
    - Feature effects are transparent and can be directly interpreted.
    - Faster to develop, validate, and deploy.
    - Lower risk of overfitting and easier to monitor for drift.
  - **Limitations:**
    - May not capture complex, nonlinear relationships in the data.
    - Potentially lower predictive accuracy compared to advanced models.

- **Complex Models (e.g., Gradient Boosting Machines, Neural Networks):**
  - **Advantages:**
    - Can model intricate patterns and interactions, often resulting in higher predictive performance.
    - Useful for large, high-dimensional datasets where relationships are not obvious.
  - **Limitations:**
    - Often considered "black boxes"—difficult to interpret and explain.
    - Require more extensive documentation, validation, and sometimes post-hoc explainability tools (e.g., SHAP, LIME).
    - May face resistance from regulators and internal risk committees.

**Business and Regulatory Trade-off:**
In highly regulated environments, the marginal gains in predictive power from complex models may not justify the increased operational and compliance risks. For example, a bank may choose a slightly less accurate logistic regression model over a gradient boosting machine if it means faster regulatory approval and easier ongoing monitoring. However, in less regulated or highly competitive markets, the business may prioritize predictive accuracy to gain a competitive edge.

**Best Practice:** Many institutions adopt a hybrid approach—using complex models for internal risk monitoring and simple, interpretable models for regulatory reporting and decision-making. ([HKMA Alternative Credit Scoring](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf), [Towards Data Science](https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03))

## Exploratory Data Analysis (EDA) - Key Insights

### Dataset Overview
- **Total Transactions:** 95,662
- **Fraud Rate:** 0.20% (193 fraudulent transactions out of 95,662 total)
- **Class Imbalance:** 494.7:1 (highly imbalanced dataset)
- **Data Quality:** No missing values found in any column
- **Memory Usage:** 66.48 MB

### Top 5 Key Insights

#### 1. **Extreme Class Imbalance Requires Special Handling**
The dataset shows a severe class imbalance with only 0.20% fraudulent transactions. This creates significant challenges for model development:
- **Impact:** Standard accuracy metrics will be misleading (99.8% accuracy with a naive "predict all non-fraud" model)
- **Solution:** Need to use techniques like SMOTE, class weights, or specialized evaluation metrics (precision, recall, F1-score, AUC-ROC)

#### 2. **Transaction Amount and Value are Strong Predictors**
The correlation analysis reveals that `Value` (0.567) and `Amount` (0.557) have the strongest correlations with fraud:
- **Value:** Absolute transaction value shows the highest correlation with fraud
- **Amount:** Transaction amount (positive for debits, negative for credits) is also highly predictive
- **Implication:** Higher-value transactions are more likely to be fraudulent, suggesting fraudsters target larger amounts

#### 3. **Provider and Product Patterns Reveal Risk Hotspots**
Analysis of categorical features shows clear fraud patterns:
- **ProviderId_3:** Highest fraud rate (2.08%) - 20x higher than average
- **ProductId_9:** Highest fraud rate (17.65%) - 87x higher than average
- **ProductCategory 'transport':** Highest fraud rate (8.0%) - 40x higher than average
- **Implication:** Certain providers and products are significantly riskier and should be flagged for additional scrutiny

#### 4. **Temporal Patterns Suggest Fraud Timing**
Time-based analysis reveals interesting patterns:
- **Peak Fraud Hour:** 21:00 (1.01% fraud rate) - likely due to reduced monitoring during late hours
- **Peak Fraud Day:** Wednesday (0.31% fraud rate) - mid-week patterns
- **Peak Fraud Month:** February (0.34% fraud rate) - seasonal patterns
- **Implication:** Real-time fraud detection should be enhanced during high-risk time periods

#### 5. **Outlier Analysis Reveals Data Quality Issues**
Significant outliers detected in key numerical features:
- **Amount:** 25.55% outliers (24,441 transactions)
- **PricingStrategy:** 16.53% outliers (15,814 transactions)
- **Value:** 9.43% outliers (9,021 transactions)
- **Implication:** Need robust outlier handling strategies and investigation of extreme values

### Feature Engineering Recommendations

#### High-Priority Features
1. **Transaction Size Categories:** Create bins for transaction amounts (small, medium, large, very large)
2. **Time-Based Features:** Hour of day, day of week, month, season
3. **Provider Risk Score:** Binary flag or risk score for high-risk providers
4. **Product Risk Score:** Binary flag or risk score for high-risk products
5. **Transaction Type:** Credit vs. debit indicator (Amount < 0)

#### Secondary Features
1. **Customer Transaction History:** Count of previous transactions, average transaction size
2. **Batch Risk:** Fraud rate within the same batch
3. **Geographic Risk:** Country-based risk patterns (all transactions are from Uganda - CountryCode 256)
4. **Channel Risk:** Risk patterns by transaction channel

### Model Development Strategy

#### Recommended Approach
1. **Handle Class Imbalance:** Use SMOTE or class weights in model training
2. **Feature Selection:** Focus on high-correlation features (Value, Amount) and engineered features
3. **Model Choice:** Start with interpretable models (Logistic Regression) for regulatory compliance
4. **Evaluation Metrics:** Use precision, recall, F1-score, and AUC-ROC instead of accuracy
5. **Cross-Validation:** Implement stratified k-fold cross-validation
6. **Threshold Optimization:** Optimize decision threshold for business requirements

#### Risk Mitigation
1. **False Positive Management:** High false positives could lead to customer dissatisfaction
2. **False Negative Management:** High false negatives could lead to financial losses
3. **Model Monitoring:** Implement drift detection for feature distributions
4. **Regular Retraining:** Update model with new data to maintain performance

# Credit Risk Probability Model

A production-ready machine learning project for predicting credit risk probabilities using FastAPI, with comprehensive exploratory data analysis and robust data processing capabilities.

## Project Structure

```
├── data/                # Data storage (raw, processed)
├── notebooks/           # Jupyter notebooks for EDA
│   ├── 01-credit-risk-eda.ipynb    # Comprehensive EDA notebook
│   └── comprehensive-eda.ipynb      # Additional EDA analysis
├── scripts/             # Python scripts for analysis
│   ├── comprehensive_eda.py         # Comprehensive EDA script
│   └── test_eda_module.py           # EDA module testing
├── src/                 # Source code
│   ├── __init__.py           # Package initialization
│   ├── data_processing.py    # Data preprocessing and feature engineering
│   ├── eda.py                # Comprehensive EDA module
│   ├── train.py              # Model training
│   ├── predict.py            # Inference
│   └── api/                  # FastAPI application
│       ├── main.py           # API entrypoint
│       └── pydantic_models.py# API schemas
├── tests/               # Unit tests
│   └── test_data_processing.py  # Data processing tests
├── config.yaml          # Configuration file
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker image definition
├── docker-compose.yml   # Docker Compose setup
├── TASK_COMPLETION_SUMMARY.md  # Project completion summary
└── README.md            # Project documentation
```

## Features

### 🔍 **Comprehensive EDA Module**
- **Automated Data Analysis:** Complete exploratory data analysis with statistical insights
- **Visualization Suite:** Interactive plots for data distribution, correlations, and patterns
- **Time Series Analysis:** Fraud pattern detection across hours, days, and months
- **Outlier Detection:** Statistical outlier analysis with IQR methodology
- **Feature Engineering Insights:** Automated feature creation and analysis
- **Data Quality Assessment:** Missing value analysis and data integrity checks

### 🛠 **Robust Data Processing**
- **Flexible Data Loading:** Automatic data file detection and loading
- **Preprocessing Pipeline:** Comprehensive data cleaning and normalization
- **Feature Engineering:** Automated creation of derived features
- **Missing Value Handling:** Multiple imputation strategies (mean, median, mode, constant)
- **Outlier Management:** Configurable outlier detection and handling

### 🤖 **Machine Learning Pipeline**
- **Model Training:** Automated model training with cross-validation
- **Feature Selection:** Correlation-based feature importance analysis
- **Class Imbalance Handling:** SMOTE and class weight strategies
- **Model Evaluation:** Comprehensive metrics (precision, recall, F1-score, AUC-ROC)

### 🚀 **Production-Ready API**
- **FastAPI Framework:** High-performance REST API
- **Docker Support:** Containerized deployment
- **Input Validation:** Pydantic models for request/response validation
- **Error Handling:** Comprehensive error management
- **Documentation:** Auto-generated API documentation

### 🧪 **Testing & Quality Assurance**
- **Unit Tests:** Comprehensive test coverage for core modules
- **Module Testing:** Automated testing of EDA functionality
- **Data Validation:** Input data integrity checks
- **Error Recovery:** Graceful handling of edge cases

## Key Improvements Made

### ✅ **Enhanced EDA Capabilities**
- Fixed pandas truth value ambiguity issues in conditional statements
- Improved error handling for missing data scenarios
- Added comprehensive time-based pattern analysis
- Enhanced visualization capabilities with better styling

### ✅ **Robust Data Processing**
- Implemented flexible data loading with automatic path detection
- Added comprehensive missing value handling strategies
- Enhanced feature engineering with transaction categorization
- Improved outlier detection and analysis

### ✅ **Production Readiness**
- Added configuration management with `config.yaml`
- Enhanced Docker setup for easy deployment
- Improved API structure with proper validation
- Added comprehensive documentation

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/wondifraw/Credit-Risk-Probability-Model.git
cd Credit-Risk-Probability-Model
```

### 2. Set up the environment
#### Option A: Local Development
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build
```

### 3. Run Exploratory Data Analysis
```bash
# Run comprehensive EDA script
python scripts/comprehensive_eda.py

# Or use the Jupyter notebook
jupyter notebook notebooks/01-credit-risk-eda.ipynb
```

### 4. Start the API Server
#### Local Development
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Docker Deployment
```bash
docker-compose up
```

The API will be available at [http://localhost:8000](http://localhost:8000)

### 5. Run Tests
```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_data_processing.py

# Run EDA module tests
python scripts/test_eda_module.py
```

## API Documentation

Once the server is running, you can access:
- **Interactive API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative API Docs:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Data Management

### Data Structure
- **Raw Data:** Place CSV files in `data/raw/`
- **Processed Data:** Automatically saved to `data/processed/`
- **Configuration:** Modify `config.yaml` for data processing settings

### Supported Data Formats
- CSV files with automatic encoding detection
- Flexible column naming and data types
- Automatic datetime parsing for time-based analysis

## Development Workflow

### 1. **Data Exploration**
```bash
# Run comprehensive EDA
python scripts/comprehensive_eda.py
```

### 2. **Model Development**
```bash
# Train model with current data
python src/train.py
```

### 3. **Testing**
```bash
# Run all tests
pytest

# Test specific functionality
python scripts/test_eda_module.py
```


## Configuration

The project uses `config.yaml` for configuration management:

```yaml
# Data processing settings
data:
  raw_data_path: "data/raw/"
  processed_data_path: "data/processed/"
  
# Model settings
model:
  target_column: "FraudResult"
  test_size: 0.2
  random_state: 42

# API settings
api:
  host: "0.0.0.0"
  port: 8000
  debug: false
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


