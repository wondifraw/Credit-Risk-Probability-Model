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


# Credit Risk Probability Model

A production-ready machine learning project for predicting credit risk probabilities using FastAPI.

## Project Structure

```
├── data/                # Data storage (raw, processed)
├── notebooks/           # Jupyter notebooks for EDA
├── src/                 # Source code
│   ├── data_processing.py    # Feature engineering
│   ├── train.py              # Model training
│   ├── predict.py            # Inference
│   └── api/                  # FastAPI app
│       ├── main.py           # API entrypoint
│       └── pydantic_models.py# API schemas
├── tests/               # Unit tests
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker image definition
├── docker-compose.yml   # Docker Compose setup
└── README.md            # Project documentation
```

## Features
- Data processing and feature engineering
- Model training and evaluation
- REST API for inference (FastAPI)
- Dockerized for easy deployment
- Unit tests for reliability

## Getting Started

### 1. Clone the repository
```bash
git clone <[repo-url](https://github.com/wondifraw/Credit-Risk-Probability-Model.git)>
cd Credit-Risk-Probability-Model
```

### 2. Set up the environment
#### Option A: Local (with virtualenv)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
#### Option B: Docker
```bash
docker-compose up --build
```

### 3. Run the API
#### Locally
```bash
uvicorn src.api.main:app --reload
```
#### With Docker
```bash
docker-compose up
```

The API will be available at [http://localhost:8000](http://localhost:8000)

### 4. Run tests
```bash
pytest
```

## Data
- Place raw data in `data/raw/`
- Processed data will be saved in `data/processed/`

## Notebooks
- Use `notebooks/` for exploratory data analysis and prototyping.

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

