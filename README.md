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

