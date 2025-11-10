# Earthquake Tsunami Prediction Project

## Overview

This project focuses on predicting whether an earthquake will trigger a tsunami using machine learning classification techniques. The goal is to build a predictive model that can identify tsunami-generating earthquakes based on seismic and geographic features, which has significant implications for early warning systems and disaster preparedness.

The project implements a complete machine learning pipeline from exploratory data analysis to model deployment, including hyperparameter optimization, experiment tracking with MLflow, and predictive analysis for future scenarios.

## Problem Statement

Earthquakes are one of the primary causes of tsunamis, but not all earthquakes generate these devastating waves. The ability to accurately predict tsunami occurrence following an earthquake is crucial for:

- **Early Warning Systems**: Providing timely alerts to coastal communities
- **Disaster Preparedness**: Enabling better resource allocation and evacuation planning
- **Risk Assessment**: Understanding the relationship between earthquake characteristics and tsunami generation

This is a **binary classification problem** where we aim to predict whether an earthquake (labeled as `tsunami = 1`) will cause a tsunami or not (`tsunami = 0`).

## Dataset

### Source
The dataset `earthquake_data_tsunami.csv` contains historical earthquake records with associated tsunami information.

### Dataset Characteristics

- **Total Samples**: 782 earthquakes
- **Time Period**: 2001-2022
- **Target Variable**: `tsunami` (binary: 0 = no tsunami, 1 = tsunami occurred)
- **Class Distribution**: 
  - No tsunami: 478 samples (61.1%)
  - Tsunami: 304 samples (38.9%)
  - *Note: The dataset is slightly imbalanced, which should be considered during model training*

### Features Description

The dataset contains 12 predictive features:

| Feature | Description | Type |
|---------|-------------|------|
| `magnitude` | Earthquake magnitude (Richter scale) | Continuous (6.5 - 9.1) |
| `cdi` | Community Decimal Intensity | Integer (0-9) |
| `mmi` | Modified Mercalli Intensity | Integer (1-9) |
| `sig` | Significance index | Continuous |
| `nst` | Number of seismic stations used | Integer |
| `dmin` | Minimum distance to epicenter (km) | Continuous |
| `gap` | Azimuthal gap (degrees) | Continuous |
| `depth` | Earthquake depth (km) | Continuous |
| `latitude` | Geographic latitude | Continuous |
| `longitude` | Geographic longitude | Continuous |
| `Year` | Year of occurrence | Integer (2001-2022) |
| `Month` | Month of occurrence | Integer (1-12) |

### Data Quality

- **Missing Values**: None (complete dataset)
- **Data Types**: All features are numeric
- **Outliers**: Some features may contain extreme values that require careful handling

## Project Structure

```
ML_II_repo/
├── data/
│   └── earthquake_data_tsunami.csv    # Main dataset
├── Codes (.ipynb & .py)/
│   └── test.ipynb                     # Complete ML pipeline notebook
├── outputs/                           # Model outputs and predictions
│   ├── best_pipeline.joblib          # Best trained model
│   ├── all_pipelines.joblib           # All trained models
│   ├── model_results.csv              # Model performance metrics
│   ├── pipeline_metadata.joblib       # Model metadata
│   ├── predictions_2023_all.csv       # All 2023 scenario predictions
│   └── predictions_2023_top10.csv     # Top 10 most probable scenarios
├── mlruns/                            # MLflow experiment tracking
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Notebook Structure

The main notebook (`test.ipynb`) is organized into 7 comprehensive parts:

### Part 1: Exploratory Data Analysis
- Initial data inspection and summary statistics
- Feature engineering (magnitude bins, geographic features, magnitude-depth ratios)
- Univariate analysis: feature distributions by class
- Bivariate analysis: feature relationships and correlations
- Temporal analysis: trends over time
- Magnitude analysis: distribution and patterns
- Geographic analysis: spatial distribution and country identification
- Statistical summary by class

### Part 2: Machine Learning Pipeline Setup
- Feature selection and preprocessing pipeline configuration
- Standardization and scaling strategies
- Pipeline architecture definition

### Part 3: Data Splitting and Model Configuration
- Temporal train-test split (training on data before 2020, testing on 2020-2022)
- Model definitions (Logistic Regression, Random Forest, Gradient Boosting, SVM, K-Nearest Neighbors)
- Complete pipeline assembly (preprocessing + model)

### Part 4: Model Training
- Training all baseline models
- MLflow integration for experiment tracking
- Hyperparameter logging
- Metric logging (accuracy, precision, recall, F1-score, ROC-AUC)
- Model versioning

### Part 5: Model Evaluation
- Performance comparison across models
- Hyperparameter optimization (Grid Search, Random Search, or Optuna)
- MLflow tracking for optimization runs
- Model comparison and selection
- Visualization of results (confusion matrices, ROC curves, precision-recall curves)

### Part 6: Model Persistence and Deployment
- Saving best model pipeline
- Saving all trained models
- Exporting results and metadata
- MLflow model registry integration
- Instructions for model loading and reuse

### Part 7: Predictive Analysis for 2023
- Generating realistic earthquake scenarios for 2023
- Predicting tsunami probabilities using the best model
- Identifying top 10 most probable tsunami scenarios
- Country identification for predicted scenarios
- Detailed analysis and visualization of high-risk scenarios
- Exporting predictions to CSV

## Methodology

### Approach

1. **Data Exploration**: Understanding feature distributions, correlations, and class separability
2. **Feature Engineering**: Creating derived features (magnitude bins, geographic features, ratios)
3. **Temporal Validation**: Using time-based train-test split to simulate real-world deployment
4. **Model Development**: Training and evaluating various classification algorithms
5. **Hyperparameter Optimization**: Using Grid Search, Random Search, or Optuna
6. **Experiment Tracking**: MLflow for versioning and reproducibility
7. **Model Evaluation**: Using appropriate metrics for imbalanced classification
8. **Predictive Analysis**: Generating future scenarios and identifying high-risk events

### Models Implemented

- **Logistic Regression**: Linear baseline model
- **Random Forest**: Ensemble method with feature importance
- **Gradient Boosting**: Sequential ensemble learning
- **Support Vector Machines (SVM)**: Kernel-based classification
- **K-Nearest Neighbors (KNN)**: Instance-based learning

### Evaluation Metrics

Given the slightly imbalanced nature of the dataset, the following metrics are used:

- **Accuracy**: Overall prediction correctness
- **Precision**: Important for minimizing false alarms
- **Recall**: Critical for not missing actual tsunami events
- **F1-Score**: Balanced metric combining precision and recall
- **ROC-AUC**: Overall model performance across different thresholds
- **Confusion Matrix**: Detailed breakdown of prediction errors

### Experiment Tracking with MLflow

The project uses MLflow for comprehensive experiment tracking:

- **Hyperparameter Logging**: All model hyperparameters are logged for each run
- **Metric Tracking**: Training and test metrics are automatically logged
- **Model Versioning**: Each trained model is saved and versioned
- **Run Comparison**: Easy comparison between baseline and optimized models
- **Reproducibility**: Complete tracking of all experiment parameters

To view MLflow UI:
```bash
mlflow ui --backend-store-uri mlruns/
```

## Installation and Usage

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd ML_II_repo
```

2. **Create and activate virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Running the Notebook

1. **Start Jupyter**:
```bash
jupyter notebook
```

2. **Open the notebook**: Navigate to `Codes (.ipynb & .py)/test.ipynb`

3. **Run all cells**: The first cell will install all dependencies from `requirements.txt`

4. **View MLflow experiments**:
```bash
mlflow ui --backend-store-uri mlruns/
```

### Using the Trained Model

After training, you can load and use the best model:

```python
import joblib
from pathlib import Path

# Load the best pipeline
pipeline = joblib.load('outputs/best_pipeline.joblib')

# Load metadata
metadata = joblib.load('outputs/pipeline_metadata.joblib')

# Prepare new data (must match FEATURE_COLS)
new_data = pd.DataFrame({
    'magnitude': [8.5],
    'depth': [10.0],
    'latitude': [35.0],
    'longitude': [140.0],
    # ... other features
})

# Make predictions
predictions = pipeline.predict(new_data)
probabilities = pipeline.predict_proba(new_data)[:, 1]
```

### Loading from MLflow

```python
import mlflow.sklearn

# Load model from MLflow
model = mlflow.sklearn.load_model("runs:/<run_id>/model")
```

## Outputs

The project generates several output files:

- **`best_pipeline.joblib`**: The best performing model pipeline
- **`all_pipelines.joblib`**: All trained model pipelines
- **`model_results.csv`**: Performance metrics for all models
- **`pipeline_metadata.joblib`**: Model configuration and metadata
- **`predictions_2023_all.csv`**: All generated scenarios with predictions
- **`predictions_2023_top10.csv`**: Top 10 most probable tsunami scenarios for 2023

## Key Features

### Feature Engineering

- **Magnitude Binning**: Categorizing earthquakes by magnitude ranges
- **Geographic Features**: Absolute latitude, country identification
- **Derived Ratios**: Magnitude-to-depth ratios
- **Temporal Features**: Year and month encoding

### Preprocessing Pipeline

- **Standardization**: Robust scaling for numerical features
- **Missing Value Handling**: Median imputation
- **Feature Selection**: Focused on most predictive features

### Hyperparameter Optimization

Three optimization methods are available:
- **Grid Search**: Exhaustive search over parameter grid
- **Random Search**: Randomized parameter sampling
- **Optuna**: Bayesian optimization for efficient search

## Results

The project evaluates multiple models and selects the best performer based on F1-score and ROC-AUC metrics. The optimized model is then used for:

1. **Performance Evaluation**: Comprehensive metrics on test set
2. **Predictive Analysis**: Identifying high-risk scenarios for 2023
3. **Geographic Insights**: Country-level risk assessment

## Future Work

- [ ] Implement advanced feature engineering (wavelet transforms, time-series features)
- [ ] Address class imbalance using techniques like SMOTE or class weighting
- [ ] Implement ensemble methods combining multiple models
- [ ] Create interactive visualization dashboard
- [ ] Implement model interpretability analysis (SHAP values, feature importance)
- [ ] Develop a production-ready API for real-time predictions
- [ ] Cross-validation strategies for more robust evaluation
- [ ] Integration with real-time earthquake monitoring systems

## Challenges Addressed

1. **Class Imbalance**: The dataset has more non-tsunami events (61.1%) than tsunami events (38.9%) - addressed through appropriate metrics
2. **Feature Selection**: Comprehensive EDA to identify most predictive features
3. **Geographic Patterns**: Reverse geocoding to identify country-level patterns
4. **Temporal Considerations**: Time-based train-test split for realistic evaluation
5. **Model Interpretability**: Feature importance analysis and detailed predictions
6. **Reproducibility**: MLflow tracking ensures all experiments are reproducible

## Dependencies

All dependencies are listed in `requirements.txt`:

- pandas>=2.0.0
- numpy>=1.24.0
- scikit-learn>=1.3.0
- joblib>=1.3.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- reverse-geocoder>=1.5.0
- pycountry>=23.0.0
- scipy>=1.10.0
- optuna>=3.0.0
- mlflow>=2.0.0
- jupyter>=1.0.0
- ipython>=8.0.0

## References

- USGS Earthquake Catalog
- NOAA Tsunami Database
- Seismological research on tsunami generation mechanisms
- MLflow Documentation: https://mlflow.org/
- Scikit-learn Documentation: https://scikit-learn.org/