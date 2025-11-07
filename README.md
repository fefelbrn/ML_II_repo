# Earthquake Tsunami Prediction Project

## Overview

This project focuses on predicting whether an earthquake will trigger a tsunami using machine learning classification techniques. The goal is to build a predictive model that can identify tsunami-generating earthquakes based on seismic and geographic features, which has significant implications for early warning systems and disaster preparedness.

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

### Key Insights from Exploratory Data Analysis

1. **Magnitude Range**: Earthquakes in the dataset range from 6.5 to 9.1 on the Richter scale, focusing on significant seismic events
2. **Geographic Distribution**: Earthquakes are distributed globally, with concentrations in seismically active regions (Pacific Ring of Fire, etc.)
3. **Temporal Coverage**: Data spans 22 years, allowing for temporal analysis and potential time-based splitting strategies
4. **Feature Engineering Opportunities**: 
   - Magnitude-to-depth ratios
   - Geographic clustering (country/region identification)
   - Magnitude binning
   - Temporal features (seasonality, trends)

## Project Structure

```
ML_II_repo/
├── data/
│   └── earthquake_data_tsunami.csv    # Main dataset
├── notebooks/
│   └── test.ipynb                      # Exploratory data analysis and feature engineering
├── src/                                # Source code (to be developed)
├── outputs/                            # Model outputs, predictions, visualizations
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Methodology

### Approach

1. **Data Exploration**: Understanding feature distributions, correlations, and class separability
2. **Feature Engineering**: Creating derived features that may improve predictive power
3. **Model Development**: Training and evaluating various classification algorithms
4. **Model Evaluation**: Using appropriate metrics for imbalanced classification (precision, recall, F1-score, ROC-AUC)
5. **Model Selection**: Choosing the best-performing model based on validation results

### Potential Models

- Logistic Regression
- Random Forest
- Gradient Boosting (XGBoost, LightGBM)
- Support Vector Machines
- Neural Networks

### Evaluation Metrics

Given the slightly imbalanced nature of the dataset, the following metrics are recommended:

- **Precision**: Important for minimizing false alarms
- **Recall**: Critical for not missing actual tsunami events
- **F1-Score**: Balanced metric combining precision and recall
- **ROC-AUC**: Overall model performance across different thresholds
- **Confusion Matrix**: Detailed breakdown of prediction errors

## Usage

### Prerequisites

Install required dependencies:

```bash
pip install -r requirements.txt
```

### Data Loading

```python
import pandas as pd

df = pd.read_csv('data/earthquake_data_tsunami.csv')
```

### Basic Exploration

```python
# Dataset shape
print(f"Shape: {df.shape}")

# Target distribution
print(df['tsunami'].value_counts())

# Basic statistics
print(df.describe())
```

## Future Work

- [ ] Implement feature engineering pipeline
- [ ] Develop and compare multiple classification models
- [ ] Address class imbalance using techniques like SMOTE or class weighting
- [ ] Perform hyperparameter tuning
- [ ] Create visualization dashboard
- [ ] Implement model interpretability analysis (SHAP values, feature importance)
- [ ] Develop a production-ready prediction pipeline
- [ ] Cross-validation and temporal validation strategies

## Challenges

1. **Class Imbalance**: The dataset has more non-tsunami events (61.1%) than tsunami events (38.9%)
2. **Feature Selection**: Determining which features are most predictive of tsunami occurrence
3. **Geographic Patterns**: Understanding regional variations in tsunami generation
4. **Temporal Considerations**: Deciding whether to use temporal features or perform time-based validation
5. **Model Interpretability**: Ensuring the model's predictions are explainable for disaster management applications

## References

- USGS Earthquake Catalog
- NOAA Tsunami Database
- Seismological research on tsunami generation mechanisms