"""
Earthquake Tsunami Prediction Project
============================================================
Complete ML Pipeline: From EDA to Predictive Analysis

This script contains the complete machine learning pipeline for predicting
tsunami occurrence based on earthquake characteristics.

NOTE: This script does NOT include visualization code (plots, charts, etc.)
      to avoid spamming plots when running from command line.
      If you want to see visualizations, please run the Jupyter notebook instead.

Structure:
  Part 1: Exploratory Data Analysis
  Part 2: Machine Learning Pipeline Setup
  Part 3: Data Splitting and Model Configuration
  Part 4: Model Training
  Part 5: Model Evaluation
  Part 6: Model Persistence and Deployment
  Part 7: Predictive Analysis for 2023
"""

# ============================================================================
# IMPORTS
# ============================================================================

from pathlib import Path
import optuna
import sys

import numpy as np
import pandas as pd

# Note: matplotlib import removed - visualizations are in the notebook only
# import matplotlib.pyplot as plt

from itertools import product
from scipy.stats import randint, uniform
import pycountry
import reverse_geocoder as rg

from mlflow import log_metric, log_param, log_params, log_artifacts
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
import joblib
import mlflow
import mlflow.sklearn

# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

# Determine project root and data path
current_dir = Path().resolve()
if current_dir.name == "Codes (.ipynb & .py)":
    project_root = current_dir.parent
elif (current_dir / "requirements.txt").exists():
    project_root = current_dir
elif (current_dir.parent / "requirements.txt").exists():
    project_root = current_dir.parent
else:
    project_root = Path("/Users/fefe/Desktop/Cours M1 Albert/Semestre 1/ML supervisé/Projet/ML_II_repo")

DATA_PATH = project_root / "data" / "earthquake_data_tsunami.csv"
OUTPUT_DIR = project_root / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded from: {DATA_PATH}")
print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print()

#======================================================================
# PART 1: EXPLORATORY DATA ANALYSIS
#======================================================================

# Initial Data Inspection
# We begin by examining the basic structure and characteristics of our dataset to get a first understanding of the data we're working with.

# Basic Statistical Summary
# This code generates descriptive statistics for all numeric columns in the dataset, including count, mean, standard deviation, min, max, and quartiles. This gave us an initial overview of the data distribution, helping us understand the scale, spread, and potential outliers in our features. It's the first step in understanding our data before diving deeper into more specific analyses. The resulting summary table shows statistical measures for each numeric column, revealing the central tendencies and variability of our earthquake features, which will guide our subsequent exploratory work.

df.describe()

# Display First Rows
# This code displays the first few rows of the dataset to inspect the actual data values and structure. This made it us to see the raw data format, column names, and sample values, which helps verify that the data was loaded correctly and gives us a sense of what each feature looks like in practice. The resulting table shows the first 5 rows with all columns visible, providing concrete examples of earthquake data entries that we'll be working with throughout the analysis.

df.head()

# Feature Engineering
# After the initial inspection, we create derived features that might capture important relationships not evident in the raw features.

# Check for Missing Values
# This code counts the number of missing (null/NaN) values in each column of the dataset. Missing data can significantly impact model performance, so identifying which columns have missing values and how many is crucial for deciding on appropriate imputation strategies or whether to exclude certain features entirely. The output shows the count of missing values per column, where columns with 0 missing values are clean and ready to use, while those with higher counts may require imputation techniques or removal from the analysis.

df.isnull().sum()

# Univariate Analysis: Feature Distributions by Class
# We now examine how each feature's distribution differs between tsunami and non-tsunami events to identify discriminative features.

# Prepare Features for Univariate Analysis
# This code prepares the dataset for univariate analysis by creating derived features and identifying all feature columns. We create abs_lat (absolute latitude) and mag_depth_ratio (magnitude to depth ratio) as these engineered features might capture relationships not evident in raw features. The code then identifies all feature columns excluding the target variable and temporal features, which will be used for our distribution analysis. The output shows the list of 12 features we'll analyze and provides a summary of the dataset shape and class distribution, confirming we have 478 non-tsunami events and 304 tsunami events for comparison.

df["abs_lat"] = df["latitude"].abs()
df["mag_depth_ratio"] = df["magnitude"] / (df["depth"] + 1.0)
TARGET_COL = "tsunami"
EXCLUDE_FEATS = {"tsunami", "Year", "Month"}
FEAT_COLS = [c for c in df.columns if c not in EXCLUDE_FEATS]
print(f"Features ({len(FEAT_COLS)}): {FEAT_COLS}")
print(df.shape, df[TARGET_COL].value_counts())

# Identify Numeric Features
# This code automatically identifies all numeric columns in the dataset, excluding the target variable and temporal features (Year, Month). We need to know which features are numeric for visualization and analysis purposes, as this list will be used for plotting histograms, boxplots, and other statistical visualizations that require numeric data. The output provides a list of 12 numeric feature names that can be used for further analysis and visualization, ensuring we work with the appropriate data types for each type of analysis.

df.describe()

# Bivariate Analysis: Feature Relationships
# We explore relationships between pairs of features and how they relate to the target variable using scatter plots to identify potential feature interactions.

# Temporal Analysis: Events by Year and Class
# This code creates a cross-tabulation table showing the number of events for each year, broken down by tsunami class (0 or 1), with missing combinations filled with 0. Temporal patterns can reveal trends over time, and understanding how tsunami events are distributed across years helps identify potential temporal dependencies in the data. This analysis is particularly important as it informs how we should split our train/test sets, deciding between a temporal split (respecting time order) versus a random split. The resulting table has years as rows and tsunami classes as columns, showing the count of events for each combination and revealing temporal trends and class distribution over time.

df.groupby(["tsunami"]).size()

# Display Year-Class Distribution Table
# This code displays the previously created cross-tabulation table showing events by year and class. Visualizing the table helps us quickly identify patterns, such as years with unusually high or low tsunami rates, or temporal trends in the data distribution. The formatted table clearly shows the distribution of events across years and classes, making temporal patterns easy to identify and interpret for our analysis.

year_cls = df.groupby(["Year", "tsunami"]).size().unstack(fill_value=0)

# Magnitude Analysis
# Magnitude is a key feature for earthquake characterization. We examine its distribution and relationship with tsunami occurrence.

# Visualize Temporal Trends
# This code creates a stacked bar chart showing the number of events per year, with different colors for tsunami (class 1) and non-tsunami (class 0) events. Visual representation makes it easier to spot temporal trends and patterns, so we could see if tsunami events are increasing or decreasing over time, or if certain years had unusual activity. This visualization informs our understanding of the data and potential temporal dependencies that might need to be accounted for in our models. The resulting bar chart has years on the x-axis and event counts on the y-axis, with each bar stacked to show the proportion of tsunami versus non-tsunami events per year, clearly revealing temporal patterns in the data.

year_cls

# Magnitude Distribution Analysis
# This code bins earthquake magnitudes into discrete ranges (0-1, 1-2, ..., 9-10) and counts how many events fall into each magnitude bin, then visualizes the overall distribution of magnitudes. Magnitude is a key feature for tsunami prediction, and understanding the distribution helps us see if most earthquakes are low-magnitude or if there's a good spread across different magnitudes. Binning also helps identify magnitude ranges that might be more associated with tsunamis, which could inform feature engineering or model interpretation. The resulting bar chart shows the count of events in each magnitude bin, revealing the distribution of earthquake magnitudes in our dataset and helping identify the most common magnitude ranges present in the data.

# Geographic Analysis
# Geographic location plays an important role in tsunami occurrence due to tectonic plate boundaries and coastal geography. We analyze the geographic distribution of events and tsunami rates by country.

# Plot removed - see notebook for visualizations
# plt.figure()
# year_cls.plot(kind="bar", stacked=True)
# plt.title("Nombre d'événements par an et par classe (tsunami 0/1)")
# plt.xlabel("Année"); plt.ylabel("Nombre d'événements")
# plt.tight_layout()
# plt.show()

# Magnitude Distribution by Class
# This code creates a stacked bar chart showing the distribution of events across magnitude bins, broken down by tsunami class, which reveals whether certain magnitude ranges are more associated with tsunamis. This analysis helps identify if there's a relationship between earthquake magnitude and tsunami occurrence, as if higher magnitude bins show more tsunamis, magnitude is likely a strong predictive feature. This visualization makes the relationship clear and helps us understand how magnitude relates to the target variable. The resulting stacked bar chart has magnitude bins on the x-axis, with each bar showing the proportion of tsunami versus non-tsunami events in that magnitude range, revealing whether higher magnitudes correlate with tsunami occurrence.

bins = np.arange(0, 11, 1)
labels = [f"{i}" for i in range(0, 10)]
df["_mag_bin"] = pd.cut(df["magnitude"].clip(lower=0, upper=10), bins=bins, labels=labels, include_lowest=True)
mag_counts = df["_mag_bin"].value_counts().sort_index()
# Plot removed - see notebook for visualizations
# plt.figure()
# mag_counts.plot(kind="bar")
# plt.title("Répartition par tranches de magnitude (globale)")
# plt.xlabel("Tranche de magnitude"); plt.ylabel("Nombre")
# plt.tight_layout()
# plt.show()

# Geographic Analysis: Country Identification
# This code uses reverse geocoding to convert latitude/longitude coordinates into country names, then counts how many events occurred in each country and creates a cross-tabulation of the top 15 countries by tsunami class. Geographic location is important for tsunami prediction as some regions are more prone to tsunamis, such as the Pacific Ring of Fire, and understanding which countries have the most events and their tsunami rates can reveal geographic patterns that might be useful as features in our models. The resulting table shows the top 15 countries with the most earthquake events, broken down by tsunami class, which reveals geographic hotspots and whether certain regions have higher tsunami rates that could be informative for prediction.

mag_cls = df.groupby(["_mag_bin", "tsunami"]).size().unstack(fill_value=0).reindex(labels)
# Plot removed - see notebook for visualizations
# plt.figure()
# mag_cls.plot(kind="bar", stacked=True)
# plt.title("Répartition par tranches de magnitude et classe (tsunami 0/1)")
# plt.xlabel("Tranche de magnitude"); plt.ylabel("Nombre")
# plt.tight_layout()
# plt.show()

# Statistical Summary by Class
# Finally, we provide a comprehensive quantitative comparison of all features between the two classes to identify the most discriminative features for our classification task.

# Top Countries by Event Count
# This code creates a bar chart showing the top 15 countries ranked by total number of earthquake events, regardless of tsunami class. This visualization helps identify which geographic regions dominate our dataset, and understanding the geographic distribution is important for assessing data representativeness and potential geographic biases that might affect model generalization. The resulting bar chart has countries on one axis and event counts on the other, clearly showing which countries have the most earthquake data in our dataset and helping us understand the geographic coverage of our training data.

coords = list(zip(df["latitude"].astype(float), df["longitude"].astype(float)))
results = rg.search(coords)
ccodes = [r["cc"] for r in results]
def cc_to_name(cc):
    try:
        return pycountry.countries.get(alpha_2=cc).name
    except Exception:
        return cc
countries = [cc_to_name(cc) for cc in ccodes]
df = df.copy()
df["country"] = countries
country_counts = df["country"].value_counts()
top15 = country_counts.head(15).index
country_cls = (df[df["country"].isin(top15)]
               .groupby(["country","tsunami"]).size()
               .unstack(fill_value=0)
               .loc[top15])
country_cls

# Country-Level Tsunami Rates
# This code creates a stacked bar chart showing the distribution of tsunami versus non-tsunami events for the top 15 countries, so we could see which countries have higher tsunami rates. This reveals geographic patterns in tsunami occurrence, as countries with higher proportions of tsunami events might share geographic characteristics such as coastal regions or tectonic plate boundaries that could be predictive. This analysis helps identify if country or region is a useful feature for our models. The resulting stacked bar chart has countries on the x-axis, with each bar showing the proportion of tsunami (class 1) versus non-tsunami (class 0) events, revealing which countries have higher tsunami rates and potential geographic risk factors that might influence tsunami occurrence.

# Plot removed - see notebook for visualizations
# plt.figure()
# country_counts.head(15).plot(kind="bar")
# plt.title("Top 15 pays par nombre d'événements (global)")
# plt.xlabel("Pays"); plt.ylabel("Nombre")
# plt.tight_layout()
# plt.show()

# Statistical Summary by Class
# This code calculates comprehensive statistics (mean, standard deviation, median, min, max) for all numeric features, grouped by tsunami class, providing a detailed comparison of feature distributions between classes. This quantitative comparison helps identify which features show the most significant differences between tsunami and non-tsunami events, where features with large differences in means or medians are likely to be more predictive. This analysis complements the visualizations with precise numerical comparisons that can guide feature selection and model development. The resulting multi-level table shows statistical measures for each numeric feature with separate columns for each class, where large differences in means or medians between classes indicate highly discriminative features for tsunami prediction.

# Plot removed - see notebook for visualizations
# plt.figure()
# country_cls.plot(kind="bar", stacked=True)
# plt.title("Top 15 pays — répartition par classe (tsunami 0/1)")
# plt.xlabel("Pays"); plt.ylabel("Nombre")
# plt.tight_layout()
# plt.show()

TARGET = "tsunami"
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [col for col in num_cols if col not in [TARGET, "Year", "Month"]]
group_stats = df.groupby(TARGET)[num_cols].agg(["mean","std","median","min","max"])
group_stats

#======================================================================
# PART 2: MACHINE LEARNING PIPELINE SETUP
#======================================================================

# Data Preparation
# Prepare the dataset for machine learning by defining features and target variable.

TARGET = "tsunami"
EXCLUDE_FEATS = {TARGET, "Year", "Month"}
FEATURE_COLS = [col for col in df.columns if col not in EXCLUDE_FEATS]
if 'country' in FEATURE_COLS:
    le = LabelEncoder()
    df['country_encoded'] = le.fit_transform(df['country'].astype(str))
    FEATURE_COLS = [col if col != 'country' else 'country_encoded' for col in FEATURE_COLS]
    print(f"Encoded 'country' column ({len(le.classes_)} unique countries)")

X = df[FEATURE_COLS].copy()
y = df[TARGET].copy()
print(f"Dataset shape: {X.shape}")
print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nTarget distribution:")
print(y.value_counts())
print(f"\nTarget balance: {y.mean():.2%} positive class")

# MLflow Setup
# Initialize MLflow for experiment tracking and model versioning. This will track all hyperparameters, metrics, and models for easy comparison and reproducibility.

current_dir = Path().resolve()
if current_dir.name == "Codes (.ipynb & .py)":
    project_root = current_dir.parent
elif (current_dir / "requirements.txt").exists():
    project_root = current_dir
elif (current_dir.parent / "requirements.txt").exists():
    project_root = current_dir.parent
else:
    project_root = Path("/Users/fefe/Desktop/Cours M1 Albert/Semestre 1/ML supervisé/Projet/ML_II_repo")
mlflow_dir = project_root / "mlruns"
mlflow.set_tracking_uri(f"file://{mlflow_dir}")
EXPERIMENT_NAME = "tsunami_prediction"
try:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    print(f"Created new MLflow experiment: {EXPERIMENT_NAME}")
except Exception:
    experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    print(f"Using existing MLflow experiment: {EXPERIMENT_NAME}")
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
print(f"Experiment ID: {experiment_id}")
print(f"MLflow runs will be saved to: {mlflow_dir}")

# Custom Transformers
# Create reusable transformers for feature engineering that can be integrated into the pipeline.

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature engineering.
    Creates derived features from existing ones.
    Works with pandas DataFrames and preserves column names.
    """
    def __init__(self):
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        """Fit the transformer and store feature names."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        X_transformed = self._transform(X)
        self.feature_names_ = list(X_transformed.columns)
        return self
    
    def _transform(self, X):
        """Internal transform method."""
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        if 'latitude' in X.columns:
            X['abs_lat'] = X['latitude'].abs()
        
        if 'magnitude' in X.columns and 'depth' in X.columns:
            X['mag_depth_ratio'] = X['magnitude'] / (X['depth'] + 1.0)
        
        if 'latitude' in X.columns and 'longitude' in X.columns:
            X['distance_from_origin'] = np.sqrt(X['latitude']**2 + X['longitude']**2)
        
        if 'magnitude' in X.columns:
            X['magnitude_squared'] = X['magnitude'] ** 2
        
        return X
    
    def transform(self, X):
        """Apply feature engineering transformations."""
        return self._transform(X)
fe_transformer = FeatureEngineeringTransformer()
X_test_transformed = fe_transformer.fit_transform(X)
print(f"Original features: {len(FEATURE_COLS)}")
print(f"Features after engineering: {X_test_transformed.shape[1]}")
new_features = set(X_test_transformed.columns) - set(FEATURE_COLS)
print(f"New features ({len(new_features)}): {new_features}")

# Preprocessing Pipeline
# Create a preprocessing pipeline that handles feature engineering and scaling.

preprocessing_pipeline = Pipeline([
    ('feature_engineering', FeatureEngineeringTransformer()),
    ('scaler', RobustScaler())
])
print("Preprocessing pipeline created!")
print("\nPipeline steps:")
for i, (name, step) in enumerate(preprocessing_pipeline.steps, 1):
    print(f"  {i}. {name}: {type(step).__name__}")

#======================================================================
# PART 3: DATA SPLITTING AND MODEL CONFIGURATION
#======================================================================

# Train-Test Split
# Split the data into training and testing sets. We'll use temporal splitting to respect the time order.

split_year = 2018
train_mask = df['Year'] < split_year
test_mask = df['Year'] >= split_year

# Sort training data by temporal order (Year, Month) for TimeSeriesSplit
# Create a combined dataframe to sort by Year and Month
train_df_temp = df[train_mask].copy()
train_df_sorted = train_df_temp.sort_values(['Year', 'Month']).reset_index(drop=True)
sorted_indices = train_df_sorted.index

# Reindex X and y based on sorted order
X_train = X[train_mask].iloc[sorted_indices].reset_index(drop=True)
y_train = y[train_mask].iloc[sorted_indices].reset_index(drop=True)
X_test = X[test_mask].copy()
y_test = y[test_mask].copy()

print(f"Temporal split (split year: {split_year}):")
print(f"  Training set: {X_train.shape[0]} samples ({train_mask.sum() / len(df):.1%})")
print(f"  Test set: {X_test.shape[0]} samples ({test_mask.sum() / len(df):.1%})")
print(f"\nTraining target distribution:")
print(y_train.value_counts())
print(f"\nTest target distribution:")
print(y_test.value_counts())
print(f"\nTraining data sorted by Year and Month for TimeSeriesSplit cross-validation")

# Model Definitions
# Define multiple classification models to compare performance.

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    ),
    'SVM': SVC(
        kernel='rbf',
        probability=True,
        random_state=42,
        class_weight='balanced'
    ),
    'K-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=5,
        weights='distance'
    )
}
print(f"Defined {len(models)} models:")
for name in models.keys():
    print(f"  - {name}")

# Complete Pipeline (Preprocessing + Model)
# Create complete pipelines that combine preprocessing and model for each algorithm.

pipelines = {}
for model_name, model in models.items():
    pipelines[model_name] = Pipeline([
        ('preprocessing', preprocessing_pipeline),
        ('classifier', model)
    ])
print("Complete pipelines created:")
for name, pipeline in pipelines.items():
    print(f"\n{name}:")
    for i, (step_name, step) in enumerate(pipeline.steps, 1):
        print(f"  {i}. {step_name}: {type(step).__name__}")

#======================================================================
# PART 4: MODEL TRAINING
#======================================================================

# Model Training
# Train all models and evaluate their performance.

results = {}
trained_pipelines = {}
print("Training models...")
print("=" * 100)
for model_name, pipeline in pipelines.items():
    print(f"\nTraining {model_name}...")
    
    with mlflow.start_run(run_name=f"{model_name}_baseline"):
        classifier = pipeline.named_steps['classifier']
        hyperparams = {}
        
        if hasattr(classifier, 'get_params'):
            model_params = classifier.get_params()
            for key, value in model_params.items():
                if key not in ['random_state', 'n_jobs', 'verbose']:
                    hyperparams[f"classifier__{key}"] = str(value)
        
        mlflow.log_params(hyperparams)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("split_year", split_year)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        pipeline.fit(X_train, y_train)
        trained_pipelines[model_name] = pipeline
        
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_roc_auc", roc_auc)
        
        mlflow.sklearn.log_model(pipeline, "model")
        
        run_id = mlflow.active_run().info.run_id
        print(f"Training complete | MLflow Run ID: {run_id}")
print("\n" + "=" * 100)
print("All models trained successfully!")
print(f"\nView MLflow UI: mlflow ui --backend-store-uri {mlflow_dir}")

#======================================================================
# PART 5: MODEL EVALUATION
#======================================================================

# Model Evaluation
# Compare model performance using comprehensive metrics.

results_df = pd.DataFrame({
    model_name: {
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1'],
        'ROC-AUC': metrics['roc_auc']
    }
    for model_name, metrics in results.items()
}).T
results_df = results_df.sort_values('F1-Score', ascending=False)
print("Model Performance Comparison:")
print("=" * 100)
print(results_df.round(4))

best_model = results_df.index[0]
print(f"Best model (by F1-Score): {best_model}")
print(f"F1-Score: {results_df.loc[best_model, 'F1-Score']:.4f}")
print(f"ROC-AUC: {results_df.loc[best_model, 'ROC-AUC']:.4f}")

# Hyperparameter Optimization
# Optimize hyperparameters for the best model using Grid Search, Random Search, or Optuna.
# This step improves model performance by finding the optimal hyperparameter combination.

OPTIMIZATION_METHOD = 'grid'
best_model_type = best_model
base_model = models[best_model_type]
param_grids = {
    'Logistic Regression': {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear', 'lbfgs']
    },
    'Random Forest': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [5, 10, 15, 20, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    'Gradient Boosting': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__subsample': [0.8, 1.0]
    },
    'SVM': {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'classifier__kernel': ['rbf', 'poly', 'sigmoid']
    },
    'K-Nearest Neighbors': {
        'classifier__n_neighbors': [3, 5, 7, 9, 11],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
    }
}
if best_model_type in param_grids:
    param_grid = param_grids[best_model_type]
    
    print(f"Optimizing hyperparameters for {best_model_type}...")
    print(f"Method: {OPTIMIZATION_METHOD}")
    print(f"Parameter grid: {list(param_grid.keys())}")
    
    # Use TimeSeriesSplit for temporal cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    print(f"Using TimeSeriesSplit with {tscv.n_splits} splits (respects temporal order)")
    
    if OPTIMIZATION_METHOD == 'grid':
        grid_search = GridSearchCV(
            trained_pipelines[best_model_type],
            param_grid,
            cv=tscv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        optimized_pipeline = grid_search.best_estimator_
        
        print(f"\nGrid Search complete!")
        print(f"Best F1-Score (CV): {grid_search.best_score_:.4f}")
        print(f"Best parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
    
    elif OPTIMIZATION_METHOD == 'random':
        random_search = RandomizedSearchCV(
            trained_pipelines[best_model_type],
            param_grid,
            n_iter=50,
            cv=tscv,
            scoring='f1',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        random_search.fit(X_train, y_train)
        optimized_pipeline = random_search.best_estimator_
        
        print(f"\nRandom Search complete!")
        print(f"Best F1-Score (CV): {random_search.best_score_:.4f}")
        print(f"Best parameters:")
        for param, value in random_search.best_params_.items():
            print(f"  {param}: {value}")
    
    elif OPTIMIZATION_METHOD == 'optuna':
        try:
            
            def objective(trial):
                if best_model_type == 'Random Forest':
                    params = {
                        'classifier__n_estimators': trial.suggest_int('classifier__n_estimators', 50, 300),
                        'classifier__max_depth': trial.suggest_int('classifier__max_depth', 5, 30),
                        'classifier__min_samples_split': trial.suggest_int('classifier__min_samples_split', 2, 20),
                        'classifier__min_samples_leaf': trial.suggest_int('classifier__min_samples_leaf', 1, 10)
                    }
                elif best_model_type == 'Gradient Boosting':
                    params = {
                        'classifier__n_estimators': trial.suggest_int('classifier__n_estimators', 50, 300),
                        'classifier__max_depth': trial.suggest_int('classifier__max_depth', 3, 10),
                        'classifier__learning_rate': trial.suggest_float('classifier__learning_rate', 0.01, 0.3, log=True),
                        'classifier__subsample': trial.suggest_float('classifier__subsample', 0.6, 1.0)
                    }
                elif best_model_type == 'Logistic Regression':
                    params = {
                        'classifier__C': trial.suggest_float('classifier__C', 0.01, 100, log=True),
                        'classifier__penalty': trial.suggest_categorical('classifier__penalty', ['l1', 'l2']),
                        'classifier__solver': trial.suggest_categorical('classifier__solver', ['liblinear', 'lbfgs'])
                    }
                else:
                    return None
                
                pipeline = Pipeline([
                    ('preprocessing', preprocessing_pipeline),
                    ('classifier', type(base_model)(**{k.replace('classifier__', ''): v for k, v in params.items()}))
                ])
                
                # Use TimeSeriesSplit for temporal cross-validation
                tscv = TimeSeriesSplit(n_splits=5)
                scores = cross_val_score(pipeline, X_train, y_train, cv=tscv, scoring='f1', n_jobs=-1)
                return scores.mean()
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=50, show_progress_bar=True)
            
            best_params = {k.replace('classifier__', ''): v for k, v in study.best_params.items()}
            optimized_pipeline = Pipeline([
                ('preprocessing', preprocessing_pipeline),
                ('classifier', type(base_model)(**best_params))
            ])
            optimized_pipeline.fit(X_train, y_train)
            
            print(f"\nOptuna optimization complete!")
            print(f"Best F1-Score (CV): {study.best_value:.4f}")
            print(f"Best parameters:")
            for param, value in study.best_params.items():
                print(f"  {param}: {value}")
        
        except ImportError:
            print("⚠ Optuna not available. Falling back to Grid Search...")
            # Use TimeSeriesSplit for temporal cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            grid_search = GridSearchCV(
                trained_pipelines[best_model_type],
                param_grid,
                cv=tscv,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            optimized_pipeline = grid_search.best_estimator_
    
    y_pred_optimized = optimized_pipeline.predict(X_test)
    y_pred_proba_optimized = optimized_pipeline.predict_proba(X_test)[:, 1]
    
    optimized_results = {
        'accuracy': accuracy_score(y_test, y_pred_optimized),
        'precision': precision_score(y_test, y_pred_optimized),
        'recall': recall_score(y_test, y_pred_optimized),
        'f1': f1_score(y_test, y_pred_optimized),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_optimized)
    }
    
    print(f"\n{'='*100}")
    print(f"Comparison: Before vs After Optimization")
    print(f"{'='*100}")
    print(f"F1-Score:  {results[best_model_type]['f1']:.4f} → {optimized_results['f1']:.4f} ({optimized_results['f1'] - results[best_model_type]['f1']:+.4f})")
    print(f"ROC-AUC:   {results[best_model_type]['roc_auc']:.4f} → {optimized_results['roc_auc']:.4f} ({optimized_results['roc_auc'] - results[best_model_type]['roc_auc']:+.4f})")
    print(f"Precision: {results[best_model_type]['precision']:.4f} → {optimized_results['precision']:.4f} ({optimized_results['precision'] - results[best_model_type]['precision']:+.4f})")
    print(f"Recall:    {results[best_model_type]['recall']:.4f} → {optimized_results['recall']:.4f} ({optimized_results['recall'] - results[best_model_type]['recall']:+.4f})")
    
    best_pipeline = optimized_pipeline
    
else:
    print(f"No parameter grid defined for {best_model_type}")
    print("Skipping hyperparameter optimization.")

# Detailed Evaluation: Best Model
# Get detailed classification report and confusion matrix for the best model.

best_pipeline = trained_pipelines[best_model]
y_pred_best = results[best_model]['y_pred']
print(f"Detailed Classification Report for {best_model}:")
print("=" * 100)
print(classification_report(y_test, y_pred_best, target_names=['No Tsunami', 'Tsunami']))
print("\nConfusion Matrix:")
print("=" * 100)
cm = confusion_matrix(y_test, y_pred_best)
cm_df = pd.DataFrame(cm, 
                     index=['Actual: No Tsunami', 'Actual: Tsunami'],
                     columns=['Predicted: No Tsunami', 'Predicted: Tsunami'])
print(cm_df)
# Plot removed - see notebook for visualizations
# plt.figure(figsize=(8, 6))
# plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title(f'Confusion Matrix - {best_model}')
# plt.colorbar()
# tick_marks = np.arange(2)
# plt.xticks(tick_marks, ['No Tsunami', 'Tsunami'])
# plt.yticks(tick_marks, ['No Tsunami', 'Tsunami'])
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')
# thresh = cm.max() / 2.
# for i, j in np.ndindex(cm.shape):
#     plt.text(j, i, format(cm[i, j], 'd'),
#              horizontalalignment="center",
#              color="white" if cm[i, j] > thresh else "black")
# plt.tight_layout()
# plt.show()

# ROC and Precision-Recall Curves
# Visualize model performance using ROC and Precision-Recall curves.
# Plot removed - see notebook for visualizations

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# for model_name, metrics in results.items():
#     fpr, tpr, _ = roc_curve(y_test, metrics['y_pred_proba'])
#     plt.plot(fpr, tpr, label=f"{model_name} (AUC = {metrics['roc_auc']:.3f})", linewidth=2)
# plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curves')
# plt.legend(loc="lower right")
# plt.grid(alpha=0.3)
# plt.subplot(1, 2, 2)
# for model_name, metrics in results.items():
#     precision, recall, _ = precision_recall_curve(y_test, metrics['y_pred_proba'])
#     plt.plot(recall, precision, label=f"{model_name}", linewidth=2)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curves')
# plt.legend(loc="lower left")
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.show()

# Feature Importance (for Tree-based Models)
# Analyze feature importance for interpretable models.

tree_models = ['Random Forest', 'Gradient Boosting']
for model_name in tree_models:
    if model_name in trained_pipelines:
        pipeline = trained_pipelines[model_name]
        classifier = pipeline.named_steps['classifier']
        
        feature_names = pipeline.named_steps['preprocessing'].named_steps['feature_engineering'].feature_names_
        
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\n{model_name} - Top 10 Most Important Features:")
            print("=" * 100)
            print(feature_importance_df.head(10))
            
            # Plot removed - see notebook for visualizations
            # plt.figure(figsize=(10, 6))
            # top_features = feature_importance_df.head(15)
            # plt.barh(range(len(top_features)), top_features['importance'].values)
            # plt.yticks(range(len(top_features)), top_features['feature'].values)
            # plt.xlabel('Importance')
            # plt.title(f'Feature Importance - {model_name}')
            # plt.gca().invert_yaxis()
            # plt.tight_layout()
            # plt.show()

#======================================================================
# PART 6: MODEL PERSISTENCE AND DEPLOYMENT
#======================================================================

# Save Pipeline for Reuse
# Save the best pipeline for future use and deployment.

current_dir = Path().resolve()
if current_dir.name == "notebooks":
    project_root = current_dir.parent
elif (current_dir / "requirements.txt").exists():
    project_root = current_dir
else:
    project_root = current_dir.parent
output_dir = project_root / "outputs"
output_dir.mkdir(exist_ok=True)
best_pipeline_path = output_dir / "best_pipeline.joblib"
joblib.dump(best_pipeline, best_pipeline_path)
print(f"Best pipeline saved to: {best_pipeline_path}")
all_pipelines_path = output_dir / "all_pipelines.joblib"
joblib.dump(trained_pipelines, all_pipelines_path)
print(f"All pipelines saved to: {all_pipelines_path}")
results_path = output_dir / "model_results.csv"
results_df.to_csv(results_path)
print(f"Results saved to: {results_path}")
metadata = {
    'best_model': best_model,
    'feature_columns': FEATURE_COLS,
    'target': TARGET,
    'train_size': len(X_train),
    'test_size': len(X_test),
    'split_year': split_year
}
metadata_path = output_dir / "pipeline_metadata.joblib"
joblib.dump(metadata, metadata_path)
print(f"Metadata saved to: {metadata_path}")
print("\n" + "=" * 100)
print("Pipeline saved successfully!")
print(f"\nLocal files:")
print(f"  -> pipeline = joblib.load('{best_pipeline_path}')")
print(f"  -> predictions = pipeline.predict(new_data)")
print(f"\nMLflow tracking:")
print(f"  -> All models and runs are tracked in: {mlflow_dir}")
print(f"  -> View UI: mlflow ui --backend-store-uri {mlflow_dir}")
print(f"  -> Load from MLflow: mlflow.sklearn.load_model('runs:/<run_id>/model')")
print("=" * 100)

# Load and Use Saved Pipeline
# Example of how to load and use the saved pipeline for new predictions.

print("To use the saved pipeline:")
print("1. Load: pipeline = joblib.load('outputs/best_pipeline.joblib')")
print("2. Predict: predictions = pipeline.predict(new_data)")
print("3. Probabilities: probabilities = pipeline.predict_proba(new_data)")

#======================================================================
# PART 7: PREDICTIVE ANALYSIS FOR 2023
#======================================================================

# Generate Scenarios for 2023
# We create realistic earthquake scenarios for 2023 by sampling from the historical distribution of features. This approach made sure that our predictions are based on plausible combinations of earthquake characteristics that have been observed in the past. We generate multiple scenarios covering different magnitude ranges, depths, and geographic locations to comprehensively assess tsunami risk.

magnitude_range = np.linspace(df['magnitude'].min(), df['magnitude'].max(), 10)
depth_range = np.linspace(df['depth'].min(), df['depth'].max(), 8)
latitude_range = np.linspace(df['latitude'].min(), df['latitude'].max(), 10)
longitude_range = np.linspace(df['longitude'].min(), df['longitude'].max(), 10)
np.random.seed(42)
n_scenarios = 100
scenarios_2023 = []
for i in range(n_scenarios):
    magnitude = np.random.choice(magnitude_range)
    depth = np.random.choice(depth_range)
    latitude = np.random.choice(latitude_range)
    longitude = np.random.choice(longitude_range)
    
    cdi = np.random.choice(df['cdi'].dropna().values) if not df['cdi'].isna().all() else np.nan
    mmi = np.random.choice(df['mmi'].dropna().values) if not df['mmi'].isna().all() else np.nan
    sig = np.random.choice(df['sig'].dropna().values) if not df['sig'].isna().all() else np.nan
    nst = np.random.choice(df['nst'].dropna().values) if not df['nst'].isna().all() else np.nan
    dmin = np.random.choice(df['dmin'].dropna().values) if not df['dmin'].isna().all() else np.nan
    gap = np.random.choice(df['gap'].dropna().values) if not df['gap'].isna().all() else np.nan
    
    scenario = {
        'magnitude': magnitude,
        'depth': depth,
        'latitude': latitude,
        'longitude': longitude,
        'cdi': cdi,
        'mmi': mmi,
        'sig': sig,
        'nst': nst,
        'dmin': dmin,
        'gap': gap,
        'Year': 2023,
        'Month': np.random.randint(1, 13)
    }
    scenarios_2023.append(scenario)
scenarios_df = pd.DataFrame(scenarios_2023)
scenarios_df['abs_lat'] = scenarios_df['latitude'].abs()
scenarios_df['mag_depth_ratio'] = scenarios_df['magnitude'] / (scenarios_df['depth'] + 1.0)
for col in ['cdi', 'mmi', 'sig', 'nst', 'dmin', 'gap']:
    if scenarios_df[col].isna().any():
        scenarios_df[col].fillna(df[col].median(), inplace=True)
print(f"Generated {len(scenarios_df)} earthquake scenarios for 2023")
print(f"\nScenario statistics:")
print(scenarios_df[['magnitude', 'depth', 'latitude', 'longitude']].describe())

# Predict Tsunami Probabilities
# Using our best-trained model, we predict the probability of tsunami occurrence for each generated scenario. The model outputs probabilities between 0 and 1, where values closer to 1 indicate higher likelihood of tsunami occurrence. We then identify the top 10 scenarios with the highest tsunami probabilities, which represent the most concerning earthquake scenarios for 2023 based on our model's learned patterns.

X_scenarios = scenarios_df.copy()
for col in FEATURE_COLS:
    if col not in X_scenarios.columns:
        if col == '_mag_bin':
            bins = np.arange(0, 11, 1)
            labels = [f"{i}" for i in range(0, 10)]
            X_scenarios['_mag_bin'] = pd.cut(X_scenarios['magnitude'].clip(lower=0, upper=10), 
                                            bins=bins, labels=labels, include_lowest=True)
        elif col == 'country_encoded':
            X_scenarios['country_encoded'] = 0  # Default value
        else:
            X_scenarios[col] = df[col].median() if col in df.columns else 0
X_scenarios = X_scenarios[FEATURE_COLS].copy()
tsunami_probabilities = best_pipeline.predict_proba(X_scenarios)[:, 1]
tsunami_predictions = best_pipeline.predict(X_scenarios)
scenarios_df['tsunami_probability'] = tsunami_probabilities
scenarios_df['tsunami_prediction'] = tsunami_predictions
top_10_scenarios = scenarios_df.nlargest(10, 'tsunami_probability').copy()
print("Identifying countries for top 10 scenarios...")
coords_top10 = list(zip(top_10_scenarios['latitude'].astype(float), top_10_scenarios['longitude'].astype(float)))
results_top10 = rg.search(coords_top10)
ccodes_top10 = [r["cc"] for r in results_top10]
def cc_to_name(cc):
    try:
        return pycountry.countries.get(alpha_2=cc).name
    except Exception:
        return cc
countries_top10 = [cc_to_name(cc) for cc in ccodes_top10]
top_10_scenarios['country'] = countries_top10
print("=" * 100)
print("TOP 10 MOST PROBABLE TSUNAMI SCENARIOS FOR 2023")
print("=" * 100)
print(f"\nModel used: {best_model}")
print(f"Total scenarios analyzed: {len(scenarios_df)}")
print(f"Scenarios predicted as tsunami: {tsunami_predictions.sum()}")
print(f"\nTop 10 scenarios by tsunami probability:\n")
display_cols = ['magnitude', 'depth', 'latitude', 'longitude', 'country', 'tsunami_probability', 'tsunami_prediction']
top_10_display = top_10_scenarios[display_cols].copy()
top_10_display['tsunami_probability'] = top_10_display['tsunami_probability'].apply(lambda x: f"{x:.4f}")
top_10_display['magnitude'] = top_10_display['magnitude'].apply(lambda x: f"{x:.2f}")
top_10_display['depth'] = top_10_display['depth'].apply(lambda x: f"{x:.2f}")
top_10_display['latitude'] = top_10_display['latitude'].apply(lambda x: f"{x:.2f}")
top_10_display['longitude'] = top_10_display['longitude'].apply(lambda x: f"{x:.2f}")
print(top_10_display)

# Detailed Analysis of Top 10 Scenarios
# We a detailed breakdown of the top 10 most probable tsunami scenarios, showing all relevant features for each scenario. This detailed view helps understand what combinations of earthquake characteristics our model considers most dangerous for tsunami generation.

print("=" * 100)
print("DETAILED FEATURES OF TOP 10 TSUNAMI SCENARIOS")
print("=" * 100)
for idx, (i, row) in enumerate(top_10_scenarios.iterrows(), 1):
    print(f"\n{'='*100}")
    print(f"SCENARIO #{idx} - Tsunami Probability: {row['tsunami_probability']:.4f} ({row['tsunami_probability']*100:.2f}%)")
    print(f"{'='*100}")
    print(f"Magnitude:        {row['magnitude']:.2f}")
    print(f"Depth (km):       {row['depth']:.2f}")
    print(f"Location:         Latitude {row['latitude']:.2f}°, Longitude {row['longitude']:.2f}°")
    print(f"Country:          {row['country']}")
    print(f"Magnitude/Depth:  {row['mag_depth_ratio']:.4f}")
    print(f"Absolute Latitude: {row['abs_lat']:.2f}°")
    print(f"CDI:              {row['cdi']:.2f}" if not pd.isna(row['cdi']) else "CDI:              N/A")
    print(f"MMI:              {row['mmi']:.2f}" if not pd.isna(row['mmi']) else "MMI:              N/A")
    print(f"Sig:              {row['sig']:.2f}" if not pd.isna(row['sig']) else "Sig:              N/A")
    print(f"Month:            {int(row['Month'])}")
    print(f"Prediction:       {'TSUNAMI LIKELY' if row['tsunami_prediction'] == 1 else 'No Tsunami'}")
print(f"\n{'='*100}")
print("SUMMARY STATISTICS OF TOP 10 SCENARIOS")
print(f"{'='*100}")
print(f"\nAverage Magnitude:        {top_10_scenarios['magnitude'].mean():.2f}")
print(f"Average Depth:            {top_10_scenarios['depth'].mean():.2f} km")
print(f"Average Tsunami Probability: {top_10_scenarios['tsunami_probability'].mean():.4f} ({top_10_scenarios['tsunami_probability'].mean()*100:.2f}%)")
print(f"Minimum Probability:     {top_10_scenarios['tsunami_probability'].min():.4f} ({top_10_scenarios['tsunami_probability'].min()*100:.2f}%)")
print(f"Maximum Probability:     {top_10_scenarios['tsunami_probability'].max():.4f} ({top_10_scenarios['tsunami_probability'].max()*100:.2f}%)")

# Visualization of Top 10 Scenarios
# We visualize the top 10 scenarios on a map to show their geographic distribution, and create a bar chart showing their tsunami probabilities. This helped us identify geographic patterns and understand the relative risk levels of different scenarios.
# All plots removed - see notebook for visualizations

# fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# # Plot 1: Geographic distribution
# ax1 = axes[0]
# scatter = ax1.scatter(top_10_scenarios['longitude'], top_10_scenarios['latitude'], 
#                      c=top_10_scenarios['tsunami_probability'], 
#                      s=top_10_scenarios['magnitude']*50, 
#                      cmap='Reds', alpha=0.7, edgecolors='black', linewidth=1.5)
# ax1.set_xlabel('Longitude', fontsize=12)
# ax1.set_ylabel('Latitude', fontsize=12)
# ax1.set_title('Geographic Distribution of Top 10 Tsunami Scenarios (2023)', fontsize=14, fontweight='bold')
# ax1.grid(True, alpha=0.3)
# plt.colorbar(scatter, ax=ax1, label='Tsunami Probability')
# for idx, (i, row) in enumerate(top_10_scenarios.iterrows(), 1):
#     ax1.annotate(f'#{idx}', 
#                 (row['longitude'], row['latitude']),
#                 fontsize=10, fontweight='bold',
#                 ha='center', va='center',
#                 color='white' if row['tsunami_probability'] > 0.5 else 'black')
# # Plot 2: Probability bar chart
# ax2 = axes[1]
# colors = plt.cm.Reds(top_10_scenarios['tsunami_probability'].values / top_10_scenarios['tsunami_probability'].max())
# bars = ax2.barh(range(10, 0, -1), top_10_scenarios['tsunami_probability'].values, color=colors)
# ax2.set_yticks(range(10, 0, -1))
# ax2.set_yticklabels([f"Scenario #{i}" for i in range(1, 11)])
# ax2.set_xlabel('Tsunami Probability', fontsize=12)
# ax2.set_title('Tsunami Probabilities - Top 10 Scenarios', fontsize=14, fontweight='bold')
# ax2.set_xlim(0, 1)
# ax2.grid(True, alpha=0.3, axis='x')
# for i, (idx, row) in enumerate(top_10_scenarios.iterrows()):
#     ax2.text(row['tsunami_probability'] + 0.02, 10-i, 
#             f"{row['tsunami_probability']:.3f}",
#             va='center', fontweight='bold')
# plt.tight_layout()
# plt.show()
# fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# # Magnitude distribution
# axes[0, 0].hist(top_10_scenarios['magnitude'], bins=5, edgecolor='black', alpha=0.7)
# axes[0, 0].set_xlabel('Magnitude')
# axes[0, 0].set_ylabel('Frequency')
# axes[0, 0].set_title('Magnitude Distribution (Top 10)')
# axes[0, 0].grid(True, alpha=0.3)
# # Depth distribution
# axes[0, 1].hist(top_10_scenarios['depth'], bins=5, edgecolor='black', alpha=0.7, color='orange')
# axes[0, 1].set_xlabel('Depth (km)')
# axes[0, 1].set_ylabel('Frequency')
# axes[0, 1].set_title('Depth Distribution (Top 10)')
# axes[0, 1].grid(True, alpha=0.3)
# # Magnitude vs Depth scatter
# scatter2 = axes[1, 0].scatter(top_10_scenarios['magnitude'], top_10_scenarios['depth'],
#                               c=top_10_scenarios['tsunami_probability'],
#                               s=100, cmap='Reds', edgecolors='black', linewidth=1.5)
# axes[1, 0].set_xlabel('Magnitude')
# axes[1, 0].set_ylabel('Depth (km)')
# axes[1, 0].set_title('Magnitude vs Depth (Top 10)')
# axes[1, 0].grid(True, alpha=0.3)
# plt.colorbar(scatter2, ax=axes[1, 0], label='Tsunami Probability')
# # Probability distribution
# axes[1, 1].bar(range(1, 11), top_10_scenarios['tsunami_probability'].values, 
#                color=plt.cm.Reds(top_10_scenarios['tsunami_probability'].values))
# axes[1, 1].set_xlabel('Scenario Rank')
# axes[1, 1].set_ylabel('Tsunami Probability')
# axes[1, 1].set_title('Probability by Rank')
# axes[1, 1].set_xticks(range(1, 11))
# axes[1, 1].grid(True, alpha=0.3, axis='y')
# plt.tight_layout()
# plt.show()

# Save Predictions
# We save the top 10 scenarios and all predictions to a CSV file for further analysis and reference. This made it for easy sharing of results and integration with other systems or reports.

current_dir = Path().resolve()
if current_dir.name == "Codes (.ipynb & .py)":
    project_root = current_dir.parent
elif (current_dir / "requirements.txt").exists():
    project_root = current_dir
else:
    project_root = current_dir.parent
output_dir = project_root / "outputs"
output_dir.mkdir(exist_ok=True)
all_predictions_path = output_dir / "predictions_2023_all.csv"
scenarios_df.to_csv(all_predictions_path, index=False)
print(f"All {len(scenarios_df)} scenarios saved to: {all_predictions_path}")
top_10_path = output_dir / "predictions_2023_top10.csv"
top_10_scenarios.to_csv(top_10_path, index=False)
print(f"Top 10 scenarios saved to: {top_10_path}")
print(f"\n{'='*100}")
print("Predictive analysis complete!")
print(f"{'='*100}")
print(f"\nKey findings:")
print(f"  - Analyzed {len(scenarios_df)} potential earthquake scenarios for 2023")
print(f"  - Identified {tsunami_predictions.sum()} scenarios with tsunami risk")
print(f"  - Top 10 scenarios have probabilities ranging from {top_10_scenarios['tsunami_probability'].min():.4f} to {top_10_scenarios['tsunami_probability'].max():.4f}")
print(f"  - Average probability of top 10: {top_10_scenarios['tsunami_probability'].mean():.4f} ({top_10_scenarios['tsunami_probability'].mean()*100:.2f}%)")
print(f"\nFiles saved:")
print(f"  - {all_predictions_path}")
print(f"  - {top_10_path}")
