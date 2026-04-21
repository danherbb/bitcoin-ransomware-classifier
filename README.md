# Bitcoin Ransomware Classifier

## Overview

This project classifies Bitcoin addresses as ransomware-related or legitimate by modeling transaction-behavior features from the BitcoinHeist dataset.

The workflow is notebook-based and is designed to run well in Google Colab, with optional GPU acceleration for supported preprocessing and model-training steps.

## Dataset

This project uses the BitcoinHeist Ransomware Dataset from Kaggle:

[BitcoinHeist Ransomware Dataset](https://www.kaggle.com/datasets/sapere0/bitcoinheist-ransomware-dataset)

Download the dataset manually and place it in the project's `data/` folder before running the notebooks.

## Recommended Environment

Google Colab is the recommended environment for this project.

Use:

- `Runtime` -> `Change runtime type`
- `Hardware accelerator` -> `GPU`

GPU support is used where available for:

- cuML `StandardScaler`
- cuML `RandomForestClassifier`
- XGBoost with CUDA

The notebooks automatically fall back to CPU-backed sklearn behavior when GPU libraries are not available.

Important:

- cuML works in Colab GPU Linux environments with NVIDIA CUDA support
- cuML does not work on local macOS setups

## Clone the Repository in Colab

Run this in a Colab cell:

```bash
!git clone --branch dev https://github.com/danherbb/bitcoin-ransomware-classifier.git
%cd bitcoin-ransomware-classifier
```

## Install Dependencies

Install the project requirements:

```bash
!pip install -r requirements.txt
```

If you want GPU acceleration in Colab, install RAPIDS packages too:

```bash
!pip install cudf-cu12 cuml-cu12 --extra-index-url=https://pypi.nvidia.com
```

Then restart the Colab runtime before continuing.

## Google Drive Setup

Each notebook includes cells for:

- mounting Google Drive
- setting the working directory

The notebooks expect this working directory when running from Drive:

```python
os.chdir('/content/drive/MyDrive/bitcoin-ransomware-classifier/notebooks')
```

If you want to run the notebooks from Google Drive, place the repository at:

```text
/content/drive/MyDrive/bitcoin-ransomware-classifier
```

## Expected Project Structure

```text
bitcoin-ransomware-classifier/
├── data/
├── notebooks/
├── README.md
└── requirements.txt
```

## Notebook Order

Run the notebooks in this order:

1. `notebooks/01_data_exploration_preprocessing.ipynb`
2. `notebooks/02_feature_engineering_and_preparation.ipynb`
3. `notebooks/04_hyperparameter_tuning.ipynb`

Optional:

4. `notebooks/03_preliminary_modeling.ipynb`

Notebook 3 is helpful for quick baseline comparisons, but notebook 4 is the main tuning and final-evaluation workflow.

## Notebook Summary

### 1. Data Exploration and Preprocessing
`notebooks/01_data_exploration_preprocessing.ipynb`

This notebook:

- loads the dataset
- explores the feature distributions
- inspects class imbalance
- outlines the modeling plan

### 2. Feature Engineering and Preparation
`notebooks/02_feature_engineering_and_preparation.ipynb`

This notebook:

- creates train, validation, and test splits
- engineers log, ratio, interaction, and cyclical features
- scales the feature set
- saves both:
  - the original scaled training split
  - the SMOTE-balanced training split
- writes prepared CSV files to `data/`

### 3. Preliminary Modeling
`notebooks/03_preliminary_modeling.ipynb`

This notebook:

- trains quick tree-based baseline models
- compares validation precision, recall, F1, and balanced accuracy

### 4. Hyperparameter Tuning
`notebooks/04_hyperparameter_tuning.ipynb`

This notebook:

- loads the prepared splits from `data/`
- compares original versus SMOTE training data
- runs `RandomizedSearchCV`
- evaluates the best searched models on the test set
- tunes thresholds without retraining the fitted models
- saves summary CSV outputs back to `data/`

## Files Saved by the Pipeline

Notebook 2 saves prepared splits such as:

- `data/X_train.csv`
- `data/X_train_original.csv`
- `data/X_val.csv`
- `data/X_test.csv`
- `data/y_train.csv`
- `data/y_train_original.csv`
- `data/y_val.csv`
- `data/y_test.csv`

Notebook 4 saves model-selection outputs such as:

- `data/best_validation_models.csv`
- `data/hyperparameter_search_results.csv`
- `data/test_results_tuned_models.csv`
- `data/threshold_search_results.csv`
- `data/test_results_threshold_tuned_models.csv`

## Practical Notes

- Restart the runtime after installing RAPIDS packages.
- Re-run notebook 2 before notebook 4 whenever you change feature engineering.
- Compare `Original` versus `SMOTE` results in notebook 4 to check whether SMOTE is improving recall at the cost of overfitting.
- Use the threshold-tuned test results as the strongest final comparison table for F1-driven model selection.

## Quick Start

If you want the shortest Colab setup path:

```bash
!git clone --branch dev https://github.com/danherbb/bitcoin-ransomware-classifier.git
%cd bitcoin-ransomware-classifier
!pip install -r requirements.txt
!pip install cudf-cu12 cuml-cu12 --extra-index-url=https://pypi.nvidia.com
```

Then:

1. restart the runtime
2. place the dataset in `data/`
3. run notebooks `1 -> 2 -> 4`
