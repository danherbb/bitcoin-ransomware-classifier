# Bitcoin Ransomware Classifier

## Overview

This project builds a machine learning classifier to identify whether a Bitcoin address is associated with ransomware activity or legitimate use, using behavioral features from the BitcoinHeist dataset.

The project focuses on evaluating how different modeling decisions impact performance in a highly imbalanced classification problem. In particular, we explore:

```text
- Tree-based models (Decision Tree, Random Forest, Gradient Boosting)
- Class imbalance handling (SMOTE vs original data)
- Dimensionality reduction (PCA)
- Model tuning and threshold optimization
- Advanced modeling using XGBoost
```

Our final model uses XGBoost trained on the original dataset with threshold tuning, achieving the best balance between precision and recall.


## Dataset

This project uses the BitcoinHeist Ransomware Dataset from Kaggle:

[BitcoinHeist Ransomware Dataset](https://www.kaggle.com/datasets/sapere0/bitcoinheist-ransomware-dataset)

- ~2.9 million records  
- Features derived from transaction graphs (provided in tabular form)  
- Highly imbalanced target (ransomware is rare)
  
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

## Project Structure

```text
bitcoin-ransomware-classifier/
├── data/
├── notebooks/
│ ├── 01_data_exploration_preprocessing.ipynb
│ ├── 02_feature_engineering_and_preparation.ipynb
│ ├── 03_preliminary_modeling.ipynb
│ ├── 04_hyperparameter_tuning.ipynb
│ └── 05_targeted_xgb_tuning.ipynb
├── README.md
└── requirements.txt
```

## Workflow Overview

Data → Preprocessing → Feature Engineering → Splitting → Modeling → Evaluation → Refinement


## Notebook Order

Run the notebooks in this order:

1. `notebooks/01_data_exploration_preprocessing.ipynb`  
2. `notebooks/02_feature_engineering_and_preparation.ipynb`  
3. `notebooks/03_preliminary_modeling.ipynb`  
4. `notebooks/04_hyperparameter_tuning.ipynb`  
5. `notebooks/05_targeted_xgb_tuning.ipynb`  

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

### 5. Targeted XGBoost Tuning 
`notebooks/05_targeted_xgb_tuning.ipynb`

This notebook:

- Introduces XGBoost as an advanced model  
- Compares performance against previous models  
- Analyzes overfitting and generalization  
- Applies threshold tuning for final model selection  
- Produces final evaluation metrics  

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

## Final Results

The best-performing model was XGBoost trained on the original (imbalanced) dataset with hyperparameter tuning and threshold optimization.

Final performance:

- Precision ≈ 0.45  
- Recall ≈ 0.45  
- F1 Score ≈ 0.45  

These results reflect a balanced tradeoff between detecting ransomware activity and minimizing false positives.


## Key Findings

- PCA reduced performance by removing important behavioral signals  
- SMOTE improved recall but introduced overfitting  
- Tree-based models tended to overfit on balanced data  
- XGBoost trained on the original dataset provided the best generalization  
- Threshold tuning improved the balance between precision and recall  
