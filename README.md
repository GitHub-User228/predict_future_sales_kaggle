# Kaggle competition: Predict Future Sales
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![cuDF](https://img.shields.io/badge/cuDF-purple?style=for-the-badge&)
![cuML](https://img.shields.io/badge/cuML-8648bd?style=for-the-badge&)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-219ebc?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-000814?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-778da9?style=for-the-badge)
![Optuna](https://img.shields.io/badge/Optuna-778da9?style=for-the-badge)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![transformers](https://img.shields.io/badge/transformers-green?style=for-the-badge&)

Repository which contains code for the Kaggle competition: [Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales)

# Description

This repository covers the following ML lifecycle steps:
- EDA
    - dealing with duplicates
    - dealing with missing values
    - dealing with outliers (boxplots and IQR filtering)
    - time series analysis (trend, seasonalities, autocorrelation, periodogram, statistical tests)
- Baseline model
    - constant prediction
- Feature Engineering
    - data regrouping by date_block_num (month block)
    - custom features from item_price & item_cnt_day
    - encoding & clustering for 'name' features via BERT, TSNE, fuzzywuzzy and Agglomerative Clustering
    - lag features
    - target encoding for categorical features
    - percentage change of item_cnt_day-based feature
    - rolling, expanding window and exponentially weighted aggregation for time-based features
- Feature Selection
    - permutation importance
    - recursive feature elimination based on importance values
- Hyperparameter tuning
    - Optuna via TPESampler
    - Train-Validation split approach

The following ML models are considered, because they can deal with missing data:
- XGBoost
- LightGBM

Also, a stacking ensemble is considered on the top of considered models. 
It is trained by splitting data in a half, where first half is used to train the first layer models, while
the second half - the meta model

CUDF library is utilised to accelerate the EDA and feature engineering using NVIDIA GPU.

# Kaggle results

List of RMSE scores:

- Constant model - 7.25558
- Tuned XGBoost on all features - 0.95259
- Tuned LightGBM on all features - 0.94863
- Tuned LightGBM on important features - 1.06981
- Stacking model (not tuned) - 0.96608
- Stacking model (not tuned) - 0.96435
- Error-analysis based modifications for tuned LightGBM - 0.82745 ([reference](https://www.kaggle.com/code/abubakar624/first-place-solution-kaggle-predict-future-sales))

# TODO

- Tune stacking model
- Perform error analysis
- Generate more features (if sufficient hardware resources are available)
- Consider cross-validation instead of train-validation split (if sufficient hardware resources are available)
