# 🏠 House Price Prediction

This project predicts house prices using machine learning. The dataset
is cleaned, engineered, and transformed before modeling. A pipeline
integrates preprocessing, scaling, and regression, with XGBoost as the
main model. Hyperparameters are optimized with **RandomizedSearchCV**,
and performance is measured using **R²** and **RMSE**.

------------------------------------------------------------------------

## 📂 Project Workflow

1.  **Data Cleaning** -- Removing outliers and handling missing values.\
2.  **Feature Engineering** -- Creating new features (e.g., `TotalSF`,
    `AgeHouse`, `TotalBath`).\
3.  **Encoding** -- Applying **OrdinalEncoder** and **OneHotEncoder**
    for categorical features.\
4.  **Pipeline** -- Preprocessing, scaling, and model integration.\
5.  **Model Training** -- Training XGBoost with randomized
    hyperparameter search.\
6.  **Evaluation** -- Assessing model using R² Score and RMSE.

------------------------------------------------------------------------

## ⚙️ Requirements

Install dependencies via:

``` bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

------------------------------------------------------------------------

## ▶️ How to Run

1.  Clone the repo and place the dataset (`train.csv`) in the project
    folder.\
2.  Run the script:\

``` bash
python main.py
```

3.  The output will display **R² Score** and **RMSE**.

------------------------------------------------------------------------

## 📊 Results

-   **R² Score:** High accuracy in predicting house prices.\
-   **RMSE:** Low error, showing strong generalization.

