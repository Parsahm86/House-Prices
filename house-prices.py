# add package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import randint, uniform

from matplotlib import rcParams

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, SGDRegressor,Ridge, ElasticNet
from sklearn.metrics import r2_score, root_mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

from xgboost import XGBRFRegressor, XGBRegressor
# ---------------------

# normal font for matplotlib
rcParams['font.family'] = 'DejaVu Sans'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

# --------------------------> read the DataSet

df_train = pd.read_csv('train.csv')

# --------------------------> clean and featur eng DataSet

# print(df_train)
# print(df_train.info())
# print(df_train.isnull().sum())

lower = df_train['SalePrice'].quantile(0.01)
upper = df_train['SalePrice'].quantile(0.99)
df_train = df_train[(df_train['SalePrice'] >= lower) & (df_train['SalePrice'] <= upper)]

# ------------------> Feature eng

df_train['TotalSF'] = df_train['TotalBsmtSF'] + df_train['1stFlrSF'] + df_train['2ndFlrSF']
df_train['AgeHouse'] = 2025 - df_train['YearBuilt']
df_train['RemodAge'] = 2025 - df_train['YearRemodAdd']
df_train['TotalBath'] = df_train['FullBath'] + 0.5 * df_train['HalfBath'] + df_train['BsmtFullBath'] + 0.5 * df_train['BsmtHalfBath']
df_train['PorchSF'] = df_train['OpenPorchSF'] + df_train['EnclosedPorch'] + df_train['3SsnPorch'] + df_train['ScreenPorch']

# ------------------> ordinal features

ordinal_features = {
    'ExterQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'ExterCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtQual': ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtCond': ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'HeatingQC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'KitchenQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'FireplaceQu': ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'GarageFinish': ['NA', 'Unf', 'RFn', 'Fin'],
    'GarageQual': ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'GarageCond': ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'PoolQC': ['NA', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtExposure': ['NA', 'No', 'Mn', 'Av', 'Gd'],
    'BsmtFinType1': ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'BsmtFinType2': ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'Functional': ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
    'CentralAir': ['N', 'Y'],
    'PavedDrive': ['N', 'P', 'Y'],
    'Fence': ['NA', 'MnW', 'GdWo', 'GdPrv', 'MnPrv','MnWw']
}

# ------------------> nominal features

nominal_features = [
    'MSZoning','Street','Alley','LotShape','LandContour','Utilities',
    'LotConfig','LandSlope','Neighborhood','Condition1','Condition2',
    'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st',
    'Exterior2nd','MasVnrType','Foundation','GarageType','MiscFeature',
    'Heating', 'Electrical', 'SaleType', 'SaleCondition'
]

# ------------------> creat X,y

X = df_train.drop('SalePrice', axis=1)
y = np.log1p(df_train['SalePrice'])

# ------------------> ColumnTransformer, Pipeline

ordinal_features_list = list(ordinal_features.keys())

preprocessor = ColumnTransformer(
    transformers=[
        ('ord', Pipeline([
            ('imputer', SimpleImputer(strategy="most_frequent")),
            ('encoder', OrdinalEncoder(categories=list(ordinal_features.values())))
        ]), ordinal_features_list),
        
        ('nom', Pipeline([
            ('imputer', SimpleImputer(strategy="most_frequent")),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), nominal_features),
        
        ('num', SimpleImputer(strategy='mean'),
         X.select_dtypes(include=["int64","float64"]).columns)
    ]
)

XGB_reg = XGBRegressor(n_estimators=100, random_state=42)

def make_pipe(model):
    return Pipeline(steps=[
        ('preprocessor', preprocessor ),
        ('scaler', StandardScaler(with_mean=False)),
        ('regressor', model)
        ])
    

# ------------------> train-test model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

XGB_reg__pipe = make_pipe(XGB_reg)

# ------------> random search for the Best hyperParm

param_dist = {
    "regressor__n_estimators": randint(50, 200),
    "regressor__max_depth": randint(3, 10),
    "regressor__learning_rate": uniform(0.05, 0.1),
    "regressor__subsample": uniform(0.7, 0.3),
    "regressor__colsample_bytree": uniform(0.7, 0.3),
    "regressor__reg_alpha": uniform(0, 1),
    "regressor__reg_lambda": uniform(0, 1)
}

scorer = make_scorer(r2_score)

random_search = RandomizedSearchCV(
    estimator=XGB_reg__pipe,
    param_distributions=param_dist,
    n_iter=12,   # چند ترکیب تست کن
    scoring=scorer,
    cv=3,
    verbose=0,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
XGB_pipe_Optimal = random_search.best_estimator_

# ---------------- Predictions ----------------

y_pred = XGB_pipe_Optimal.predict(X_test)
y_test_exp = np.expm1(y_test)
y_pred_exp = np.expm1(y_pred)

r2_sc = r2_score(y_test_exp, y_pred_exp)
RMSE = root_mean_squared_error(y_test_exp, y_pred_exp)

print(f"R2 Score: {r2_sc}\n")
print(f"RMSE: {RMSE}\n")



# # نمودار 20 ستون اول
# plt.figure(figsize=(10,6))
# feat_importances.head(20).plot(kind='barh')
# plt.gca().invert_yaxis()
# plt.title('Top 20 Feature Importances')
# plt.show()

















