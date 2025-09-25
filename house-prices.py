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
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRFRegressor
# ------------------

# normal font for matplotlib
rcParams['font.family'] = 'DejaVu Sans'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

# --------------------------> read the DataSet

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# --------------------------> clean and featur eng DataSet

# print(df_train)
# print(df_train.info())
# print(df_train.isnull().sum())



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

# ------------------> fillna 

for col in ordinal_features.keys():
    df_train[col] = df_train[col].fillna('NA')

for col in nominal_features:
    df_train[col] = df_train[col].fillna('Missing')

# df_train['LotFrontage'] = df_train['LotFrontage'].fillna(df_train['LotFrontage'].mean())

# ------------------> onehot-ordinal

# ordinal
for col, categories in ordinal_features.items():
    oe = OrdinalEncoder(categories=[categories])
    df_train[col] = oe.fit_transform(df_train[[col]])


# onehot
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
nominal_encoded = ohe.fit_transform(df_train[nominal_features])

nominal_df = pd.DataFrame(
    nominal_encoded,
    columns=ohe.get_feature_names_out(nominal_features),
    index=df_train.index
)
df_train = pd.concat([df_train.drop(nominal_features, axis=1), nominal_df], axis=1)

# ------------------> creat X,y

X = df_train.drop('SalePrice', axis=1)
y = df_train['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

# SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# ---------> Comparison between models regression # -> the Best = 

models = {
    'lin_reg':LinearRegression(),
    'random_F_reg':RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42),
    'XGB_reg':XGBRFRegressor(n_estimators=500, random_state=42),
    'Ridge':Ridge(alpha=1),
    }

results = []
for name, model in models.items():
    print(f"\n--------->{name}<---------\n")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"r2_score {name}: {r2_score(y_test, y_pred)}")
    print(f"RMSE {name}: {root_mean_squared_error(y_test, y_pred)}")
    r2_sc = r2_score(y_test, y_pred)
    RMSE = root_mean_squared_error(y_test, y_pred)
    results.append({
        'model':name,
        'r2_score':r2_sc,
        'RMSE':RMSE
        })
    
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('r2_score', ascending=False)

# ---------------> sgd, ElasticNet  model with scale
scale = StandardScaler()
X_train_scale = scale.fit_transform(X_train)
X_test_scale = scale.transform(X_test)
sgd_model = SGDRegressor(max_iter=5000,tol=1e-4, eta0=0.001, learning_rate='constant', random_state=42)
sgd_model.fit(X_train_scale, y_train)
y_pred_sgd = sgd_model.predict(X_test_scale)
r2_sgd = r2_score(y_test, y_pred_sgd)
RMSE_sgd = root_mean_squared_error(y_test, y_pred_sgd)

elasticNet_model = ElasticNet(alpha=1, l1_ratio=.5)
elasticNet_model.fit(X_train_scale, y_train)
y_pred_ElasticNet = elasticNet_model.predict(X_test_scale)
r2_ElasticNet = r2_score(y_test, y_pred_ElasticNet)
RMSE_ElasticNet = root_mean_squared_error(y_test, y_pred_ElasticNet)

print(f"r2_score sgd: {r2_sgd}")
print(f"RMSE sgd: {RMSE_sgd}")

print(f"r2_score ElasticNet: {r2_ElasticNet}")
print(f"RMSE RMSE_ElasticNet: {RMSE_ElasticNet}")

metrics = ['R2 Score', 'RMSE']
sgd_values = [r2_sgd, RMSE_sgd]
elastic_values = [r2_ElasticNet, RMSE_ElasticNet]

# -------- نمودار SGDRegressor --------
plt.figure(figsize=(6,4))
sns.barplot(x=metrics, y=sgd_values, palette='Blues_d', legend=False, dodge=False)
plt.title('SGDRegressor Metrics')
for i, v in enumerate(sgd_values):
    plt.text(i, v + max(sgd_values)*0.01, f"{v:.2f}", ha='center', fontweight='bold')
plt.ylim(0, max(sgd_values)*1.2)  # فاصله برای متن
plt.show()

# -------- نمودار ElasticNet --------
plt.figure(figsize=(6,4))
sns.barplot(x=metrics, y=elastic_values, palette='Reds_d', legend=False, dodge=False)
plt.title('ElasticNet Metrics')
for i, v in enumerate(elastic_values):
    plt.text(i, v + max(elastic_values)*0.01, f"{v:.2f}", ha='center', fontweight='bold')
plt.ylim(0, max(elastic_values)*1.2)
plt.show()


# ------------------> run-train model













