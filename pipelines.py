from utils import fit_predict_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

rs = 1

def read(file):
  return pd.read_csv(f'./datasets/home-data/{file}.csv', index_col='Id')

if __name__ == "__main__":
  data = read('train')

  # drop N/A values on the SalePrice column (target)
  data.dropna(subset=['SalePrice'], axis='index', inplace=True)
  y = data['SalePrice']
  # remove the SalePrice column
  data.drop(['SalePrice'], axis='columns', inplace=True)

  # numerical columns
  numerical_cols = [
    col for col in data.columns
    if data[col].dtype in ['int64', 'float64']
  ]

  # categorical columns with less than 10 unique values
  categorical_cols = [
    col for col in data.columns
    if data[col].dtype == 'object' and data[col].nunique() < 10
  ]
  
  cols = numerical_cols + categorical_cols
 
  # split train & validation
  X_train_full, X_valid_full, y_train, y_valid = train_test_split(data, y, train_size=0.8, random_state=rs)

  X_train = X_train_full[cols]
  X_valid = X_valid_full[cols]

  # fillers
  numerical_transformer = SimpleImputer(strategy='mean')
  categorical_transformer = Pipeline(
    steps=[
      ('imputer', SimpleImputer(strategy='constant')),
      ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
  )

  pre_processor = ColumnTransformer(
    transformers=[
      ('numeric_values', numerical_transformer, numerical_cols),
      ('categorical_values', categorical_transformer, categorical_cols)
    ]
  )

  model = RandomForestRegressor(n_estimators=100, random_state=rs)

  pipeline = Pipeline(
    steps=[
      ('pre_process', pre_processor),
      ('model', model)
    ]
  )

  print(f'MAE {fit_predict_score(pipeline, X_train, X_valid, y_train, y_valid)}')
