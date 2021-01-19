import pandas as pd
from utils import fit_predict_score

if __name__ == "__main__":
  X_train = pd.read_csv('./datasets/home-data/train.csv', index_col='Id').select_dtypes(exclude=['object'])

  print(f'columns : {X_train.columns}')
  
  missing_values_per_column = X_train.isnull().sum()
  print(f'missing values per columns {missing_values_per_column}')

  columns_with_missing_values = [column for column in X_train.columns if X_train[column].isnull().sum() > 0]
  print(f'columns with missing values: {columns_with_missing_values}')

  columns_with_missing_values = X_train[columns_with_missing_values].isnull().sum()
  print(f'colunms with missing values only {columns_with_missing_values}')