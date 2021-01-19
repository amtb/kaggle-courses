import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
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
 
  X_train = data[numerical_cols]

  # fillers
  numerical_transformer = SimpleImputer(strategy='mean')

  pre_processor = ColumnTransformer(
    transformers=[
      ('numeric_values', numerical_transformer, numerical_cols)
    ]
  )

  model = RandomForestRegressor(n_estimators=100, random_state=rs)

  pipeline = Pipeline(
    steps=[
      ('pre_process', pre_processor),
      ('model', model)
    ]
  )

  scores = -1 * cross_val_score(pipeline, X_train, y, cv=5, scoring='neg_mean_absolute_error')

  print(f'MAE {scores.mean()}')

  def get_score(n_estimators):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=rs)
    pipeline = Pipeline(
      steps=[
        ('preprocess', SimpleImputer()),
        ('model', model)
      ]
    )

    scores = cross_val_score(pipeline, X_train, y, cv=3, scoring='neg_mean_absolute_error')
    return -1 * scores.mean()
  
  estimators = [50 * i for i in range(1, 9)]

  results = {}
  for n_estimators in estimators:
    score = get_score(n_estimators)
    results[n_estimators] = score
    print(f'n_estimators = {n_estimators} - score = {score}')

  plt.plot(list(results.keys()), list(results.values()))
  plt.show()