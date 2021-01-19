import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

random_state = 1

def get_mae(nb_leaves, train_X, val_X, train_y, val_y):
  model = DecisionTreeRegressor(max_leaf_nodes = nb_leaves, random_state = random_state)
  model.fit(train_X, train_y)
  return mean_absolute_error(val_y, model.predict(val_X))

if __name__ == "__main__":
  melbourne_data = pd.read_csv('./datasets/melb_data.csv')
  melbourne_data = melbourne_data.dropna(axis = 0)

  features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
  X = melbourne_data[features]
  y = melbourne_data.Price

  print(y)

  # model
  melbourne_model = DecisionTreeRegressor(random_state = random_state)
  melbourne_model.fit(X, y)

  print("Making predictions for the following 5 houses:")
  print(X.head())
  print("The predictions are")
  print(melbourne_model.predict(X.head()))

  predictions = melbourne_model.predict(X)
  # error
  print(f'mean absolute error on the same training data : { mean_absolute_error(y, predictions)}')

  train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=random_state)
  melbourne_model.fit(train_X, train_y)

  predictions = melbourne_model.predict(val_X)
  print(f'mean absolute error on split training data : { mean_absolute_error(val_y, predictions)}')

  # find best config for the number of leaves
  number_of_leaves = [2, 5, 10, 20, 50, 100, 200, 500]

  min_mae = None
  best_nb_of_leaves = None
  for number_of_leaf_nodes in number_of_leaves:
    mae = get_mae(number_of_leaf_nodes, train_X, val_X, train_y, val_y)
    if min_mae == None or min_mae < mae:
      min_mae = mae
      best_nb_of_leaves = number_of_leaf_nodes
  
  print(f'The best number of leaves: {best_nb_of_leaves} which has an mae of {min_mae}')

  # retrain the model on the whole set
  best_model = DecisionTreeRegressor(max_leaf_nodes = best_nb_of_leaves, random_state = random_state)
  best_model.fit(X, y)

  # check the mae 
  mae_best_dt = mean_absolute_error(y, best_model.predict(X))
  print(f'mae of the best model : {mae_best_dt}')

  # using a random forest
  rf_model = RandomForestRegressor(random_state=random_state)
  rf_model.fit(X, y)
  mae_rf = mean_absolute_error(y, rf_model.predict(X))
  print(f'mae of random forest model : {mae_rf}')

  print(f'Better by {100 * (1 - mae_rf / mae_best_dt)} %')


  
