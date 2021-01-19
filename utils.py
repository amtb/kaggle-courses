from sklearn.metrics import mean_absolute_error

def fit_predict_score(model, X_train, X_val, y_train, y_val):
  '''
  Fits and returns the mean absolute error
  '''
  model.fit(X_train, y_train)
  return mean_absolute_error(y_val, model.predict(X_val))
