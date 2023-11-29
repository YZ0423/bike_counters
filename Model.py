from problem import get_train_data, get_test_data, get_cv
from estimator import get_estimator, _encode_dates
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

X, y = get_train_data()
X_test = get_test_data()

pipeline = get_estimator() # Ridge Regression
y_preds = []

# Iterate over the generator to get train and validation indices for each fold
for train_idx, val_idx in get_cv(X, y, random_state=1):
    # Split your data into training and testing sets using the indices
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Fit the model on the training set
    pipeline.fit(X_train, y_train)

    # Evaluate the model on the testing set and store the score
    y_pred = pipeline.predict(X_test)
    y_preds.append(y_pred)

# Output the performance metric (e.g., mean squared error)
print(y_preds)

y_pred = np.mean(y_preds, axis=0)
results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
results.to_csv("submission.csv", index=False)
