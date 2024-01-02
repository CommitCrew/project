#IMPORTING REQUIRED LIBRARIES
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import joblib

#--------------------------------------------------------------- MODEL TRAINING -------------------------------------------------------

#IMPORTING PREPROCESSED NORMALIZED DATA INTO A DATAFRAME
dataframe = pd.read_csv('preprocesseddata.csv')

#SPLITTING THE DATA INTO TEST AND TRAIN SETS USING 80/20 RATIO
train_size = int(0.8 * len(dataframe))
train, test = dataframe[:train_size], dataframe[train_size:]

#DEFINING FEATURES (HOUR, MONTH, DAY, NUMERICMACHINE_ID, NUMERICSENSOR_ID) AND TARGET VARIABLE (NORMALIZED READING)
X_train, y_train = train.drop('NormalizedReading', axis=1), train['NormalizedReading']
X_test, y_test = test.drop('NormalizedReading', axis=1), test['NormalizedReading']

#--------------------------------------------------------------- RANDOM FOREST MODEL -------------------------------------------------------

# HYPERPARAMETER TUNING FOR RANDOM FOREST USING GridSearchCV
param_grid_randomforest = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10]
}
randomforest_model = RandomForestRegressor(random_state=42)
grid_search_randomforest = GridSearchCV(estimator=randomforest_model, param_grid=param_grid_randomforest, cv=2, n_jobs=2, verbose=2)
grid_search_randomforest.fit(X_train, y_train)

# FIND OUT THE BEST RANDOM FOREST MODEL
best_randomforest_model = grid_search_randomforest.best_estimator_

# TRANINING THE BEST RANDOM FOREST MODEL ON THE TRAIN SET
best_randomforest_model.fit(X_train, y_train)

#TAKING PREDICTION USING BEST RANDOM FOREST MODEL ON THE TEST SET 
y_pred_randomforest = best_randomforest_model.predict(X_test)

# EVALUATING THE BEST RANDOM FOREST MODEL ON THE BASIS OF MEAN SQUARE ERROR
mse_randomforest = mean_squared_error(y_test, y_pred_randomforest)
print(f'Mean Squared Error for Random Forest Model: {mse_randomforest}')

# PLOTTING ACTUAL AND PREDICTED VALUES FOR VISUAL CLARITY
# plt.figure(figsize=(12, 6))
# plt.plot(test.index, y_test, label='Actual')
# plt.plot(test.index, y_pred_rf, label='Predicted')
# plt.legend()
# plt.title('Random Forest Model Predictions for Time Series')
# plt.show()

#--------------------------------------------------------------- XGBOOST MODEL -------------------------------------------------------
# # HYPERPARAMETER TUNING FOR XGBOOST USING GridSearchCV
param_grid_xgboost = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10]
}

xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search_xgboost = GridSearchCV(estimator=xgboost_model, param_grid=param_grid_xgboost, cv=2, n_jobs=2, verbose=2)
grid_search_xgboost.fit(X_train, y_train)

# FIND OUT THE BEST XGBOOST MODEL
best_xgboost_model = grid_search_xgboost.best_estimator_

# TRANINING THE BEST XGBOOST ON THE TRAIN SET
best_xgboost_model.fit(X_train, y_train)

#TAKING PREDICTION USING BEST XGBOOST MODEL ON THE TEST SET 
y_pred_xgboost = best_xgboost_model.predict(X_test)

# EVALUATING THE BEST XGBOOST MODEL ON THE BASIS OF MEAN SQUARE ERROR
mse_xgboost = mean_squared_error(y_test, y_pred_xgboost)
print(f'Mean Squared Error for XGBoost Model: {mse_xgboost}')

# PLOTTING ACTUAL AND PREDICTED VALUES FOR VISUAL CLARITY
# plt.figure(figsize=(12, 6))
# plt.plot(test.index, y_test, label='Actual')
# plt.plot(test.index, y_pred_xgb, label='Predicted')
# plt.legend()
# plt.title('XGBoost Model Predictions for Time Series')
# plt.show()

#--------------------------------------------------------------- MODEL EVALUATION -------------------------------------------------------

# COMPARING THE RANDOM FOREST AND XGBOOST MODEL USING MLFLOW
with mlflow.start_run() as run:
    
    # LOGGING METRICS FOR BOTH MODELS
    mlflow.log_metrics({'mse_rf': mse_randomforest, 'mse_xgb': mse_xgboost})

    # PRINTING THE BEST MODEL NAME BY COMPARING ITS MSE
    if (mse_randomforest < mse_xgboost):
        print("BEST MODEL IS: RANDOM FOREST")
        
    else:
         print("BEST MODEL IS: XGBOOST")
         
    # SELECTING THE BEST MODEL BY COMPARING ITS MSE
    BestModel = best_randomforest_model if mse_randomforest < mse_xgboost else best_xgboost_model

    # LOGGING THE BEST SELECTED MODEL
    mlflow.sklearn.log_model(BestModel, "BestModel")
    
    # SAVING THE BEST SELECTED MODEL USING JOBLIB
    BestModelfilename = 'model/BestModel.joblib'
    joblib.dump(BestModel, BestModelfilename)

    # REGISTING THE BEST MODEL IN MLFLOW REGISTRY
    model_name = "BestModel"
    run_id = run.info.run_id
    mlflow.register_model(f"runs:/{run_id}/BestModel", model_name)
    
    # PRINTING THE RUN ID
    print("MLflow Run ID:", run.info.run_id)