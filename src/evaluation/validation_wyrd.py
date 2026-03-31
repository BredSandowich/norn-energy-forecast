#src/validation.py

import pandas as pd

#Using walk forward validation to update the model with new data continuously
def walk_forward_validation(df, features, target, model, horizon=1): 
    predictions, actuals, timestamps = [], [], []
    for i in range(len(df)-horizon):
        train = df.iloc[:i+horizon]
        test = df.iloc[i+horizon:i+horizon+1]
        if len(test)==0:
            break
        
        x_train = train[features]
        y_train = train[target]
        X_test = test[features]
        Y_test = test[target]
        
        #Fit and predict
        model.fit(x_train, y_train)
        prediction = model.predict(X_test)[0]
        
        predictions.append(prediction)
        actuals.append(Y_test.values[0])
        timestamps.append(test["Datetime"].values[0])
    return pd.DataFrame({"Datetime": timestamps, "Actual": actuals, "Prediction": predictions})
