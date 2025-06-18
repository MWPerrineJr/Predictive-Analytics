


# This code imports the libraries for the analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# This code imports the dataset and displays the first five rows
dodgers = pd.read_csv(r"dodgers-2022.csv")
dodgers.head()
# This code drops the month and day columns
dodgers = dodgers.drop(columns=["month", "day", "day_of_week" ], inplace=True)
dodgers.head()
# This code runs the data through one hot encoding
dodgers_dummy = pd.get_dummies(dodgers,
        columns=[
        "opponent",
        "skies",
        "day_night",
        "cap",
        "shirt",
        "fireworks",
        "bobblehead"],
    drop_first=True)
# This code converts the true and false to 0,1
dodgers_dummy = dodgers_dummy.replace({True: 1, False: 0})

# This code displays the first ten rows of the transformed data
dodgers_dummy.head(10)
# This code creates the X variable for the ML model
X = dodgers_dummy.drop(("attend"), axis=1)
# This code displays the first five rows of the X Variable
X.head()
# This code creates the y variable for the ML model
y = dodgers_dummy["attend"]
# This code displays the first five rows of the y variable
y.head()
# This code splits the X and y variables into train ad test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=99)
# This code shows the dimension of the X and y variable
X.shape
y.shape
# This code will normalize the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
X_train_scaled.shape
# This code creates the linear model object
lm = LinearRegression()
# This code fits the linear model to the data
model = lm.fit(X_train, y_train)
# This code makes a prediction
y_pred = model.predict(X_test)
# evaluate the model
mse = mean_squared_error(y_test, y_pred)
mse
r2= r2_score(y_test, y_pred)
r2
lm1 = LinearRegression()
model1 = lm1.fit(X_train_scaled, y_train)
y_pred1 = model.predict(X_test_scaled)
mse1 = mean_squared_error(y_test, y_pred1)
r2_1 = r2_score(y_test, y_pred1)
mse
r2_1
r2
print('Feature Coefficients :', model.coef_)