# This code imports the libraries for the analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

# This code imports the dataset and displays the first five rows
dodgers = pd.read_csv(r"dodgers-2022.csv")
dodgers.head()

# This code runs the data through one hot encoding
dodgers_dummy = pd.get_dummies(
    dodgers,
    columns=[
        "day_of_week",
        "opponent",
        "skies",
        "day_night",
        "cap",
        "shirt",
        "fireworks",
        "bobblehead",
    ],
    drop_first=True,
)
# This code converts the true and false to 0,1
dodgers_dummy = dodgers_dummy.replace({True: 1, False: 0})
# This code drops the month and day columns
dodgers_dummy = dodgers_dummy.drop(columns=["month", "day"])
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
    X, y, test_size=0.20, random_state=99
)
# This code shows the dimension of the X and y variable
X.shape
y.shape
