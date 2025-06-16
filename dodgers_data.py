import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


dodgers = pd.read_csv(r"dodgers-2022.csv")
dodgers.head()

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

dodgers_dummy = dodgers_dummy.replace({True: 1, False: 0})
dodgers_dummy = dodgers_dummy.drop(columns=["month", "day"])
dodgers_dummy.head(10)
