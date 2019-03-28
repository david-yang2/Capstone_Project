import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

def rf_score(df):
    X = df[['duration_sec', 'start_station_id', 'hour', 'day_of_week']]
    y = df.end_station_id


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)   
    mean_score = cross_val_score(clf,X_test, y_test ).mean()
    return mean_score