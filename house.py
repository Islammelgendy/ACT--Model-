import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('house.csv')
df = df[['id', 'bedrooms', 'sqft_basement', 'view', 'bathrooms',
         'sqft_living15', 'sqft_above', 'grade', 'sqft_living', 'price']]
X = df.iloc[:, 1:-1]

scaler = StandardScaler()
X = scaler.fit_transform(df)

y = df['price']

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=.3, random_state=42)

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

print(np.sqrt(mean_squared_error(y_pred, y_test)))

pickle.dump(lr, open('fish.pkl', 'wb'))

print('model_saved')
