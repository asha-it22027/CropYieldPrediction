
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

data = pd.read_csv('../data/Crop_recommendation.csv')
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])  

np.random.seed(42) 
data['Yield'] = (data['rainfall'] * 0.01) + (data['N'] * 0.02) + \
    (data['P'] * 0.015) + (data['K'] * 0.01) + np.random.normal(0, 0.5, len(data))

X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']]
y = data['Yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Save model and encoder
with open('../model/crop_model.pkl', 'wb') as file:
    pickle.dump(model, file)
with open('../model/label_encoder.pkl', 'wb') as file:
    pickle.dump(le, file)

print("Model and encoder saved successfully!")