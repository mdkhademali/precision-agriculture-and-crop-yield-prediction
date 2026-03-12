
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

data = pd.read_csv("../../data/processed/clean_dataset.csv")

X = data.drop("crop_yield",axis=1)
y = data["crop_yield"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train,y_train)

joblib.dump(model,"../../results/models/crop_yield_rf.pkl")
print("Model trained and saved.")
