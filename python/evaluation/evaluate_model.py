
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("../../data/processed/clean_dataset.csv")

X=data.drop("crop_yield",axis=1)
y=data["crop_yield"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=joblib.load("../../results/models/crop_yield_rf.pkl")
pred=model.predict(X_test)

print("MSE:",mean_squared_error(y_test,pred))
print("R2:",r2_score(y_test,pred))
