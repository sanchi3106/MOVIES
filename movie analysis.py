
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
with zipfile.ZipFile(r"C:\Users\khanc\Downloads\archive.zip") as z:
    with z.open("IMDb Movies India.csv") as f:
        df=pd.read_csv(f,encoding='latin1')
   
    

print("First 5 rows:\n", df.head())
print("\nDataset Info:\n")
print(df.info())


if 'id' in df.columns:
    df.drop(['id'], axis=1, inplace=True)

# Step 4: Handle missing values
df.dropna(subset=['rating'], inplace=True)  
df.fillna('Unknown', inplace=True)          

features = ['genre', 'director', 'actors']  
target = 'rating'

X = df[features]
y = df[target]


categorical_features = ['genre', 'director', 'actors']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))


plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Movie Ratings")
plt.grid(True)
plt.show()
