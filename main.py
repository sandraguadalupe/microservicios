from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import uvicorn
import os

app = FastAPI()

# Modelo global
model = None

@app.get("/")
def home():
    return {"mensaje": "Microservicio de predicción de precios de casas activo"}

@app.post("/train")
def train_model():
    global model

    df = pd.read_csv("train.csv")

    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df_encoded.drop("SalePrice", axis=1)
    y = df_encoded["SalePrice"]

    X = X.select_dtypes(include=np.number).dropna(axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    r2 = model.score(X_test, y_test)

    return {
        "mensaje": "Modelo entrenado exitosamente",
        "r2_score": round(r2, 3)
    }

@app.post("/predict")
def predict_house(data: dict):
    global model

    if model is None:
        return {"error": "El modelo no está entrenado. Ejecuta /train primero."}

    df_input = pd.DataFrame([data])

    for col in model.feature_names_in_:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[model.feature_names_in_]

    pred = model.predict(df_input)[0]

    return {
        "prediccion_saleprice": round(float(pred), 2)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
