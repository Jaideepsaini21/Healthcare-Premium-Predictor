import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
import sqlite3
import os

def retrain_model():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ================== 🔹 LOAD ORIGINAL DATA ==================
    young_path = os.path.join(BASE_DIR, "datasets", "premiums_young_with_gr.xlsx")
    rest_path = os.path.join(BASE_DIR, "datasets", "premiums_rest.xlsx")

    df_young = pd.read_excel(young_path)
    df_rest = pd.read_excel(rest_path)

    # ================== 🔹 LOAD DB DATA ==================
    db_path = os.path.join(BASE_DIR, "data", "user_data.db")
    conn = sqlite3.connect(db_path)

    df_new = pd.read_sql("SELECT * FROM user_data", conn)

    # ❗ Check target column
    if "premium" not in df_new.columns:
        print("❌ No 'premium' column in DB → skipping retrain")
        return

    # ================== 🔹 SPLIT DB DATA ==================
    df_young_db = df_new[df_new["Age"] <= 25]
    df_rest_db = df_new[df_new["Age"] > 25]

    # ================== 🔹 COMBINE ==================
    df_young_final = pd.concat([df_young, df_young_db], ignore_index=True)
    df_rest_final = pd.concat([df_rest, df_rest_db], ignore_index=True)

    # ================== 🔹 TRAIN YOUNG MODEL ==================
    X_y = df_young_final.drop("premium", axis=1)
    y_y = df_young_final["premium"]

    model_young = LinearRegression()
    model_young.fit(X_y, y_y)

    # ================== 🔹 TRAIN REST MODEL ==================
    X_r = df_rest_final.drop("premium", axis=1)
    y_r = df_rest_final["premium"]

    model_rest = LinearRegression()
    model_rest.fit(X_r, y_r)

    # ================== 🔹 SAVE MODELS ==================
    joblib.dump(model_young, os.path.join(BASE_DIR, "artifacts", "model_young_updated.joblib"))
    joblib.dump(model_rest, os.path.join(BASE_DIR, "artifacts", "model_rest_updated.joblib"))

    print("✅ Models retrained successfully!")