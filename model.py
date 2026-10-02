import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load your merged data
df = pd.read_csv("merged_annual.csv")

X = df[["growing_season_rainfall_mm", "growing_season_solar_kwh_m2"]]
y = df["wheat_yield_kg_ha"]

model = LinearRegression().fit(X, y)

joblib.dump(model, "wheat_model.pkl")
print(f"Model trained on yield. R² = {model.score(X, y):.4f}")
print(f"Intercept: {model.intercept_:,.1f} kg/ha")
print(f"  rainfall coef: {model.coef_[0]:,.3f} kg/ha per mm")
print(f"  solar coef:    {model.coef_[1]:,.1f} kg/ha per kWh/m²/day")