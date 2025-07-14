from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import random
import uvicorn

app = FastAPI()

# Load data and scaler
data = joblib.load("models/data.pkl")
scaler = joblib.load("models/scaler.pkl")

# Define features
num_features = ['Age', 'Height', 'Weight', 'BMI']
full_features = ['Sex', 'Age', 'Height', 'Weight', 'Hypertension', 'Diabetes',
                 'BMI', 'Level', 'Fitness Goal', 'Fitness Type']

# Request Body Schema
class UserInput(BaseModel):
    Sex: int
    Age: float
    Height: float
    Weight: float
    Hypertension: int
    Diabetes: int
    BMI: float
    Level: int
    Fitness_Goal: int
    Fitness_Type: int

@app.post("/recommend/")
def recommend(user: UserInput):
    # Step 1: Convert input to dict & normalize
    input_dict = {
        'Sex': user.Sex,
        'Age': user.Age,
        'Height': user.Height,
        'Weight': user.Weight,
        'Hypertension': user.Hypertension,
        'Diabetes': user.Diabetes,
        'BMI': user.BMI,
        'Level': user.Level,
        'Fitness Goal': user.Fitness_Goal,
        'Fitness Type': user.Fitness_Type
    }

    user_df = pd.DataFrame([input_dict])
    user_df[num_features] = scaler.transform(user_df[num_features])
    user_input_scaled = user_df[full_features]

    # Step 2: Cosine similarity
    similarity_scores = cosine_similarity(data[full_features], user_input_scaled).flatten()
    top_indices = similarity_scores.argsort()[-5:][::-1]
    similar_users = data.iloc[top_indices]
    recommendation_1 = similar_users[['Exercises', 'Diet', 'Equipment']].mode().iloc[0]

    # Step 3: Generate 2 simulated variations
    recommendations = [recommendation_1]
    for _ in range(2):
        mod_input = input_dict.copy()
        mod_input['Age'] += random.randint(-5, 5)
        mod_input['Weight'] += random.uniform(-5, 5)
        mod_input['BMI'] += random.uniform(-1, 1)

        mod_df = pd.DataFrame([mod_input])
        mod_df[num_features] = scaler.transform(mod_df[num_features])
        mod_scaled = mod_df[full_features]

        mod_scores = cosine_similarity(data[full_features], mod_scaled).flatten()
        mod_indices = mod_scores.argsort()[-5:][::-1]
        mod_users = data.iloc[mod_indices]
        rec = mod_users[['Exercises', 'Diet', 'Equipment']].mode().iloc[0]

        if not any(
            rec['Exercises'] == r['Exercises'] and
            rec['Diet'] == r['Diet'] and
            rec['Equipment'] == r['Equipment']
            for r in recommendations
        ):
            recommendations.append(rec)

    # Step 4: Return JSON
    return {
        "recommendations": [
            {
                "Exercises": rec['Exercises'],
                "Equipment": rec['Equipment'],
                "Diet": rec['Diet']
            }
            for rec in recommendations
        ]
    }

# === Run the server on port 8080 ===
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
