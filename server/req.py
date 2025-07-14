import requests

# API endpoint
url = "http://127.0.0.1:8080/recommend/"

# Input data (same structure as expected by FastAPI's Pydantic model)
payload = {
    "Sex": 1,
    "Age": 20,
    "Height": 1.70,
    "Weight": 55,
    "Hypertension": 0,
    "Diabetes": 0,
    "BMI": 18.9,
    "Level": 0,
    "Fitness_Goal": 1,
    "Fitness_Type": 1
}

# Make the POST request
response = requests.post(url, json=payload)

# Check if successful
if response.status_code == 200:
    print("✅ Recommendations received:")
    for i, rec in enumerate(response.json()["recommendations"], 1):
        print(f"\nRecommendation {i}:")
        print(f"  Exercises: {rec['Exercises']}")
        print(f"  Equipment: {rec['Equipment']}")
        print(f"  Diet: {rec['Diet']}")
else:
    print("❌ Failed:", response.status_code)
    print(response.text)
