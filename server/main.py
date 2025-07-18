# -*- coding: utf-8 -*-
"""workout-diet-recommendation-system-content-b (1).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1YbJefA8A4LouIrV8Ux8JTxWfmy3n9kMl

# Workout Recommendation System using Content - Based Filtering
___
By : Hossam Farhoud

The Workout Recommendation System is designed to provide personalized fitness and diet plans based on user-specific details using a content-based filtering approach. This system uses user attributes like age, weight, BMI, and fitness goals to recommend exercises, diets, and equipment that closely align with their individual needs.

Key Features:
1-Content-Based Filtering: Matches the user's attributes with a dataset of existing users to provide tailored recommendations.

2-Cosine Similarity: Calculates the similarity between the new user and existing users to identify the most relevant fitness plans.

3-Personalized Recommendations: Suggests workout and diet plans based on the most similar users.

4-Feedback and Evaluation: Collects user feedback on the recommendations and calculates the Mean Reciprocal Rank (MRR) to measure the relevance and effectiveness of the system.
Workflow:
1-Data Loading and Cleaning: The dataset is prepared by encoding categorical features and normalizing numerical attributes for consistency.

2-Similarity Calculation: User details are compared with the dataset to identify similar users.

3-Recommendations: Provides personalized fitness plans and generates alternative suggestions by slightly varying user inputs.

4-Feedback and Evaluation: Collects user feedback and calculates MRR to assess the relevance of 4-4-recommendations.

This system is ideal for fitness enthusiasts looking for tailored workout and diet plans to achieve their health goals effectively.

## 1. DATA LOADING & CLEANING
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_excel("gym recommendation.xlsx")
data.columns

data.drop(columns=['ID'], inplace = True)

data.head()

data.shape

data.columns

# Import LabelEncoder from sklearn
from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
label_enc = LabelEncoder()

# Apply Label Encoding to the specified categorical columns
for col in ['Sex', 'Hypertension', 'Diabetes', 'Level', 'Fitness Goal', 'Fitness Type']:
    data[col] = label_enc.fit_transform(data[col])

data.head()

"""## 3. NORMALIZATION"""

scaler=StandardScaler()
data[['Age', 'Height', 'Weight', 'BMI',]]= scaler.fit_transform(data[['Age', 'Height', 'Weight', 'BMI',]])

data.head()

"""## 4. RECOMMENDATION, FEEDBACK & EVALUATION

`Cosine similarity` is a metric used to measure how similar two vectors are
  - Once the cosine similarity scores are calculated between the new user's profile and existing users, the system identifies the top similar users (those with the highest similarity scores).
  - The system then generates recommendations based on the most common exercises and diets of these similar users, thereby tailoring suggestions to the new user's profile.
"""

import random  # Importing the random module for generating random variations

def get_recommendation(top_n=3):
    # Start the recommendation process with user inputs
    print("Please enter your details for a personalized workout and diet recommendation.")

    # Collect user inputs for their details and health information
    user_input = {
        'Sex': int(input("Enter Sex (Male : 1/Female : 0): ")),  # User's gender
        'Age': float(input("Enter Age: ")),  # User's age
        'Height': float(input("Enter Height in meters (e.g., 1.75): ")),  # User's height
        'Weight': float(input("Enter Weight in kg: ")),  # User's weight
        'Hypertension': int(input("Do you have Hypertension (Yes : 1/No : 0): ")),  # Hypertension status
        'Diabetes': int(input("Do you have Diabetes (Yes : 1/No : 0): ")),  # Diabetes status
        'BMI': float(input("Enter BMI: ")),  # User's BMI
        'Level': int(input("Enter Level (Underweight : 3, Normal : 0, Overweight : 2, Obese : 1): ")),  # Fitness level
        'Fitness Goal': int(input("Enter Fitness Goal (Weight Gain : 0, Weight Loss : 1): ")),  # Fitness goal
        'Fitness Type': int(input("Enter Fitness Type (Muscular Fitness : 1, Cardio Fitness : 0): "))  # Fitness type
    }

    # Normalize numerical features for consistency
    num_features = ['Age', 'Height', 'Weight', 'BMI']  # Columns to normalize
    user_df = pd.DataFrame([user_input], columns=num_features)  # Create a DataFrame for user input
    user_df[num_features] = scaler.transform(user_df[num_features])  # Normalize numerical features
    user_input.update(user_df.iloc[0].to_dict())  # Update the normalized values in the user input dictionary
    user_df = pd.DataFrame([user_input])  # Create a new DataFrame with updated user input

    # Calculate similarity scores between user input and dataset
    user_features = data[['Sex', 'Age', 'Height', 'Weight', 'Hypertension', 'Diabetes', 'BMI', 'Level', 'Fitness Goal', 'Fitness Type']]  # Features for similarity
    similarity_scores = cosine_similarity(user_features, user_df).flatten()  # Calculate similarity scores

    # Retrieve the top 5 similar users
    similar_user_indices = similarity_scores.argsort()[-5:][::-1]  # Get indices of the most similar users
    similar_users = data.iloc[similar_user_indices]  # Extract similar user data
    recommendation_1 = similar_users[['Exercises', 'Diet', 'Equipment']].mode().iloc[0]  # Get the most common recommendation

    # Generate two additional recommendations by slightly varying the input
    simulated_recommendations = []

    for _ in range(2):  # Loop for generating two additional recommendations
        modified_input = user_input.copy()  # Create a copy of the original input

        # Randomly adjust Age, Weight, and BMI with larger variations
        modified_input['Age'] += random.randint(-5, 5)  # Randomly vary age
        modified_input['Weight'] += random.uniform(-5, 5)  # Randomly vary weight
        modified_input['BMI'] += random.uniform(-1, 1)  # Randomly vary BMI

        # Normalize the modified input values
        modified_user_df = pd.DataFrame([modified_input], columns=num_features)  # Create a DataFrame
        modified_user_df[num_features] = scaler.transform(modified_user_df[num_features])  # Normalize numerical features
        modified_input.update(modified_user_df.iloc[0].to_dict())  # Update normalized values in modified input

        # Calculate similarity scores for the modified input
        modified_similarity_scores = cosine_similarity(user_features, pd.DataFrame([modified_input])).flatten()  # Calculate similarity
        modified_similar_user_indices = modified_similarity_scores.argsort()[-5:][::-1]  # Get indices of similar users
        modified_similar_users = data.iloc[modified_similar_user_indices]  # Extract similar user data
        recommendation = modified_similar_users[['Exercises', 'Diet', 'Equipment']].mode().iloc[0]  # Get the most common recommendation

        # Ensure the new recommendation is unique
        if not any(rec['Exercises'] == recommendation['Exercises'] and rec['Diet'] == recommendation['Diet'] and rec['Equipment'] == recommendation['Equipment'] for rec in simulated_recommendations):
            simulated_recommendations.append(recommendation)  # Add unique recommendation

    # Display recommendations to the user
    print("\nRecommended Workout and Diet Plans based on your input:")
    print("\nRecommendation 1 (Exact match):")
    print("EXERCISES:", recommendation_1['Exercises'])  # Display exercises
    print("EQUIPMENTS:", recommendation_1['Equipment'])  # Display equipment
    print("DIET:", recommendation_1['Diet'])  # Display diet

    # Display additional recommendations
    for idx, rec in enumerate(simulated_recommendations, start=2):
        print(f"\nRecommendation {idx} (Slight variation):")
        print("EXERCISES:", rec['Exercises'])  # Display exercises
        print("EQUIPMENTS:", rec['Equipment'])  # Display equipment
        print("DIET:", rec['Diet'])  # Display diet

    # Collect feedback from the user
    feedback_matrix = []
    for i in range(len(simulated_recommendations) + 1):  # Loop through all recommendations
        feedback = int(input(f"Was recommendation {i+1} relevant? (Yes: 1, No: 0): "))  # Get feedback
        feedback_matrix.append(feedback)  # Store feedback

    # Calculate Mean Reciprocal Rank (MRR)
    relevant_indices = [i + 1 for i, feedback in enumerate(feedback_matrix) if feedback == 1]  # Get ranks of relevant recommendations
    if relevant_indices:  # Check if there are any relevant recommendations
        mrr = np.mean([1 / rank for rank in relevant_indices])  # Calculate MRR
    else:
        mrr = 0.0  # Set MRR to 0 if no recommendations were relevant

    # Display MRR score
    print(f"\nMean Reciprocal Rank (MRR): {mrr:.2f}")

    return [recommendation_1] + simulated_recommendations  # Return all recommendations

# Example:
# recommendations = get_recommendation(top_n=3)  edit and give

""" - Mean Reciprocal Rank (MRR)
The MRR is computed based on the feedback:

It measures the relevance of recommendations.
"""

import random  # For generating random variations

def get_recommendation(top_n=3):
    print("Please enter your details for a personalized workout and diet recommendation.")

    # 1. User input
    user_input = {
        'Sex': int(input("Enter Sex (Male : 1 / Female : 0): ")),
        'Age': float(input("Enter Age: ")),
        'Height': float(input("Enter Height in meters (e.g., 1.75): ")),
        'Weight': float(input("Enter Weight in kg: ")),
        'Hypertension': int(input("Do you have Hypertension (Yes : 1 / No : 0): ")),
        'Diabetes': int(input("Do you have Diabetes (Yes : 1 / No : 0): ")),
        'BMI': float(input("Enter BMI: ")),
        'Level': int(input("Enter Level (Underweight : 3, Normal : 0, Overweight : 2, Obese : 1): ")),
        'Fitness Goal': int(input("Enter Fitness Goal (Weight Gain : 0, Weight Loss : 1): ")),
        'Fitness Type': int(input("Enter Fitness Type (Muscular Fitness : 1, Cardio Fitness : 0): "))
    }

    # 2. Normalize numeric fields
    num_features = ['Age', 'Height', 'Weight', 'BMI']
    full_features = ['Sex', 'Age', 'Height', 'Weight', 'Hypertension', 'Diabetes',
                     'BMI', 'Level', 'Fitness Goal', 'Fitness Type']

    user_df = pd.DataFrame([user_input])
    user_df[num_features] = scaler.transform(user_df[num_features])
    user_input.update(user_df.iloc[0][num_features].to_dict())
    user_df = pd.DataFrame([user_input])[full_features]

    # 3. Compute cosine similarity
    user_features = data[full_features]
    similarity_scores = cosine_similarity(user_features, user_df).flatten()
    similar_user_indices = similarity_scores.argsort()[-5:][::-1]
    similar_users = data.iloc[similar_user_indices]
    recommendation_1 = similar_users[['Exercises', 'Diet', 'Equipment']].mode().iloc[0]

    # 4. Generate 2 variations
    simulated_recommendations = []

    for _ in range(2):
        modified_input = user_input.copy()
        modified_input['Age'] += random.randint(-5, 5)
        modified_input['Weight'] += random.uniform(-5, 5)
        modified_input['BMI'] += random.uniform(-1, 1)

        mod_df = pd.DataFrame([modified_input])
        mod_df[num_features] = scaler.transform(mod_df[num_features])
        modified_input.update(mod_df.iloc[0][num_features].to_dict())
        mod_df = pd.DataFrame([modified_input])[full_features]

        mod_scores = cosine_similarity(user_features, mod_df).flatten()
        mod_indices = mod_scores.argsort()[-5:][::-1]
        mod_users = data.iloc[mod_indices]
        recommendation = mod_users[['Exercises', 'Diet', 'Equipment']].mode().iloc[0]

        # Ensure uniqueness
        if not any(
            rec['Exercises'] == recommendation['Exercises'] and
            rec['Diet'] == recommendation['Diet'] and
            rec['Equipment'] == recommendation['Equipment']
            for rec in simulated_recommendations
        ):
            simulated_recommendations.append(recommendation)

    # 5. Display recommendations
    all_recs = [recommendation_1] + simulated_recommendations

    print("\n🔹 Recommended Workout and Diet Plans Based on Your Input 🔹")
    for i, rec in enumerate(all_recs, 1):
        print(f"\nRecommendation {i}:")
        print("EXERCISES:", rec['Exercises'])
        print("EQUIPMENT:", rec['Equipment'])
        print("DIET:", rec['Diet'])

    # 6. Collect feedback
    feedback_matrix = []
    for i in range(len(all_recs)):
        feedback = int(input(f"\nWas recommendation {i+1} relevant? (Yes: 1, No: 0): "))
        feedback_matrix.append(feedback)

    # 7. Compute MRR
    relevant_indices = [i + 1 for i, f in enumerate(feedback_matrix) if f == 1]
    if relevant_indices:
        mrr = np.mean([1 / rank for rank in relevant_indices])
    else:
        mrr = 0.0

    print(f"\n📊 Mean Reciprocal Rank (MRR): {mrr:.2f}")

    return all_recs

# Call the function and display returned output
recommendations = get_recommendation()
print("\n✅ Final Returned Recommendations:")
for i, rec in enumerate(recommendations, 1):
    print(f"\nRecommendation {i}")
    print("EXERCISES:", rec['Exercises'])
    print("EQUIPMENT:", rec['Equipment'])
    print("DIET:", rec['Diet'])