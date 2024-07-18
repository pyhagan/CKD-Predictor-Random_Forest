import numpy as np
import pandas as pd
from flask import Flask, request,jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Load the dataset
ckd = pd.read_csv('kidney_disease_complete_2c.csv')

# Preprocess data as before
x = ckd.drop(["classification"], axis=1)
y = ckd["classification"]

# Split data for training and testing
scaler = StandardScaler()
X = scaler.fit_transform(x)

# Train RandomForestClassifier
randomforest_classifier = RandomForestClassifier(n_estimators=10, random_state=2)
randomforest_classifier.fit(X, y)

# Function to determine diet plan based on prediction result
def diet_plan(result):
    if result == "The patient is likely suffering from CKD":
        diet_suggestion = """
        To improve your condition, do the following:
        
        - Consult with a dietitian for a personalized plan.
        - Drink adequate water, but not excessively.
        - Minimize intake of fizzy drinks.
        - Avoid alcohol
        - Limit protein intake to reduce the workload on kidneys.
        - Choose high-quality protein sources like fish, poultry, and eggs.
        - Reduce sodium intake to control blood pressure.
        - Limit foods high in phosphorus such as dairy, nuts, seeds.
        - Limit potassium-rich foods such as bananas, oranges, potatoes.
        """
    else:
        diet_suggestion = """
        Diet Suggestions for Healthy Kidneys:
        
        - Eat plenty of fruits and vegetables.
        - Minimize intake of fizzy drinks.
        - Minimize alcohol intake
        - Maintain a balanced diet with a variety of foods.
        - Ensure adequate hydration by drinking enough water.
        - Include whole grains and lean proteins.
        - Limit intake of processed foods and high-sodium snacks.
        - Avoid excessive amounts of sugar and saturated fats.
        """
    return diet_suggestion



@app.route('/predict', methods=['GET','POST'])
def predict():
        data = request.json
        input_data = [
            data['age'], data['diastolic bp'], data['sg'], data['al'], data['su'], data['bgr'], data['bu'],
            data['sc'], data['sod'], data['pot'], data['hemo'], data['pcv'], data['wc'], data['rc'],
            data['rbc'], data['pc'], data['pcc'], data['ba'], data['htn'], data['dm'], data['cad'],
            data['pe'], data['ane']
        ]
        
        input_data = np.array(input_data).reshape(1, -1)
        
        prediction = randomforest_classifier.predict(input_data)
        prediction_proba = randomforest_classifier.predict_proba(input_data)
        probability = prediction_proba[0][prediction[0]] * 100
        
        if prediction[0] == 0:
            prediction_result = f'The patient is likely suffering from CKD with a probability of {probability:.2f}%'
            diet_suggestion = diet_plan("The patient is likely suffering from CKD")
        else:
            result = f'Patient is healthy with a probability of {probability:.2f}%'
            diet_suggestion = diet_plan("Patient is healthy")
        
        return jsonify({'Prediction': prediction_result, 'Diet Suggestion': diet_suggestion})
        
        
if __name__ == '__main__':
    app.run(debug=True)
