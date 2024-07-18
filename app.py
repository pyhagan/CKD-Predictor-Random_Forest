from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Load and prepare the dataset
ckd = pd.read_csv('kidney_disease_complete_2c.csv')
x = ckd.drop(["classification"], axis=1)
y = ckd["classification"]

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(x)

# Train the model
randomforest_classifier = RandomForestClassifier(n_estimators=10)
randomforest_classifier.fit(X, y)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Endpoint to predict CKD status based on input data.
    """
    try:
        data = request.json
        input_data = [
            data['age'],
            data['diastolic bp'],
            data['sg'],
            data['al'],
            data['su'],
            data['bgr'],
            data['bu'],
            data['sc'],
            data['sod'],
            data['pot'],
            data['hemo'],
            data['pcv'],
            data['wc'],
            data['rc'],
            data['rbc'],
            data['pc'],
            data['pcc'],
            data['ba'],
            data['htn'],
            data['dm'],
            data['cad'],
            data['pe'],
            data['ane']
        ]

        # Convert input data to numpy array and dataframe
        input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
        input_data_df = pd.DataFrame(input_data_as_numpy_array, columns=x.columns)

        # Standardize the input data
        input_data_transformed = scaler.transform(input_data_df)

        # Make prediction
        prediction = randomforest_classifier.predict(input_data_transformed)
        prediction_proba = randomforest_classifier.predict_proba(input_data_transformed)
        probability = prediction_proba[0][prediction[0]] * 100

        # Formulate result
        if prediction[0] == 0:
            result = f'The patient is likely suffering from CKD with a probability of {probability:.2f}%'
        else:
            result = f'The patient is healthy with a probability of {probability:.2f}%'

        # Get diet suggestion based on result
        diet_suggestion = diet_plan(result)

        return jsonify(result=result, diet_suggestion=diet_suggestion)

    except KeyError as e:
        return jsonify(error=f'Missing data for {str(e)}'), 400
    except Exception as e:
        return jsonify(error=str(e)), 500


def diet_plan(result):
    """
    Provide diet suggestions based on CKD status.
    """
    if "CKD" in result:
        diet_suggestion = """
        To improve your condition, do the following:

        - Consult with a dietitian for a personalized plan.
        - Drink adequate water, but not excessively.
        - Minimize intake of fizzy drinks.
        - Avoid alcohol.
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
        - Minimize alcohol intake.
        - Maintain a balanced diet with a variety of foods.
        - Ensure adequate hydration by drinking enough water.
        - Include whole grains and lean proteins.
        - Limit intake of processed foods and high-sodium snacks.
        - Avoid excessive amounts of sugar and saturated fats.
        """
    return diet_suggestion


if __name__ == '__main__':
    app.run(debug=True)  # Change debug=True to debug=False in production
