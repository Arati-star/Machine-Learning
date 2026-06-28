from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and preprocessor
model = pickle.load(open("model.pkl", "rb"))
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    input_data = pd.DataFrame({
        'Age': [int(request.form['Age'])],
        'MonthlyIncome': [int(request.form['MonthlyIncome'])],
        'OverTime': [request.form['OverTime']],
        'JobSatisfaction': [int(request.form['JobSatisfaction'])],
        'YearsAtCompany': [int(request.form['YearsAtCompany'])],
        'TotalWorkingYears': [int(request.form['TotalWorkingYears'])],
        'JobLevel': [int(request.form['JobLevel'])],
        'Department': [request.form['Department']]
    })

    processed_data = preprocessor.transform(input_data)

    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]

    if prediction == 1:
        result = f"Employee is likely to leave the company. Probability: {probability:.2%}"
    else:
        result = f"Employee is likely to stay in the company. Attrition Probability: {probability:.2%}"

    return render_template(
        'index.html',
        prediction_text=result
    )

if __name__ == "__main__":
    app.run(debug=True)