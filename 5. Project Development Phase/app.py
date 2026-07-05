from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("models/card_model.joblib")

# Label Encoding Dictionaries
gender_map = {
    "Male": 1,
    "Female": 0
}

income_map = {
    "Salaried": 0,
    "Business": 1,
    "Self-Employed": 2
}

education_map = {
    "High School": 0,
    "Graduate": 1,
    "Postgraduate": 2
}

marital_map = {
    "Single": 0,
    "Married": 1
}

housing_map = {
    "Own": 0,
    "Rent": 1
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    gender = gender_map[request.form["gender"]]

    income_type = income_map[request.form["income_type"]]

    annual_income = float(request.form["annual_income"])

    employment_duration = float(request.form["employment_duration"])

    education = education_map[request.form["education"]]

    marital = marital_map[request.form["marital_status"]]

    housing = housing_map[request.form["housing_type"]]

    credit_history = int(request.form["credit_history"])

    loan_amount = float(request.form["loan_amount"])

    debt = float(request.form["existing_debt"])

    data = pd.DataFrame([[

        gender,

        income_type,

        annual_income,

        employment_duration,

        education,

        marital,

        housing,

        credit_history,

        loan_amount,

        debt

    ]],
    columns=[

        "Gender",

        "Income_Type",

        "Annual_Income",

        "Employment_Duration",

        "Education_Level",

        "Marital_Status",

        "Housing_Type",

        "Credit_History",

        "Loan_Amount",

        "Existing_Debt"

    ])

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "✅ Credit Card Approved"
    else:
        result = "❌ Credit Card Rejected"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)