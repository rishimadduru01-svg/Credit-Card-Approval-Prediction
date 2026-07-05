from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/card_model.joblib")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [
        float(request.form["Ind_ID"]),
        float(request.form["GENDER"]),
        float(request.form["Car_Owner"]),
        float(request.form["Propert_Owner"]),
        float(request.form["CHILDREN"]),
        float(request.form["Annual_income"]),
        float(request.form["Type_Income"]),
        float(request.form["EDUCATION"]),
        float(request.form["Marital_status"]),
        float(request.form["Housing_type"]),
        float(request.form["Birthday_count"]),
        float(request.form["Employed_days"]),
        float(request.form["Mobile_phone"]),
        float(request.form["Work_Phone"]),
        float(request.form["Phone"]),
        float(request.form["EMAIL_ID"]),
        float(request.form["Type_Occupation"]),
        float(request.form["Family_Members"])
    ]

    prediction = model.predict([features])[0]

    if prediction == 1:
        result = "✅ Credit Card Approved"
    else:
        result = "❌ Credit Card Rejected"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)