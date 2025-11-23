from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def start_page():
    # Show clean form (NO result)
    return render_template('index.html', result=None, css_class=None)


@app.route('/predict', methods=['POST'])
def predict():
    married = int(request.form['married'])
    income = float(request.form['income'])
    loan_amount = float(request.form['loan_amount'])
    education = int(request.form['education'])
    self_emp = int(request.form['self_emp'])
    term = float(request.form['term'])
    credit = int(request.form['credit'])
    area = int(request.form['area'])

    features = np.array([[married, income, loan_amount, area, credit, term, self_emp, education]])
    prediction = model.predict(features)[0]

    if prediction == 1:
        result = "✔ Loan Approved"
        css_class = "result-success"
    else:
        result = "✖ Loan Not Approved"
        css_class = "result-fail"

    # Show RESULT only once
    return render_template('index.html', result=result, css_class=css_class)


if __name__ == "__main__":
    app.run(debug=True)
