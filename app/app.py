from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('models/rf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    # features = np.array([data['Age'], data['Income'], data['LoanAmount'], data['CreditScore'], data['MonthsEmployed'], data['NumCreditLines'], data['InterestRate'], data['LoanTerm'], data['DTIRatio'], data['Education'], data['EmploymentType'], data['MaritalStatus'], data['HasMortgage'], data['HasDependents'], data['LoanPurpose'], data['HasCoSigner']])
    features=[np.array(int_features)]
    prediction = model.predict(features)

    output = round(prediction[0],2)
    answer=""
    if output == 0:
        answer = "no default"
    else:
        answer = "default"
    return render_template('result.html', prediction=answer+"   ")

if __name__ == '__main__':
    app.run(debug=True)
