
import pickle
from flask import Flask
from flask import request
from flask import jsonify
import pandas as pd
import numpy as np


# Load the model

input_file = 'model.bin'


# Load the model from the pickle file
with open(input_file, 'rb') as f_in: 
    dv, model = pickle.load(f_in)


app = Flask('fake_job')

def fill_missing_values(x):
    if type(x) == str:
        return 1
    else:
        return 0

@app.route('/predict', methods=['POST'])
def predict():
    job = request.get_json()
    # Delete id and numberflight from the request
    df = pd.DataFrame([job])
    if 'job_id' in df.columns:
        del df['job_id']
    if 'salary_range' in df.columns:
        del df['salary_range']

    if 'fraudulent' in df.columns:
        del df['fraudulent']

    df['location'] = df['location'].fillna("No location")
    df['country'] = df['location'].apply(lambda x: x.split(", ")[0])
    del df['location']

    df['benefits'] = df['benefits'].apply(fill_missing_values)
    df['company_profile'] = df['company_profile'].apply(fill_missing_values)
    df['description'] = df['description'].apply(fill_missing_values)
    df['requirements'] = df['requirements'].apply(fill_missing_values)

    df['department'] = df['department'].fillna("No registered")
    df['required_education'] = df['required_education'].fillna("No registered")
    df['required_experience'] = df['required_experience'].fillna("No registered")
    df['industry'] = df['industry'].fillna("No registered")
    df['function'] = df['function'].fillna("No registered")
    df['employment_type'] = df['employment_type'].fillna("No registered")

    df = df [['department','company_profile','description','requirements','benefits','telecommuting','has_company_logo','has_questions','employment_type','required_experience','required_education','industry','function','country']]

    sample = df.to_dict(orient='records')[0]
    for key, value in sample.items():
        if isinstance(value, float) and np.isnan(value):
            sample[key] = None

    X = dv.transform([sample])
    y_pred = model.predict(X)[0]
    result = {
        'job': bool(y_pred)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0',port=9696)


