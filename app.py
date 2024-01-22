import streamlit as st
import pandas as pd
import requests
from PIL import Image

# Cargar y mostrar el logo
logo = Image.open('logo.png') 
st.image(logo, width=200) 

# Título y subtítulo
st.title("SafeJobFinder")
st.write("Your job search assistant to avoid fraud")


url = "http://localhost:9696/predict"
def send_to_model(row):
    # Convert float to a fixed decimal string if necessary
    if isinstance(row, pd.Series):
        row = row.apply(lambda x: f"{x:.2f}" if isinstance(x, float) else x).to_dict()
    # row = row.apply(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
    job_id = str(row['job_id'])
    job_title = row['title']
    job_location = row['location']
    response = requests.post(url, json=
                             row #.to_dict()
                             )
    if response.status_code == 200:
        if response.json()['job']== True:
            # return 'The job with id '+ job_id + ' / '+ job_title +' in '+job_location+' seems to be a fraud'
            return f"The job with id {job_id} / {job_title} in {job_location} seems to be a fraud"
        else:
            # return 'The job with id '+ job_id + ' / '+ job_title +' in '+job_location+' seems to be a real job'
            return f"The job with id {job_id} / {job_title} in {job_location} seems to be a real job"
    else:
        st.error("Error in model response: " + response.text)
        return None

# Page configuration
st.title("Fake Job Classifier")
st.write("Upload a CSV file or enter the job details for classification.")

# CSV File Uploader
uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if st.button('Process File') and uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if len(data) == 1:
        result = send_to_model(data.iloc[0])
        if result:
            st.write(result)
        else:
            st.write("Error processing the file.")
    else:
        st.error("The CSV file should contain only one data row. Please correct the file and upload again.")
else:
    # Fields for manual data entry
    with st.form("job_input_form"):
        job_id = st.number_input('Job ID', min_value=1, format='%d')
        title = st.text_input('Title')
        location = st.text_input('Location')
        department = st.text_input('Department', value="")
        salary_range = st.text_input('Salary Range', value="")
        company_profile = st.text_input('Company Profile', value="")
        description = st.text_input('Description', value="")
        requirements = st.text_input('Requirements', value="")
        benefits = st.text_input('Benefits', value="")
        telecommuting = st.selectbox('Telecommuting', [0, 1], format_func=lambda x: 'Sí' if x == 1 else 'No')
        has_company_logo = st.selectbox('Has Company Logo', [0, 1], format_func=lambda x: 'Sí' if x == 1 else 'No')
        has_questions = st.selectbox('Has Questions', [0, 1], format_func=lambda x: 'Sí' if x == 1 else 'No')
        employment_type = st.text_input('Employment Type', value="")
        required_experience = st.text_input('Required Experience', value="")
        required_education = st.text_input('Required Education', value="")
        industry = st.text_input('Industry', value="")
        function = st.text_input('Function', value="")

        submit_button = st.form_submit_button('Classify')

        if submit_button:
            job_data = {
                'job_id': job_id,
                'title': title,
                'location': location,
                'department': department,
                'salary_range': salary_range,
                'company_profile': company_profile,
                'description': description,
                'requirements': requirements,
                'benefits': benefits,
                'telecommuting': telecommuting,
                'has_company_logo': has_company_logo,
                'has_questions': has_questions,
                'employment_type': employment_type,
                'required_experience': required_experience,
                'required_education': required_education,
                'industry': industry,
                'function': function
            }

            result = send_to_model(job_data)
            if result:
                st.write(result)
