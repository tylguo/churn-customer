import streamlit as st 
from st_files_connection import FilesConnection
import pandas as pd 
import numpy as np
# import scipy as sp
import pickle
# import datetime as dt
import json
import xgboost as xgb
import sklearn
import boto3
from io import BytesIO
import feature_engine
# from io import StringIO
# from collections import Counter


# title of the Web App
st.title("Customer Churn Risk Score Predictor")
st.subheader("This application predicts the risk score associated with a customer leaving in some way (cancelling subscriptions, stop purchasing goods/services, etc.) Scores range between 1 and 5, least likely to most likely")
st.write("Specify input conditions (parameters)")

# S3 bucket details
bucket_name = 'churn-challenge'
file_key = 'churn-challenge/cleaned_data.csv'

conn = st.connection('s3', type=FilesConnection)

# Read the CSV file from S3
df = conn.read("churn-challenge/clean_data.csv", input_format="csv", ttl=600)
freq_dict = conn.read("churn-challenge/freq_dict.json", input_format="json", ttl=600)

# fe_pipe = conn.read("churn-challenge/fe_pipe.pkl", ttl=600)
# model = conn.read("churn-challenge/xgb3.pkl", ttl=600)

del df["Unnamed: 0"]
# load saved model
# show team how to make a bucket with the secret key, etc.

s3 = boto3.resource('s3')
   
with BytesIO() as mod:
   s3.Bucket("churn-challenge").download_fileobj("model.pkl", mod)
   mod.seek(0)    # move back to the beginning after writing
   model = pickle.load(mod)

with BytesIO() as file:
   s3.Bucket("churn-challenge").download_fileobj("fe_pipe.pkl", file)
   file.seek(0)    # move back to the beginning after writing
   fe_pipe = pickle.load(file)

# st.dataframe(df)

# transform the user_input as we have been transforming the data as before
def user_inputs():
    """
    define inputs should a user input into the app using streamlit's functions

    be sure to check all steps where we changed outliers, cleaned up odd strings, 
    cleaned data, and ignored some variables from EDA and feature_importance

    Args:
        None

    Returns:
        df: dataframe containing a single data point (1 row) with relevant columns.

    """
    # numerical 
    age = st.slider("How old is the customer", min_value=1, max_value=80, step=1)
    days_since_last_login = st.slider("Days since last login", min_value=int(np.min(df["days_since_last_login"])), max_value=int(np.max(df["days_since_last_login"])), step=1)
    points_in_wallet = st.slider("Wallet Points", min_value=int(np.min(df["points_in_wallet"])), max_value=int(np.max(df["points_in_wallet"])))
    joining_date = st.number_input("Date joined: YYYYMMDD")
    avg_time_spent = st.slider("Average time spent", min_value=np.min(df['avg_time_spent']), max_value=np.max(df['avg_time_spent']))
    avg_frequency_login_days = st.selectbox("Average login days", df["avg_frequency_login_days"].unique())
    avg_transaction_value = st.number_input(f"Average Transaction value, between {np.min(df['avg_transaction_value'])} and {np.max(df['avg_transaction_value'])}")
    # categorical 
    membership_category = st.selectbox("Select Membership Category", df["membership_category"].unique())
    feedback = st.selectbox("Select Feedback", df["feedback"].unique())
    complaint_status = st.selectbox("Select Complaint Status", df["complaint_status"].unique())
    region_category = st.selectbox("Select Region Category", df["region_category"].unique())
    medium_of_operation = st.selectbox("Select Medium of Operation", df["medium_of_operation"].unique())
    preferred_offer_types = st.selectbox("Preferred Offer Types", df["preferred_offer_types"].unique())
    internet_option = st.selectbox("Select internet_option", df["internet_option"].unique())
    gender = st.selectbox("Gender", df["gender"].unique())
    used_special_discount = st.selectbox("Used Special Discount", df["used_special_discount"].unique())
    joined_through_referral = st.selectbox("Referral?", df["joined_through_referral"].unique())
    offer_application_preference = st.selectbox("Application Preference?", df["offer_application_preference"].unique())
    past_complaint = st.selectbox("Past Complaint", df["past_complaint"].unique())

    data = {
        'age': age,
        'days_since_last_login': days_since_last_login,
        'points_in_wallet': points_in_wallet,
        'joining_date': joining_date,
        'avg_time_spent': avg_time_spent,
        'avg_frequency_login_days': avg_frequency_login_days,
        'avg_transaction_value': avg_transaction_value,
        'membership_category': membership_category,
        'feedback': feedback,
        'complaint_status': complaint_status,
        'region_category': region_category,
        'medium_of_operation': medium_of_operation,
        'preferred_offer_types': preferred_offer_types,
        'internet_option': internet_option,
        'gender': gender,
        'used_special_discount': used_special_discount,
        'joined_through_referral': joined_through_referral,
        'offer_application_preference': offer_application_preference,
        'past_complaint': past_complaint}
    
    categorical = [
                    'gender',
                    'joined_through_referral',
                    'used_special_discount',
                    'offer_application_preference',
                    'past_complaint',
                    'region_category',
                    'membership_category',
                    'preferred_offer_types',
                    'medium_of_operation',
                    'internet_option',
                    'complaint_status',
                    'feedback']
                        
    # numerical = [i for i in data.keys() if i not in categorical]

    x_input = pd.DataFrame(data, index=[0])
    return x_input, categorical

def data_transform(df, user_input, freq_dict, categorical, fe_pipe):
    """
    define probability ratio encoding and/or other encodings that you have done.

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    per_columns = ['region_category',
       'membership_category', 'preferred_offer_types',
       'medium_of_operation', 'internet_option',
       'complaint_status', 'feedback']
    
    for c in per_columns:
        subdict = freq_dict[c]
        user_input[f'per_{c}'] = user_input[c].map(subdict)

    numerical = [c for c in user_input.columns if c not in categorical]
    x_numerical = user_input[numerical]

    # recall that feature engine pipeline takes care of OHE and Mean Encoding 
    x_pipe = user_input[categorical]
    t_x_pipe = fe_pipe.transform(x_pipe)

    x = pd.concat([t_x_pipe, x_numerical], axis=1)

    order = ['age', 'joining_date', 'days_since_last_login', 'avg_time_spent',
       'avg_transaction_value', 'avg_frequency_login_days', 'points_in_wallet',
       'per_region_category', 'per_membership_category',
       'per_preferred_offer_types', 'per_medium_of_operation',
       'per_internet_option', 'per_complaint_status', 'per_feedback',
       'region_category', 'membership_category', 'preferred_offer_types',
       'medium_of_operation', 'internet_option', 'complaint_status',
       'feedback', 'gender_F', 'gender_M', 'gender_Unknown',
       'joined_through_referral_No', 'joined_through_referral_Yes',
       'joined_through_referral_unknown', 'used_special_discount_Yes',
       'used_special_discount_No', 'offer_application_preference_Yes',
       'offer_application_preference_No', 'past_complaint_No',
       'past_complaint_Yes']

    x = x[order]
    return xgb.DMatrix(x)

# Predict with the model 
def predict(model, transformed):
    output = np.rint(model.predict(transformed))
    return output

def main():
    # A confirmation so the user knows what the input row looks like
    x_input, categorical = user_inputs()
    st.write('You selected:')
    st.dataframe(x_input)

    # design user interface
    if st.button("Predict Churn Risk Score"):
        transformed = data_transform(df, x_input, freq_dict, categorical, fe_pipe)
        prediction = predict(model, transformed)
        st.subheader("Prediction based on your inputs:")

        # here, define more informative statements, such as recommended actions, cautions, statistics you want to include, etc...
        st.write(f"...\n {prediction}\n")
        
if __name__ == "__main__":
    main()
