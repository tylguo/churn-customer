# churn

Slides: https://docs.google.com/presentation/d/154CZFGDa6q2nYaiFpdvI7hrswipfTKf2nvhQv2mAKKw/edit?usp=sharing

Demo App with Streamlit: https://customer-churn-risk.streamlit.app/

1. Problem Statement

Companies that have a subscription model heavily rely on customer royalty to thrive. However, it is not always clear why some customers decide to unsubcribe to services, even if they may provide reasons before they cancel their services. There is thus a need for data science to help companies predict if a particular customer is going to leave so they may do targeted marketing to prevent them from leaving. 

2. Data Science/Machine Learning Project Steps

#### Data Collection

Identified a dataset from a ML competition, which corresponds to a telecommunication company's data to understand why customers leave.

#### Data Exploration & Preprocessing

Performed Exploratory Data Analysis (EDA) to understand data patterns and relationships.
Cleaned the data by handling missing values, outliers, and inconsistencies.
Ensured data quality and address any ethical considerations related to privacy and bias (such as removing names and other personally identifiable information).

#### Feature Engineering & Selection

Created new features from existing ones to potentially improve model performance, especially probability-based features to encode textual data.
Selected the most relevant features to reduce dimensionality and prevent overfitting with feature selection techniques. 

#### Model Selection & Training

Chose appropriate machine learning algorithms based on the problem type and data characteristics.
Split data into training, validation, and test sets.
Trained the selected model, an XGBoost classifier and tuned hyperparameters for optimal performance.

#### Model Evaluation & Deployment

Evaluated the final model's performance on a held-out test set.
Deployed the model on Streamlit by first designing an intuitive Streamlit interface and then hosting in with private app keys and secrets.

3. Outcomes Achieved

#### Quantifiable Results:

Built a streamlit web application that users can input information and see how risky a customer is to unsubscribe (on a scale of 1-5). Improved F-1 score by 10%+ after more feature engineering and model parameter tuning. 

#### Impact:

When the parent company knows who are at highest risk, they are now able to conduct targeted marketing, significantly saving time and money spent on customer retention.

