# Fake News Detection — Identifying Real and Fake News

A machine learning project developed as part of the **Elevate Labs Data Analyst Internship** to identify whether a news article is Real or Fake using Natural Language Processing and Machine Learning.

## Project Objective

The objective of this project is to build a machine learning model that can classify news articles as **Real or Fake** based on their textual content.

## Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TF-IDF
- Logistic Regression
- Multinomial Naive Bayes
- Streamlit
- Joblib
- PyPDF

## Project Workflow

1. Data Collection
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis
4. Text Preprocessing
5. TF-IDF Vectorization
6. Model Training
7. Model Evaluation
8. Model Selection
9. Streamlit Application Development

## Machine Learning

The project uses Natural Language Processing (NLP) to process news articles and convert their textual content into numerical features using **TF-IDF Vectorization**.

Two classification models were explored:

- Logistic Regression
- Multinomial Naive Bayes

The trained model and TF-IDF vectorizer were saved using Joblib and integrated into the Streamlit application.

## Streamlit Application

The project includes an interactive Streamlit application that allows users to:

- Enter news article text
- Upload `.txt` files
- Upload `.pdf` files
- Get a Real or Fake prediction
- View prediction confidence
- View previous predictions during the session

## Project Structure

```text
News-Project/
│
├── app.py
├── news_classification.ipynb
├── news_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md
