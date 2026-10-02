# Fake News Detection — Identifying Real and Fake News

A machine learning project developed as part of the **Elevate Labs Data Analyst Internship** to identify whether a news article is Real or Fake using Natural Language Processing and Machine Learning.

## Project Objective

The objective of this project is to build a machine learning model that can classify news articles as **Real or Fake** based on their textual content.

## ✨ Features

- 📰 **Real/Fake News Classification**  
  Classifies a news article as Real or Fake using a trained machine learning model.

- 📊 **Confidence Breakdown**  
  Displays a visual percentage breakdown showing how strongly the model predicts the article as Real or Fake.

- 📄 **PDF File Upload**  
  Users can upload a PDF containing a news article, and the application extracts the text for classification.

- 📝 **TXT File Upload**  
  Users can upload a `.txt` file containing news article text for analysis.

- ✍️ **Manual Article Input**  
  Users can directly paste or type a news article into the application.

- 🕒 **Prediction History**  
  The application keeps a history of articles checked during the current session, including the prediction and confidence.

- 🗑️ **Clear History**  
  Users can clear the prediction history whenever required.

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

## 📊 Results

Two machine learning models were evaluated for fake and real news classification.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 99.22% | 98.89% | 99.49% | 99.19% |
| Multinomial Naive Bayes | 95.19% | 94.65% | 95.31% | 94.98% |

Logistic Regression was selected as the final model and integrated into the Streamlit application.

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
```

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/ZaraSiddiqi7/Fake-News-Detection-Identifying-Real-and-Fake-News.git
```

### 2. Open the Project Folder

```bash
cd news-project
```

### 3. Install the Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

### 5. Open the Application

After running the command, open the local URL provided by Streamlit in your browser.

## 💼 Internship

This project was developed as part of my **Data Analyst Internship at Elevate Labs**.

The project focuses on using Natural Language Processing (NLP) and Machine Learning to classify news articles as Real or Fake.
