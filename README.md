# Email / SMS Spam Detection Using Machine Learning

A complete beginner-friendly Data Science project using NLP, TF-IDF, Multinomial Naive Bayes, Logistic Regression, and Streamlit.

## Features
- Loads a labelled SMS dataset (`spam` / `ham`)
- Cleans text and splits the data into training and testing sets
- Trains and compares Naive Bayes and Logistic Regression
- Reports accuracy, precision, recall, F1-score, and confusion matrices
- Saves the best model pipeline
- Provides a Streamlit interface to test messages

## Dataset
Download the **SMS Spam Collection** from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The downloaded archive commonly contains a file named `SMSSpamCollection` with tab-separated label and message columns. Place that file in `data/SMSSpamCollection`.

This is an SMS dataset, not a full email dataset. The same workflow can be adapted to an email dataset with labelled spam and legitimate emails.

## Setup (Windows)
1. Install Python 3.10 or newer.
2. Open Command Prompt in this folder.
3. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Put the dataset at `data/SMSSpamCollection`.
6. Train and evaluate:
   ```bash
   python train_model.py
   ```
7. Launch the app:
   ```bash
   streamlit run app.py
   ```

## Files created after training
- `models/spam_model.joblib` — trained pipeline
- `reports/model_metrics.csv` — model comparison
- `reports/confusion_matrix_*.png` — confusion matrices
- `reports/classification_report_*.txt` — detailed metrics

## Important
- Results depend on the dataset and train/test split; do not claim an accuracy before running evaluation.
- A spam prediction is a model estimate, not a guarantee that a message is malicious.
- The model uses text only. It does not inspect links, sender reputation, attachments, or email headers.
