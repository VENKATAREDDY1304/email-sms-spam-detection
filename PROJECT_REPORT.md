# Project Report: Email / SMS Spam Detection Using Machine Learning

## Title
Email / SMS Spam Detection Using Natural Language Processing and Machine Learning

## Abstract
This project develops a text-classification system to distinguish spam messages from legitimate messages. Text is normalized and transformed into numerical features using Term Frequency–Inverse Document Frequency (TF-IDF). Two supervised learning algorithms—Multinomial Naive Bayes and Logistic Regression—are trained and evaluated on a held-out test set. The models are compared using accuracy, spam precision, spam recall, spam F1-score, and confusion matrices. The model with the best spam F1-score is saved and used by a Streamlit application that accepts a message and displays a prediction.

## 1. Introduction
Spam messages include unwanted promotions, deceptive offers, and potentially harmful communications. Manual filtering is difficult at scale. Machine learning can learn patterns from labelled examples and classify unseen messages based on their text.

## 2. Problem Statement
The increasing use of email and messaging platforms has led to a rise in unwanted and potentially deceptive messages. The aim is to develop a machine learning model that classifies a message as Spam or Legitimate using text features. NLP preprocessing and TF-IDF feature extraction are used with Naive Bayes and Logistic Regression, and their performance is compared using standard classification metrics.

## 3. Objectives
1. Obtain a labelled dataset containing spam and legitimate messages.
2. Clean and normalize message text.
3. Transform text into numerical TF-IDF features.
4. Train Multinomial Naive Bayes and Logistic Regression classifiers.
5. Compare models using a held-out test set.
6. Build an interactive interface for single-message prediction.

## 4. Dataset
The starter project uses the UCI SMS Spam Collection. Its labels are `ham` (legitimate) and `spam`. Download it from:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The dataset is SMS-focused. For a true email classifier, use a properly labelled email corpus and adapt the input columns. The model can only learn patterns represented in its training data.

## 5. Methodology
### 5.1 Data preparation
- Read the tab-separated dataset.
- Remove missing and duplicate rows.
- Normalize case and standardize URL/email placeholders.
- Split into 80% training and 20% testing data with stratification.

### 5.2 TF-IDF
TF-IDF assigns weights to terms based on how informative they are within a message and across the corpus. Unigrams and bigrams are used to capture single words and short phrases. The vectorizer is fitted only on training data through a scikit-learn pipeline to avoid test-data leakage.

### 5.3 Models
- **Multinomial Naive Bayes:** a probabilistic classifier commonly used for word-count and TF-IDF text features.
- **Logistic Regression:** a linear classifier that estimates class probabilities from text features.

### 5.4 Evaluation
- **Accuracy:** fraction of all predictions that are correct.
- **Precision (spam):** fraction of predicted spam messages that are actually spam.
- **Recall (spam):** fraction of actual spam messages detected.
- **F1-score (spam):** harmonic mean of spam precision and recall.
- **Confusion matrix:** counts correct and incorrect predictions for each class.

Because spam and legitimate examples may be imbalanced, accuracy should not be used alone.

## 6. System Architecture
Dataset → Text normalization → Train/test split → TF-IDF → Model training → Evaluation → Save selected pipeline → Streamlit prediction interface.

## 7. Implementation
The project is implemented in Python using Pandas, scikit-learn, Matplotlib, Seaborn, Joblib, and Streamlit. See `train_model.py` for training/evaluation and `app.py` for the interface.

## 8. Results
Run `python train_model.py` to generate the actual results. Copy the values from `reports/model_metrics.csv` into the table below after execution.

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1-score |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | To be measured | To be measured | To be measured | To be measured |
| Logistic Regression | To be measured | To be measured | To be measured | To be measured |

Do not fill in invented performance figures. Results can vary with the dataset and preprocessing choices.

## 9. Limitations
- The starter dataset contains SMS messages, not full email headers or attachments.
- Text-only classification may miss phishing messages that look legitimate.
- Language, writing style, and spam tactics change over time.
- Probability estimates are not guarantees of safety.
- False positives may cause legitimate messages to be incorrectly flagged.

## 10. Future Enhancements
- Train and evaluate on a labelled email dataset.
- Add sender, URL, and email-header features.
- Explore character-level features and transformer-based models.
- Add model monitoring and retraining as spam patterns change.
- Improve the interface with batch CSV prediction and downloadable results.

## 11. Conclusion
This project demonstrates a complete NLP classification workflow, from data preparation and TF-IDF feature extraction to model comparison and deployment in a small web application. The evaluation results provide a data-driven basis for selecting a model, while the limitations highlight the need for further testing before real-world security use.

## References
1. UCI Machine Learning Repository, SMS Spam Collection: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
2. scikit-learn documentation: https://scikit-learn.org/stable/
3. Streamlit documentation: https://docs.streamlit.io/
