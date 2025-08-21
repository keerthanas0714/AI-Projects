# 📧 Spam Email Classification using Naive Bayes & TF-IDF

This project implements a **Spam Email Classifier** using **Multinomial Naive Bayes** and **TF-IDF vectorization**.  
It classifies messages as either **Spam** or **Ham (Not Spam)**.

---

## 📌 Project Overview
- Dataset: `spam_sample.csv` containing email/text messages and their labels.
- Preprocessing:
  - Converted text to lowercase.
  - Tokenized text using NLTK.
  - Removed stopwords and punctuation.
  - Applied stemming using **Porter Stemmer**.
- Feature Extraction: Used **TF-IDF (Term Frequency - Inverse Document Frequency)**.
- Model: **Multinomial Naive Bayes** for classification.
- Evaluation Metrics:
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)
  - Accuracy Score

---

## 📂 Dataset
The dataset contains:
- `message` → The email/text message.  
- `label` → The target (spam or ham).  

Example:
```
label    message
spam     Congratulations! You’ve won!
ham      Are we meeting tomorrow?
```

---

## 🚀 Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/spam-classifier.git
   cd spam-classifier
   ```

2. Install dependencies:
   ```bash
   pip install pandas scikit-learn nltk
   ```

3. Download NLTK resources (first time only):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

4. Place `spam_sample.csv` inside the project folder.

---

## 🧹 Data Preprocessing

Steps applied in preprocessing:
- Convert text to lowercase  
- Tokenization (`word_tokenize`)  
- Remove stopwords and punctuation  
- Apply stemming with Porter Stemmer  

Example function:
```python
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    filtered_tokens = [
        stemmer.stem(word)
        for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]
    return " ".join(filtered_tokens)
```

---

## 📊 Model Training

- Feature extraction using TF-IDF:
```python
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
```

- Train-Test Split (80/20).  
- Train model using **Multinomial Naive Bayes**.  
- Evaluate with accuracy, confusion matrix, and classification report.

---

## ✅ Results

Example Output:
```
Accuracy: 0.97
Confusion Matrix:
[[965   3]
 [ 12 150]]

Classification Report:
              precision    recall  f1-score   support
         ham       0.99      0.99      0.99       968
        spam      0.98      0.93      0.96       162
```

---

## 📌 Notes
- You can experiment with **Logistic Regression, SVM, or Deep Learning models** for better performance.  
- Try tuning `TfidfVectorizer` parameters (e.g., `max_features`, `ngram_range`).

---

## 📜 License
This project is open-source and available under the **MIT License**.
