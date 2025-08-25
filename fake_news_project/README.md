# 📰 Fake News Detection using Logistic Regression & TF-IDF

This project implements a **Fake News Detection system** using **Logistic Regression** and **TF-IDF vectorization**.  
It classifies news articles as either **Fake (1)** or **True (0)**.

---

## 📌 Project Overview
- Dataset: Two CSV files (`Fake.csv` and `True.csv`) containing fake and true news articles.
- Preprocessing:
  - Removed unnecessary columns (`subject`, `date`).
  - Combined `title` and `text` into a single `content` column.
  - Lowercased text, removed punctuation, stopwords, and applied stemming.
- Feature Extraction: Used **TF-IDF (Term Frequency - Inverse Document Frequency)** with 500 max features.
- Model: **Logistic Regression** for classification.
- Evaluation Metrics: 
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)
  - Accuracy Score

---

## 📂 Dataset
The dataset contains:
- **Fake.csv** → News articles labeled as *fake*.
- **True.csv** → News articles labeled as *true*.

Each file includes:
- `title` → Headline of the article  
- `text` → Main body of the article  
- `subject` → Category of news (dropped in preprocessing)  
- `date` → Publication date (dropped in preprocessing)  

---

## 🚀 Installation & Setup

 Download NLTK stopwords (first time only):
   ```python
   import nltk
   nltk.download('stopwords')
   ```

4. Place `Fake.csv` and `True.csv` inside the project folder.

---

## 🧹 Data Preprocessing

- Convert text to lowercase  
- Remove punctuation  
- Remove stopwords  
- Apply stemming (Porter Stemmer)  
- Merge title and text → `content`  
- Create `clean_text` column for training  



---

## 📊 Model Training

- Split dataset (80% train, 20% test).
- Convert text into **TF-IDF vectors**.
- Train a **Logistic Regression** classifier.

---

## ✅ Results

- **Confusion Matrix**  
- **Classification Report** (Precision, Recall, F1-score)  
- **Accuracy Score**

Example Output:
```
Confusion Matrix:
[[420   5]
 [ 10 400]]

Classification Report:
              precision    recall  f1-score   support
           0       0.98      0.99      0.98       425
           1       0.99      0.98      0.98       410

Accuracy Score: 0.985
```

---

