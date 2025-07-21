import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
df=pd.read_csv(r"C:\Users\keerthanar\Desktop\AI Projects\spam_classifier_project\spam_sample.csv")
print(df.head())
print(df.isnull())

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words=set(stopwords.words('english'))
stemmer=PorterStemmer()
def preprocess_text(text):
    text=text.lower()
    tokens= word_tokenize(text)
    filtered_tokens = [
        stemmer.stem(word)
        for word in tokens
        if word not in stop_words and word not in string.punctuation

    ]
    return ''.join(filtered_tokens)


df['clean_text'] = df['message'].apply(preprocess_text)
print(df[['message','clean_text']].head())


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer()

X= vectorizer.fit_transform(df['clean_text'])
print("TF-IDF feature matrix shape", X.shape)

tf_idf_df= pd.DataFrame(X.toarray(), columns= vectorizer.get_feature_names_out())
print(tf_idf_df.head())

y= df['label']
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

model= MultinomialNB()
model.fit(X_train, y_train)

y_pred=model.predict(X_test)


print("Accuracy:", accuracy_score(y_test,y_pred))
print("Confusion matrix:", confusion_matrix(y_test, y_pred))
print("Classification report:", classification_report(y_test, y_pred))
print(model.predict(vectorizer.transform(["Congratulations! You’ve won!"])))
