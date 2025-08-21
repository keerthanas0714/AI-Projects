import string
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
df_fake= pd.read_csv("Fake.csv")
df_True= pd.read_csv("True.csv")
print(df_fake.head())
print(df_True.head())
df_True['label']=0
df_fake['label']=1
df= pd.concat([df_fake,df_True], ignore_index=True)
df=df.sample(frac=1, random_state=42).reset_index(drop=True)
print(df.head())
df=df.drop(columns=['subject', 'date'])
print(df.head())
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
df['title'] = df['title'].fillna('')
df['text'] = df['text'].fillna('')
df['content'] = df['title'] + " " + df['text']
print(df['content'].dtype)
nltk.download('stopwords')
stopwords= set(stopwords.words('english'))
stemmer= PorterStemmer()
def clean_text(text):
    text=text.lower()
    text=text.translate(str.maketrans('','',string.punctuation))
    words=text.split()
    words=[stemmer.stem(word) for word in words if word not in stopwords]
    return "".join(words)
print(df['content'].iloc[0])  # See the first row's content
print(clean_text(df['content'].iloc[0]))  # Try cleaning just one row
df['content'] = df['content'].astype(str)
df['clean_text']=df['content'].apply(clean_text)
print(df[['label','clean_text']].head())
X=df['clean_text']
y=df['label']
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=42)
tfidf=TfidfVectorizer(max_features=500)
X_train_tfidf=tfidf.fit_transform(X_train)
X_test_tfidf=tfidf.fit_transform(X_test)
model= LogisticRegression()
model.fit(X_train_tfidf,y_train)
y_pred=model.predict(X_test_tfidf)
print("confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))