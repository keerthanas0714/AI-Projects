import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df=pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
print(df.info())
print(df.isnull().sum())
df.drop(columns=['EmployeeNumber','EmployeeCount', 'Over18', 'StandardHours'], inplace=True)
def encode_categorical_columns(df):
    df = df.copy()  # avoid modifying original
    # Identify categorical columns
    cat_cols = df.select_dtypes(include='object').columns
    binary_cols = [col for col in cat_cols if df[col].nunique() == 2]
    multi_class_cols = [col for col in cat_cols if df[col].nunique() > 2]
    # Apply Label Encoding to binary columns
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])
    # Apply One-Hot Encoding to multi-class columns
    df = pd.get_dummies(df, columns=multi_class_cols, drop_first=True)
    # Convert any bool columns to int
    df = df.astype({col: int for col in df.columns if df[col].dtype == 'bool'})
    return df
df_encoded=encode_categorical_columns(df)
print(df_encoded.head())
X=df_encoded.drop('Attrition', axis=1)
y=df_encoded['Attrition']
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_train, X_test, y_train, y_test=train_test_split(X_scaled,y,random_state=42,test_size=0.2, stratify=y)
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_pred_logreg=logreg.predict(X_test)

rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred_rf=rf.predict(X_test)
# Accuracy
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
# Detailed classification report
print("\n--- Logistic Regression Report ---")
print(classification_report(y_test, y_pred_logreg))
print("\n--- Random Forest Report ---")
print(classification_report(y_test, y_pred_rf))
#Confusion Matrix
print("Confusion Matrix (LogReg):\n", confusion_matrix(y_test, y_pred_logreg))
print("Confusion Matrix (RF):\n", confusion_matrix(y_test, y_pred_rf))