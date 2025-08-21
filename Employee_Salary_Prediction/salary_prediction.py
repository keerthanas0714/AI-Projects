import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
df=pd.read_csv('Salary_Data.csv')
print(df.info())
print(df.describe())
print(df.head())
print(df.isnull().sum())
df.dropna(inplace=True)
print(df.isnull().sum())
def encode_categorical_columns(df):
    df=df.copy()
    cat_col=df.select_dtypes(include='object').columns
    binary_cols= [col for col in cat_col if df[col].nunique()==2]
    multi_class_col=[col for col in cat_col if df[col].nunique()>2]
    le= LabelEncoder()
    for col in binary_cols:
        df[col]=le.fit_transform()
    df=pd.get_dummies(df,columns=multi_class_col, drop_first=True)
    df = df.astype({col: int for col in df.columns if df[col].dtype == 'bool'})
    return df
#df.drop(columns=['Age','Years of Experience','Salary'], inplace=True)
df_encoded=encode_categorical_columns(df)
print(df_encoded.head())
X=df_encoded.drop('Salary', axis=1)
y=df_encoded['Salary']
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
print(X_scaled)
X_train, X_test, y_train, y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
