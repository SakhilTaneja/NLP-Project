import joblib
import pandas as pd
import spacy
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline

# Load the data
df = pd.read_csv('reports_complete.csv')

#Data Prepocessing
#python -m spacy download en_core_web_sm
nlp=spacy.load('en_core_web_sm')
#nlp=en_core_web_sm.load()

def preprocess(text):
    text=re.sub(re.compile('<.*?>'), '', text)
    text=text.lower()
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    return ' '.join(filtered_tokens)

def preprocessing(df):
    # Preprocess the data
    df=df[df['judgement_result']!=2]
    df['preprocessed_judgements']=df['judgement'].apply(lambda judgement: preprocess(judgement))    
    # Handling Class Imbalance
    count_class_1,count_class_0=df.judgement_result.value_counts()
    df_class_1=df[df['judgement_result']==1]
    df_class_0=df[df['judgement_result']==0]
    # Oversample 1-class and concat the DataFrames of both classes
    df_class_0_over = df_class_0.sample(count_class_1, replace=True)
    df_test_over = pd.concat([df_class_1, df_class_0_over], axis=0)
    return df_test_over

def split_data(df_test_over):
    #Splitting the data
    X=df_test_over['preprocessed_judgements']
    y = df_test_over['judgement_result']
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=15, stratify=y)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    # Model pipeline
    model = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', AdaBoostClassifier())
    ])
    # Train the model
    model.fit(X_train, y_train)
    # Save the model
    joblib.dump(model, 'text_classifier.pkl')
    print("Classification model saved to text_classifier.pkl")
    return model

# Function to load the model and make predictions
def classify(text):
    model = joblib.load('text_classifier.pkl')
    text=preprocess(text)
    return model.predict([text])[0]

if __name__ == "__main__":
    df_test_over=preprocessing(df)
    X_train, X_test, y_train, y_test=split_data(df_test_over)
    model=train_model(X_train, y_train)
    print(f"result = {classify(df['preprocessed_judgements'][0])}")
