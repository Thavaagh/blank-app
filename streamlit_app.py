import pandas as pd
import string
import joblib
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    punc_numbers = string.punctuation + '0123456789'
    return ''.join([char for char in text if char not in punc_numbers])

# Load datasets
def load_data():
    train_articles = pd.read_csv('https://raw.githubusercontent.com/Jana-Liebenberg/2401PTDS_Classification_Project/refs/heads/main/Data/processed/train.csv')
    test_articles = pd.read_csv('https://raw.githubusercontent.com/Jana-Liebenberg/2401PTDS_Classification_Project/refs/heads/main/Data/processed/test.csv')
    train_articles['combined_text'] = train_articles['headlines'] + ' ' + train_articles['description'] + ' ' + train_articles['content']
    test_articles['combined_text'] = test_articles['headlines'] + ' ' + test_articles['description'] + ' ' + test_articles['content']
    train_articles['combined_text'] = train_articles['combined_text'].apply(preprocess_text)
    test_articles['combined_text'] = test_articles['combined_text'].apply(preprocess_text)
    return train_articles, test_articles

# Train models and save
def train_and_save_models():
    train_articles, test_articles = load_data()
    vect = CountVectorizer(stop_words='english')
    X_train = vect.fit_transform(train_articles['combined_text'])
    y_train = train_articles['category']
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    models = {
        'Logistic Regression': LogisticRegression(),
        'KNN': KNeighborsClassifier(3),
        'SVC Linear': SVC(kernel="linear", C=0.025),
        'SVC RBF': SVC(gamma=2, C=1)
    }

    # Train and save models
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train_encoded)
        joblib.dump(model, f'{name}.pkl')
        st.write(f"Model {name} trained in {time.time() - start_time:.2f} seconds.")

    joblib.dump(vect, 'vectorizer.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    st.write("All models trained and saved successfully!")

# Classify text
def classify_text():
    st.title("News Article Classifier")
    input_text = st.text_area("Enter the news article text here:")
    vect = joblib.load('vectorizer.pkl')
    le = joblib.load('label_encoder.pkl')
    
    models = {
        'Logistic Regression': joblib.load('Logistic Regression.pkl'),
        'KNN': joblib.load('KNN.pkl'),
        'SVC Linear': joblib.load('SVC Linear.pkl'),
        'SVC RBF': joblib.load('SVC RBF.pkl')
    }
    
    model_name = st.selectbox("Choose a model", list(models.keys()))
    
    if st.button("Classify"):
        if input_text:
            cleaned_input = preprocess_text(input_text)
            input_vect = vect.transform([cleaned_input])
            model = models[model_name]
            prediction_encoded = model.predict(input_vect)
            prediction = le.inverse_transform(prediction_encoded)
            st.write(f"Using {model_name}, the article belongs to the category: {prediction[0]}")
        else:
            st.write("Please enter some text to classify.")

# Main function for Streamlit
if __name__ == '__main__':
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose an option", ["Train Models", "Classify Article"])

    if option == "Train Models":
        st.write("Training models...")
        train_and_save_models()
    else:
        classify_text()
