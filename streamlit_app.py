# Import necessary libraries and models
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

# Preprocess function to clean input text
def preprocess_text(text):
    text = text.lower()
    punc_numbers = string.punctuation + '0123456789'
    return ''.join([char for char in text if char not in punc_numbers])

# Step 1: Load and prepare the dataset
def load_data():
    # Load training and testing data
    train_articles = pd.read_csv('https://raw.githubusercontent.com/Jana-Liebenberg/2401PTDS_Classification_Project/refs/heads/main/Data/processed/train.csv')  
    test_articles = pd.read_csv('https://raw.githubusercontent.com/Jana-Liebenberg/2401PTDS_Classification_Project/refs/heads/main/Data/processed/test.csv')    

    # Combine text columns
    train_articles['combined_text'] = train_articles['headlines'] + ' ' + train_articles['description'] + ' ' + train_articles['content']
    test_articles['combined_text'] = test_articles['headlines'] + ' ' + test_articles['description'] + ' ' + test_articles['content']

    # Clean the text
    train_articles['combined_text'] = train_articles['combined_text'].apply(preprocess_text)
    test_articles['combined_text'] = test_articles['combined_text'].apply(preprocess_text)

    return train_articles, test_articles

# Step 2: Model training and saving function
def train_and_save_models():
    # Load data
    train_articles, test_articles = load_data()

    # Initialize vectorizer
    vect = CountVectorizer(stop_words='english')

    # Fit the vectorizer on the training set and transform the text
    X_train = vect.fit_transform(train_articles['combined_text'])
    y_train = train_articles['category']

    # Initialize label encoder to encode the labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(),
        'KNN': KNeighborsClassifier(3),
        'SVC Linear': SVC(kernel="linear", C=0.025),
        'SVC RBF': SVC(gamma=2, C=1)
    }

    # Fit and save each model
    for name, model in models.items():
        model.fit(X_train, y_train_encoded)
        joblib.dump(model, f'{name}.pkl')

    # Save the vectorizer and label encoder
    joblib.dump(vect, 'vectorizer.pkl')
    joblib.dump(le, 'label_encoder.pkl')

# Step 3: Streamlit UI for classification
def classify_text():
    st.title("News Article Classifier")

    # Input box for users to paste news articles
    input_text = st.text_area("Enter the news article text here:")

    # Load the saved vectorizer and label encoder
    vect = joblib.load('vectorizer.pkl')
    le = joblib.load('label_encoder.pkl')

    # Load the models
    models = {
        'Logistic Regression': joblib.load('Logistic Regression.pkl'),
        'KNN': joblib.load('KNN.pkl'),
        'SVC Linear': joblib.load('SVC Linear.pkl'),
        'SVC RBF': joblib.load('SVC RBF.pkl')
    }

    # Dropdown to select the model
    model_name = st.selectbox("Choose a model", list(models.keys()))

    # Button to trigger classification
    if st.button("Classify"):
        if input_text:
            # Preprocess and vectorize input
            cleaned_input = preprocess_text(input_text)
            input_vect = vect.transform([cleaned_input])

            # Load the selected model
            model = models[model_name]

            # Predict category
            prediction_encoded = model.predict(input_vect)
            prediction = le.inverse_transform(prediction_encoded)

            # Display prediction
            st.write(f"Using {model_name}, the article belongs to the category: {prediction[0]}")
        else:
            st.write("Please enter some text to classify.")

# Main function to trigger model training or classification based on user input
if __name__ == '__main__':
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose an option", ["Train Models", "Classify Article"])

    if option == "Train Models":
        st.write("Training models...")
        train_and_save_models()
        st.write("Models trained and saved successfully!")
    else:
        classify_text()
