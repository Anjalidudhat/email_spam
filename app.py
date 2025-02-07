


# import streamlit as st
# import pickle
# import re
# import nltk
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from nltk.corpus import stopwords

# # Load trained RandomForest model and TfidfVectorizer
# with open("model/Email.pkl", "rb") as model_file:
#     model = pickle.load(model_file)

# with open("model/TfidfVectorizer.pkl", "rb") as vectorizer_file:
#     vectorizer = pickle.load(vectorizer_file)

# # Download NLTK resources
# nltk.download("stopwords")
# nltk.download("wordnet")

# # Text Preprocessing Tools
# stemmer = PorterStemmer()
# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words("english"))

# # Define text preprocessing function
# def preprocess_text(text):
#     text = re.sub(r"[^\w\s]", "", text).lower()
#     tokenized_text = text.split()
#     tokens = [word for word in tokenized_text if word not in stop_words]
#     tokens = [stemmer.stem(word) for word in tokens]
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     return " ".join(tokens)

# # Streamlit UI
# st.title("Email Spam Classifier")
# # st.write("Enter an email message below and click **Predict** to check if it's spam or not.")

# # Text Input
# email_input = st.text_area("Enter Email Content", height=150)

# # Prediction Button
# if st.button("Predict"):
#     if email_input.strip() == "":
#         st.error("⚠️ Please enter an email to classify.")
#     else:
#         # Preprocess input
#         clean_email = preprocess_text(email_input)

#         # Transform using the same trained vectorizer
#         email_tfidf = vectorizer.transform([clean_email])  # **Use transform(), not fit_transform()**

#         # Predict using RandomForest
#         prediction = model.predict(email_tfidf)[0]

#         # Display Prediction
#         if prediction == 1:
#             st.error(" This email is **SPAM**! ")
#         else:
#             st.success("This email is **NOT SPAM**!")



import streamlit as st
import pickle
import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and vectorizer
with open("model/Email.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("model/TfidfVectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

# Text Preprocessing Tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Function to preprocess email text
def preprocess_text(text):
    text = re.sub(r"[^\w\s]", "", text).lower()
    tokenized_text = text.split()
    tokens = [word for word in tokenized_text if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Streamlit UI
st.title("Email Spam Classifier")

# Text Input
email_input = st.text_area("Enter Email Content", height=150)

# Prediction Button
if st.button("Predict"):
    if email_input.strip() == "":
        st.error("⚠️ Please enter an email to classify.")
    else:
        # Preprocess input
        clean_email = preprocess_text(email_input)

        # Transform using the trained vectorizer
        email_tfidf = vectorizer.transform([clean_email])  # **Use transform(), not fit_transform()**

        # Predict using RandomForest
        prediction = model.predict(email_tfidf)[0]

        # Display Prediction
        if prediction == 1:
            st.error("❌ This email is **SPAM**! ")
        else:
            st.success("✅ This email is **NOT SPAM**!")

# # Testing with Sample Emails
# sample_non_spam = "Hey, your invoice for last month is attached."
# sample_spam = "Congratulations! You won a free iPhone. Click here now!"

# st.subheader("🔹 Sample Test Cases")
# for email in [sample_non_spam, sample_spam]:
#     processed = preprocess_text(email)
#     email_tfidf = vectorizer.transform([processed])
#     pred = model.predict(email_tfidf)[0]
#     st.write(f"📩 **Email:** {email} → **Prediction:** {'Spam' if pred == 1 else 'Not Spam'}")

