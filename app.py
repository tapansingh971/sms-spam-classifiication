import streamlit as st
import pickle

# Load saved vectorizer and model
cv = pickle.load(open('Vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# App title
st.title("📩 SMS Spam Detection App")

# Input from user
input_sms = st.text_input("✍️ Enter a message")

if st.button("Predict"):
    if input_sms.strip() != "":
        # Transform input
        vector_input = cv.transform([input_sms]).toarray()
        # Predict
        result = model.predict(vector_input)[0]

        if result == 1:   # assuming 1 = spam, 0 = ham
            st.error("🚨 This message is **SPAM**!")
        else:
            st.success("✅ This message is **NOT Spam**.")
    else:
        st.warning("⚠️ Please enter a message before predicting.")
