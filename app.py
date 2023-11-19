import streamlit as st
import pickle
import string
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)



tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


st.set_page_config(
    page_title="SMS Spam Classifier",
    menu_items={
        'Report a bug': 'https://github.com/VidhiyaSB',
        'Get help': 'https://github.com/VidhiyaSB',
        'About': "SMS and Email Spam Classifier \n python -m streamlit run app.py  \n\n:)"
    }
)



st.title("SMS/Email Spam Detector")
logo = "logo.jpeg"

# Load the logo and set its width to 200 pixels
st.image(logo, width=200)
st.subheader('Check if your Message is spam or not before you proceed')
with st.sidebar:
    st.subheader("About the app")
    st.write("This SMS / Mail Spam detector checks if the message you received is Spam or Not Spam saving your time and money. ")
    st.subheader("Curators")
    st.write("Vidhiya S B ,Vaishnavi C and Vaisharli S")
    st.subheader("Contact")
    st.write("www.twitter.com/VidhiyaSB")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.header("Spam")       
    else:
        st.header("Not Spam")
        st.write("You can carry this conversation forward")