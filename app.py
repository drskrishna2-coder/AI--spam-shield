import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="AI Spam Shield")

st.title("AI Spam Shield")
st.write("Enter an SMS below to check whether it is Spam or Ham.")

df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)
x = df["message"]
y = df["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer()

x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

classifier = MultinomialNB()

classifier.fit(x_train_tfidf, y_train)

message = st.text_area("Enter your SMS")

if st.button("Check Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:

        new_message = vectorizer.transform([message])

        prediction = classifier.predict(new_message)

        probability = classifier.predict_proba(new_message)

        spam_probability = probability[0][
            list(classifier.classes_).index("spam")
        ] * 100

        if prediction[0] == "spam":

            st.error("This message is SPAM!")

            st.write(
                f"Spam Probability: **{spam_probability:.2f}%**"
            )

        else:

            st.success("This message is HAM (Not Spam).")

            st.write(
                f"Spam Probability: **{spam_probability:.2f}%**"
            )
