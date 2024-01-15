import streamlit as st
import pickle

# Load your trained models
cv = pickle.load(open("models/cv.pkl", 'rb'))
clf = pickle.load(open("models/clf.pkl", 'rb'))

# Streamlit application
def main():
    st.title("Email Spam Classifier")

    # Custom CSS for the layout
    st.markdown("""
        <style>
            .column {
                padding: 10px;
                display: inline;
                text-align: center;
                float: left;
                height: auto;
            }
            .center {
                text-align: center;
            }
            .border {
                border-right: solid black;
            }
        </style>
        """, unsafe_allow_html=True)

    # Creating two columns
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("### SAMPLE EMAILS")
        st.markdown("<h4 style='color: red;'>SPAM</h4>", unsafe_allow_html=True)
        st.markdown("Your spam sample text here...")
        st.markdown("<h4 style='color: green;'>NOT SPAM</h4>", unsafe_allow_html=True)
        st.markdown("Your not spam sample text here...")

    with col2:
        email = st.text_area("Type or paste your email here", height=300)
        if st.button("Check"):
            # Perform prediction
            X = cv.transform([email])
            prediction = clf.predict(X)
            prediction = 1 if prediction == 1 else -1

            # Display results
            if prediction == 1:
                st.error("Spam")
            else:
                st.success("Not Spam")

if __name__ == '__main__':
    main()
