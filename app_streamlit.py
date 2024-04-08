import streamlit as st
import requests

# Define the API endpoint
API_URL = "http://localhost:5000/summarize"  # Replace with your actual API endpoint

# Streamlit UI
def main():
    st.title("Text Summarization")

    # Text input box for user input
    input_text = st.text_area("Enter the text to summarize:", height=200)

    # Button to trigger the summarization
    if st.button("Summarize"):
        # Make a POST request to the API
        response = requests.post(API_URL, json={"text": input_text})

        # Check if the request was successful
        if response.status_code == 200:
            # Display the generated summary
            summary = response.json()["summary"]
            st.subheader("Generated Summary:")
            st.write(summary)
        else:
            st.error("Failed to generate summary. Please try again.")

if __name__ == "__main__":
    main()