# Import UI dependencies
import streamlit as st
import os
# Bring in LLMs
from transformers import pipeline
# For creating dataframes
import pandas as pd
import re
# For visualizing charts
import matplotlib.pyplot as plt
import seaborn as sns

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

if os.path.exists('./sentiment_analytics.csv'): 
    df = pd.read_csv('sentiment_analytics.csv', index_col=None)

# Load the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

with st.sidebar:
    st.image('https://nexval.com/wp-content/uploads/2021/06/NEX_WEB_LOGO_NEXVAL.png')
    st.title("Chat Sentiment Analysis")
    choice = st.radio("Options", ["Prediction", "Analytics"])
    st.info("This project application helps to understand the users overall sentiment.")

if choice == "Prediction":
    st.title("Sentiment Prediction")
    file = st.file_uploader("Upload Your Dataset")

    if file:
        # Perform sentiment prediction
        transcript = file.read().decode("utf-8")

        # Split the transcript into individual messages
        messages = transcript.split('\n')

        # Define the labels for emotions
        labels = ["happy", "sad", "excited", "angry"]

        # Initialize a dictionary to store emotions for each message
        message_emotions = {message: [] for message in messages}

        # Initialize a list to store usernames
        usernames = []

        # Initialize a list to store emotions for each message
        emotions = []

        # Classify each message into emotions
        for message in messages:
            # Ensure the message is not empty
            if message.strip():
                # Extract username from the message using a regular expression
                username = re.search(r'\[([^\]]+)\]', message)
                if username:
                    usernames.append(username.group(1))
                for label in labels:
                    # Classify the message against the current label
                    result = classifier(message, [label])

                    # Extract the scores and labels
                    scores = result["scores"]
                    label_predictions = result["labels"]

                    # Iterate through the predictions
                    for score, label_prediction in zip(scores, label_predictions):
                        if score > 0.5:
                            message_emotions[message].append(label_prediction)

        # Create a chat summary with messages categorized by emotion
        chat_summary = {label: [] for label in labels}
        for message, emotions in message_emotions.items():
            for emotion in emotions:
                chat_summary[emotion].append(message)

        # Ensure all arrays have the same length
        max_length = max(len(messages) for messages in chat_summary.values())
        for label in labels:
            chat_summary[label] += [''] * (max_length - len(chat_summary[label]))

        # Create a Pandas DataFrame
        df = pd.DataFrame(chat_summary)

        # Display the results in a table
        st.info("Model Predictions")
        st.dataframe(df)

        # Save the DataFrame to a CSV file
        df.to_csv('sentiment_prediction_results.csv', index=None)  

        # Save the DataFrame to a CSV file for analytics
        analytics_df = pd.DataFrame({
            "Username": usernames,
            "Message": messages,
            "Happy": chat_summary["happy"],
            "Sad": chat_summary["sad"],
            "Excited": chat_summary["excited"],
            "Angry": chat_summary["angry"]
        })

        analytics_df.to_csv('analytics_results.csv', index=None)  # Save for analytics

        # Save the DataFrame to a CSV file
        df.to_csv('sentiment_prediction_results.csv', index=None)                  


if choice == "Analytics":
    st.title("Exploratory Data Analytics")
    # You can create a list of unique usernames
    unique_usernames = df['Username'].unique()
    chosen_target = st.selectbox('Choose employee for further analytics', unique_usernames)
    # Filter the DataFrame based on the selected employee
    selected_data = df[df['Username'] == choice]
    # Create charts and analytics
    st.info(f'Analytics for {chosen_target}')
    # Generate two columns 
    col1, col2 = st.columns(2)
    with col1:
        # Create a bar chart for sentiment analysis
        plt.figure(figsize=(8, 6))
        plt.bar(selected_data['Time'], selected_data['Happy'], label='Happy', color='yellow')
        plt.bar(selected_data['Time'], selected_data['Sad'], label='Sad', color='blue', bottom=selected_data['Happy'])
        plt.bar(selected_data['Time'], selected_data['Excited'], label='Excited', color='green', bottom=selected_data['Happy'] + selected_data['Sad'])
        plt.bar(selected_data['Time'], selected_data['Angry'], label='Angry', color='red', bottom=selected_data['Happy'] + selected_data['Sad'] + selected_data['Excited'])
        plt.xlabel('Time')
        plt.ylabel('Sentiment Value')
        plt.title('Sentiment Analysis Over Time')
        plt.legend()
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf())
    with col2:
        st.info('This is the total emotional activity of the employee')
        # Calculate and display an info line
        total_happy = selected_data['Happy'].sum()
        total_sad = selected_data['Sad'].sum()
        total_excited = selected_data['Excited'].sum()
        total_angry = selected_data['Angry'].sum()

        st.info('This is their choice')
        st.write(f"{choice} was:")
        st.write(f"- Happy: {total_happy}")
        st.write(f"- Sad: {total_sad}")
        st.write(f"- Excited: {total_excited}")
        st.write(f"- Angry: {total_angry}")

    # Suggest actions based on sentiment
    if total_happy > total_sad:
        st.success(f"Suggestion: Encourage and reward {chosen_target} for their positivity!")
    else:
        st.error(f"Suggestion: Check in with {chosen_target} to understand their concerns.")