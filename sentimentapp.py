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
import plotly.express as px

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
        # 1. Displaying the prediction for users
        # Perform sentiment prediction
        transcript = file.read().decode("utf-8")
        # Split the transcript into individual messages
        messages = transcript.strip().split('\n')
        # Define the labels for emotions
        labels = ['happy', 'funny', 'relaxed', 'sad', 'angry', 'unhappy', 'sarcastic', 'abusive', 'irritated', 'demorosed', 'arrogant', 'agitated']
        # Initialize a dictionary to store emotions for each message
        message_emotions = {message: [] for message in messages}
        # Initialize a list to store emotions for each message
        emotions = []
        # Classify each message into emotions
        for message in messages:
            # Ensure the message is not empty
            if message.strip():
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

        # 2. Save the DataFrame to a CSV file for futher analysis
        # Split messages into new lines
        messages = [message.strip() for message in transcript.split('\n') if message.strip()]
        print(len(messages))
        # Regular expression pattern to extract usernames
        pattern = r'\[(\d+:\d+ [APM]{2})\] (\w+)(?: \[\w+ [A-Za-z\s]+\]:)?'
        # Find all matches in the text
        matches = re.findall(pattern, transcript)
        # Print the extracted usernames
        if matches:
            usernames = [match[1] for match in matches]
        # Initialize lists to store data
        times = []
        usernames = []
        # Process each match
        for match in matches:
            time = match[0]
            username = match[1]
            times.append(time)
            usernames.append(username)
        df = pd.DataFrame({'Time': times, 'Username': usernames, 'Message': messages})
        labels = ['happy', 'funny', 'relaxed', 'sad', 'angry', 'unhappy', 'sarcastic', 'abusive', 'irritated', 'demorosed', 'arrogant', 'agitated']
        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            message = row["Message"]            
            # Use zero-shot classification to classify the message
            result = classifier(message, labels)
            # Extract the label with the highest score
            predicted_label = result["labels"][0]
            # Update the corresponding label columns
            for label in labels:
                if label == predicted_label:
                    df.at[index, label] = 1
                else:
                    df.at[index, label] = 0
        # Display the dataframe
        st.info('Employee Analytics Prediction')
        st.dataframe(df)
        # Save the DataFrame to a CSV file for analytics
        df.to_csv('sentiment.csv', index=None)                  


if choice == "Analytics":
    st.title("Exploratory Data Analytics")
    # You can create a list of unique usernames
    unique_usernames = df['Username'].unique()
    chosen_target = st.selectbox('Choose employee for further analytics', unique_usernames)
    # Filter the DataFrame based on the selected employee
    selected_data = df[df['Username'] == chosen_target]
    # Create charts and analytics
    st.subheader(f'Analytics for {chosen_target}')
    # Generate two columns 
    col1, col2 = st.columns(2)
    # happy	funny	relaxed	sad	angry	unhappy	sarcastic	abusive	irritated	demorosed	arrogant	agitated
    with col1:
        # Calculate sentiment totals
        total_happy = selected_data['happy'].sum()
        total_funny = selected_data['funny'].sum()
        total_relaxed = selected_data['relaxed'].sum()
        total_sad = selected_data['sad'].sum()
        total_angry = selected_data['angry'].sum()
        total_unhappy = selected_data['unhappy'].sum()
        total_sarcastic = selected_data['sarcastic'].sum()
        total_abusive = selected_data['abusive'].sum()
        total_irritated = selected_data['irritated'].sum()
        total_demorosed = selected_data['demorosed'].sum()
        total_arrogant = selected_data['arrogant'].sum()
        total_agitated = selected_data['agitated'].sum()
        # Create a DataFrame for pie chart
        sentiment_data = pd.DataFrame({
            'Sentiment': ['Happy', 'Funny', 'Relaxed', 'Sad', 'Angry', 'Unhappy','Sarcastic', 'Abusive','Irritated', 'Demorosed', 'Arrogant', 'Agitated'],
            'Value': [total_happy, total_funny, total_relaxed, total_sad, total_angry, total_unhappy, total_sarcastic, total_abusive, total_irritated, total_demorosed, total_arrogant, total_agitated]
        })
        #st.info(f'Sentiment Analysis for {chosen_target}')
        st.info('Pie Chart Analysis as follows..')
         # Create a pie chart using Plotly Express
        fig = px.pie(sentiment_data, values='Value', names='Sentiment', color_discrete_sequence=px.colors.sequential.RdBu)
        # Set the width of the chart to ensure it fits within col1
        fig.update_layout(width=400)  # Adjust the width as needed
        # Display the chart in streamlit
        st.plotly_chart(fig)

    with col2:
        st.info('Heatmap of Sentiment Analysis')
        # Transpose the DataFrame for plotting
        transposed_data = selected_data[['happy', 'funny', 'relaxed', 'sad', 'angry', 'unhappy', 'sarcastic', 'abusive', 'irritated', 'demorosed', 'arrogant', 'agitated']].T
        transposed_data.columns = selected_data['Time']  # Set the time as column names
        # Create a heatmap for sentiment analysis using Plotly Express
        fig = px.imshow(
            transposed_data,
            labels=dict(x="Time", y="Sentiment"),
            color_continuous_scale='RdBu',  # You can adjust the color scale as needed
            zmin=0,
            zmax=1,
        )
        fig.update_xaxes(side="top")
        fig.update_layout(width=400, height = 500)  # Adjust the size as needed
        st.plotly_chart(fig)
    # Create a DataFrame for the emotional summary
    emotional_summary = pd.DataFrame({
        'Emotion': ['Happy', 'Funny', 'Relaxed', 'Sad', 'Angry', 'Unhappy','Sarcastic', 'Abusive','Irritated', 'Demorosed', 'Arrogant', 'Agitated'],
        'Emoji': ['😄', '😂', '😌', '😢', '😠', '😞', '😏', '😡', '😤', '😔', '😒', '😤'],
        'Value': [total_happy, total_funny, total_relaxed, total_sad, total_angry, total_unhappy, total_sarcastic, total_abusive, total_irritated, total_demorosed, total_arrogant, total_agitated]
    }).set_index('Emotion').T
    # Display the emotional summary in a DataFrame
    st.info(f'Emotional Summary of {chosen_target}')
    st.dataframe(emotional_summary)
    
    # Suggest actions based on sentiment
    if total_happy > total_sad:
        st.success(f"Suggestion: Encourage and reward {chosen_target} for their positivity!")
        if total_happy > total_angry:
            st.success(f"Additionally, it seems {chosen_target} is quite excited! Consider leveraging their enthusiasm.")
        elif total_happy < total_angry:
            st.warning(f"However, there are signs of anger. It's essential to address any concerns to maintain a positive atmosphere.")
        else:
            st.info(f"Moreover, {chosen_target} appears to have a balanced mix of excitement and anger. Keep an eye on the dynamics.")
    else:
        st.error(f"Suggestion: Check in with {chosen_target} to understand their concerns.")
        if total_happy > total_angry:
            st.info(f"It appears {chosen_target} is excited despite their concerns. Engage with them to address any issues.")
        elif total_happy < total_angry:
            st.error(f"Furthermore, there are signs of both anger and concern. Urgently address the issues to improve the situation.")
        else:
            st.warning(f"Moreover, {chosen_target} has a mixed sentiment of excitement and anger. Investigate the underlying reasons.")





