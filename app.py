# Importing modules
import numpy as np
import streamlit as st
import cv2
import pandas as pd
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Load dataset
df = pd.read_csv(r"C:\Users\sanid\OneDrive\Desktop\project\muse_v3.csv")
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']

df['link'] = "https://www.youtube.com/results?search_query=" + df['track'].astype(str) + "+" + df['artist'].astype(str)

# Final selected columns
df = df[['name', 'emotional', 'pleasant', 'link', 'artist']]
df = df.sort_values(by=["emotional", "pleasant"]).reset_index(drop=True)

# Define emotion subsets
df_sad = df[:18000]
df_fear = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]

# Recommendation function
def fun(emotions_list):
    data = pd.DataFrame()
    t = 30  # number of songs to recommend

    def fallback():
        st.info("Showing general mood-based recommendations.")
        return df.sample(n=min(t, len(df)), replace=True)

    if not emotions_list:
        return fallback()

    v = emotions_list[0]

    try:
        if v == 'Neutral':
            data = df_neutral.sample(n=t, replace=True)
        elif v == 'Angry':
            data = df_angry.sample(n=t, replace=True)
        elif v == 'Fearful':
            data = df_fear.sample(n=t, replace=True)
        elif v == 'Happy':
            data = df_happy.sample(n=t, replace=True)
        elif v == 'Sad':
            data = df_sad.sample(n=t, replace=True)
        else:
            data = fallback()
    except:
        data = fallback()

    return data

def pre(emotion_list):
    result = []
    emotion_counts = Counter(emotion_list)
    for emotion, count in emotion_counts.items():
        result.extend([emotion] * count)
    unique_emotions = list(dict.fromkeys(result))  # Preserve order
    return unique_emotions

# Model setup
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.load_weights(r'C:\Users\sanid\OneDrive\Desktop\project\model.h5')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Streamlit UI setup
page_bg_img = '''
<style>
body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white'><b>Emotion-based Music Recommendation</b></h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'><b>Click on the name of recommended song to search on YouTube</b></h5>", unsafe_allow_html=True)

# Age input field
age = st.number_input("Enter your age:", min_value=10, max_value=100, value=25, step=1)
st.write(f"Selected Age: {age}")

# Emotion detection and recommendation trigger
emotion_list = []
cap = cv2.VideoCapture(0)

if st.button('SCAN EMOTION (Click here)'):
    count = 0
    emotion_list.clear()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5)
        count += 1

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img, verbose=0)
            max_index = int(np.argmax(prediction))
            emotion_list.append(emotion_dict[max_index])

            cv2.putText(frame, emotion_dict[max_index], (x + 20, y - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        st.image(frame, channels="BGR", caption="Emotion Detection in Progress")
        if count >= 20:
            break

    cap.release()
    emotion_list = pre(emotion_list)
    st.success("Emotions successfully detected.")
    st.write(f"Detected emotions: {emotion_list}")

    # Show recommendations after scanning
    new_df = fun(emotion_list)

    st.write("")
    st.markdown("<h5 style='text-align: center; color: grey;'><b>Recommended songs with artist names</b></h5>", unsafe_allow_html=True)
    st.write("---------------------------------------------------------------------------------------------------------------------")

    try:
        if new_df.empty:
            st.warning("No song recommendations available based on detected emotions.")
        else:
            for l, a, n, i in zip(new_df["link"], new_df['artist'], new_df['name'], range(len(new_df))):
                st.markdown(f"""<h4 style='text-align: center;'><a href="{l}" target="_blank">{i+1}. {n}</a></h4>""", unsafe_allow_html=True)
                st.markdown(f"<h5 style='text-align: center; color: grey;'><i>{a}</i></h5>", unsafe_allow_html=True)
                st.write("---------------------------------------------------------------------------------------------------------------------")
    except Exception as e:
        st.error(f"Error in displaying recommendations: {str(e)}")
