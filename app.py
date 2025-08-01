import streamlit as st
import time
import cv2
import os
import numpy as np
import librosa
import random
import speech_recognition as sr
from pydub import AudioSegment
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from gtts import gTTS
from deepface import DeepFace
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from dotenv import load_dotenv
from pydub.playback import play

# Load environment variables
load_dotenv()

# Streamlit config
st.set_page_config(page_title="AI-Assistant", page_icon=":robot:")

# Create audio folders
os.makedirs("../PythonP-3/audio/human", exist_ok=True)
os.makedirs("../PythonP-3/audio/AI", exist_ok=True)

# Initialize recognizer
recognizer = sr.Recognizer()

# Load SavedModel using Option 2 (TF format)
try:
    model = load_model("emotion_detection_model.h5")
    # _ = model.predict(np.zeros((1, 193, 1)))  # Dummy call to validate .predict
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# LLM Init
llm = ChatGroq(model_name="llama-3.3-70b-versatile")

# Session state defaults
if "sessionMessages" not in st.session_state:
    st.session_state.sessionMessages = [{
        "role": "assistant",
        "content": "Hi, I’m Anaya, your smart and polite conversational AI voice assistant created by Shivam Sharma...and my responses should be in 20 to 30 words"
    }]

if "button" not in st.session_state:
    st.session_state.button = "START"

if "face_data" not in st.session_state:
    st.session_state.face_data = {"emotion": "neutral", "gender": "unknown", "age": 25, "race": "other"}

# Detect face once
def detect_face_once():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        try:
            result = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False)
            data = result[0]
            st.session_state.face_data = {
                "emotion": data['dominant_emotion'],
                "gender": data['dominant_gender'],
                "age": data['age'],
                "race": data['dominant_race']
            }
        except Exception as e:
            st.warning(f"Face detection error: {e}")
    cap.release()
    cv2.destroyAllWindows()

# Audio feature extraction
def extract_features(data, sample_rate=22050):
    result = np.array([])
    result = np.hstack((result, np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)))
    stft = np.abs(librosa.stft(data))
    result = np.hstack((result, np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)))
    result = np.hstack((result, np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)))
    result = np.hstack((result, np.mean(librosa.feature.rms(y=data).T, axis=0)))
    result = np.hstack((result, np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)))
    return result

# Multimodal Emotion Analysis
def speech_mood(audio):
    with open("output.wav", "wb") as f:
        f.write(audio.get_wav_data())

    AudioSegment.from_wav("output.wav").export("audio/human/output.mp3", format="mp3")
    y, sr_ = librosa.load("audio/human/output.mp3")
    features = extract_features(y, sr_)
    reshaped = np.expand_dims(np.expand_dims(features, axis=0), axis=2)
    prediction = model.predict(reshaped)
    emotion_labels = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    encoder = LabelEncoder()
    encoder.fit(emotion_labels)
    voice_emotion = encoder.classes_[np.argmax(prediction)]
    return voice_emotion

# Audio Playback
def play_audio(file_path):
    if not os.path.exists(file_path):
        st.warning(f"Audio file not found: {file_path}")
        return
    sound = AudioSegment.from_file(file_path, format="mp3")
    play(sound)

def sound(response):
    try:
        tts = gTTS(response, lang='en')
        tts.save("audio/AI/ai.mp3")
        play_audio("audio/AI/ai.mp3")
    except Exception as e:
        st.error(f"TTS failed: {e}")

# Voice capture
def recording():
    x = random.randint(0,100)
    if x % 7 == 0:
        detect_face_once()

    with sr.Microphone() as source:
        play_audio("../PythonP-3/audio/system/in.mp3")
        msg_placeholder.markdown("🎙️ Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=4)

        while True:
            try:
                audio = recognizer.listen(source, timeout=2, phrase_time_limit=30)
                msg_placeholder.markdown("🎧 Processing...")
                result_google = recognizer.recognize_google(audio)
                st.sidebar.write(":bust_in_silhouette:", result_google)
                llm_response(result_google, audio)
                break
            except Exception as e:
                sound("Sorry, I didn't get that.")
                msg_placeholder.markdown("❌ Didn't catch that. Retrying...")
                time.sleep(2)

# LLM Emotional response
def llm_response(question, audio):
    if any(k in question.lower() for k in ["stop", "bye", "exit"]):
        st.session_state.button = "START"
        msg_placeholder.markdown("stopped...")
        return "Anaya stopped listening"

    face_data = st.session_state.face_data
    face_emotion = face_data["emotion"]
    gender = face_data["gender"]
    age = face_data["age"]
    race = face_data["race"]

    speech_emotion = speech_mood(audio)

    st.write("Facial expression recognition results:", face_emotion, "     speech_emotion: ", speech_emotion)

    prompt = (
        f"Given the user's gender is '{gender}', age is '{age}', race is '{race}', "
        f"facial emotion is '{face_emotion}', and voice emotion is '{speech_emotion}', "
        "respond as an emotionally intelligent AI assistant by speaking in a culturally respectful "
        "and age-appropriate manner, acknowledging their feelings and offering empathetic, comforting, "
        "or uplifting words accordingly."
    )

    st.session_state.sessionMessages.append({
        "role": "system",
        "content": prompt
    })
    st.session_state.sessionMessages.append({"role": "user", "content": question})

    response = llm.invoke(st.session_state.sessionMessages)
    st.session_state.sessionMessages.append({"role": "assistant", "content": response.content})
    st.sidebar.write("🤖", response.content)
    sound(response.content)
    recording()


# --------------------
# Streamlit UI Layout
# --------------------

placeholder = st.empty()
msg_placeholder = st.empty()

st.markdown("<h1 style='text-align: center;'>Hey, I'm Your AI Assistant</h1>", unsafe_allow_html=True)

detect_face_once()
# Display Anaya animation
st.markdown("""
<div style="display: flex; justify-content: center;">
    <img src="https://i.pinimg.com/originals/0b/1b/ff/0b1bff36918c2e231d1a980b2c4c3cef.gif" width="400">
</div>
""", unsafe_allow_html=True)

# Voice interaction
if st.sidebar.button("🎤 Start Listening"):
    recording()
