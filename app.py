import streamlit as st
from gtts import gTTS
from playsound import playsound
import speech_recognition as sr
from deepface import DeepFace
import cv2
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from pydub import AudioSegment
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

model = load_model("speech_emotion_recognition_model.keras")
recognizer = sr.Recognizer()
llm = ChatGroq(model_name="llama-3.3-70b-versatile")

if "sessionMessages" not in st.session_state:
    st.session_state.sessionMessages = [{"role": "assistant", "content": "Hi, I’m Anaya, your smart and polite conversational AI voice assistant created by Shivam Sharma, a passionate Python developer and AI enthusiast. I am designed to talk like a friendly Indian girl, using a warm, respectful, and casual tone. I communicate in a natural, human-like way and keep my replies short, clear, and helpful, usually between thirty to forty words. I avoid overly technical or robotic language and focus on making conversations easy to understand, precise, and engaging. My goal is to assist you politely and efficiently, just like a thoughtful and well-mannered Indian companion"}]
if "button" not in st.session_state:
    st.session_state.button= "START"
if "emotion_done" not in st.session_state:
    st.session_state.emotion_done = False

def llm_response(question,audio):
    if any(keyword in question.lower() for keyword in ["song", "play", "gana", "gaana"]):
        st.session_state.button = "START"
        return "Anaya stopped listening"

    elif not st.session_state.emotion_done:
       speech_emotion = speech_mood(audio=audio, face_emotion=emotion, transcribed_text= question)
       st.session_state.sessionMessages.append({"role": "system", "content": f"You are Anaya, a kind, intelligent, and emotionally aware AI assistant.You just received an emotional analysis of the user based on their voice, face, and spoken text.Here is the emotional summary: {speech_emotion}Respond like a human friend who cares.Acknowledge the user's current emotion and mood.Offer supportive, empathetic, or cheerful replies based on their mental state.Optionally, suggest a short activity, motivational line, or music if the user seems sad or stressed.Keep your response under 4 sentences, natural and heartwarming.Important: Speak casually but respectfully. Do not repeat the input values directly unless needed for empathy. Prioritize comfort and connection."})

    else:
        st.session_state.sessionMessages.append({"role": "user", "content": question})
        response = llm.invoke(st.session_state.sessionMessages)
        st.session_state.sessionMessages.append({"role": "assistant", "content": response.content})
        st.sidebar.write("🤖 ",response.content)
        sound(response.content)
    recording()
    return response.content

def sound(response):
    tts = gTTS(response, lang='en')
    tts.save(r"audio/AI/ai.mp3")
    playsound("audio/AI/ai.mp3")

def recording():
    with sr.Microphone() as source:
        playsound("audio/system/in.mp3")
        msg_placeholder.markdown("listening...")

        recognizer.adjust_for_ambient_noise(source, duration=2)
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=30)
        msg_placeholder.markdown("listening stoped!!!")
        try:
            result_google = recognizer.recognize_google(audio)
            st.sidebar.write(":bust_in_silhouette: ", result_google)
            llm_response(result_google,audio)

        except Exception as e:
            sound("Sorry, I didn't get that")
            recording()


def face_detect():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    try:
        # Analyze the frame
        result = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion', 'race'])

        # Extract dominant emotion
        emotion = result[0]['dominant_emotion']
        gender = result[0]['dominant_gender']
        age = result[0]['age']
        race = result[0]['dominant_race']

        st.markdown(f"<p style='text-align: right;'> Emotion:  {emotion}  </p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: right;'> Gender: {gender}  </p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: right;'>  Age: {age}   </p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: right;'> Race:  {race} </p>", unsafe_allow_html=True)

    except Exception as e:
        print(f"Error: {e}")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    return emotion, gender, age, race

def extract_features(data, sample_rate=22050):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr))

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    return result

def  speech_mood(audio, face_emotion , transcribed_text):
    with open("output.wav", "wb") as f:
        f.write(audio.get_wav_data())

    sound = AudioSegment.from_wav("output.wav")
    sound.export("audio/human/output.mp3", format="mp3")

    emotion_categories = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    encoder = LabelEncoder()
    encoder.fit(emotion_categories)
    audio_path = "audio/human/output.mp3"
    y, sr = librosa.load(audio_path)
    sample_rate = sr
    features = extract_features(y, sample_rate=sample_rate)
    reshaped_features = np.expand_dims(np.expand_dims(features, axis=0), axis=2)
    prediction = model.predict(reshaped_features)
    predicted_emotion_index = np.argmax(prediction)
    emotion_categories = encoder.classes_
    predicted_emotion = emotion_categories[predicted_emotion_index]


    prompt = PromptTemplate(
        input_variables=["sentence", "prediction"],
        template="""
You are an expert in multimodal emotion analysis. 

You are given three inputs for a user:

1. **Voice-based Emotion Prediction** (from an audio model)
2. **Facial Emotion Detection** (from a facial expression model)
3. **Transcribed Text** (user's spoken sentence)

Your job is to deeply analyze all three inputs and determine:
- The most likely **Emotion** (one word)
- The overall **Mood** of the user (one word)
- The user's **Current Feeling** or mental state (one short sentence)
- The user's likely **Facial Emotion** (one word like: smiling, frowning, neutral, etc.)

These models might be partially incorrect. You must prioritize human-like interpretation, cross-verify the predictions with the spoken text, and resolve any contradictions among the sources.

### Input:
- **Voice Model Prediction**: {voice_emotion}
- **Facial Emotion Prediction**: {facial_emotion}
- **Spoken Text**: {transcribed_text}

### Output:
Emotion: <one word>  
Mood: <one word>  
Current Feeling: <one short sentence>  
Facial Emotion: <one word>


    """
    )

    emotion_chain = LLMChain(llm=llm, prompt=prompt)

    result = emotion_chain.invoke({
        "transcribed_text": transcribed_text,
        "voice_emotion": predicted_emotion
    })

    return result["text"]


st.set_page_config(page_title="AI Chatbot", page_icon=":robot:")

placeholder = st.empty()

st.markdown(
    """
    <h1 style='text-align: center; font-size: 40px;'>
        Hey, I'm Anaya Your AI Assistant
    </h1>
    """,
    unsafe_allow_html=True
)

emotion, gender, age, race = face_detect()

if age >= 0 and age <= 2:
    age_ = "a Baby"
elif age >= 3 and age <= 12:
    age_ = "a Child"
elif age >= 13 and age <= 19:
    age_ = "a Teenager"
elif age >= 20 and age <= 35:
    age_ = "a Young Adult"
elif age >= 36 and age <= 64:
    age_ = "a Middle-aged Adult"
elif age >= 65:
    age_ = "a Senior"
else:
    age_ = "Invalid age"


st.markdown(f"<p style='text-align: right;'>Emotion : {emotion}</p>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: right;'>Gender : {gender}</p>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: right;'>Region : {race}</p>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: right;'> You are {age_}</p>", unsafe_allow_html=True)

st.markdown(f"""
    <div style="display: flex; justify-content: center;">
        <img src="https://mir-s3-cdn-cf.behance.net/project_modules/hd/524820111444627.6001ca53344af.gif" alt="GIF" width="400">
    </div>
""", unsafe_allow_html=True)


side_button = st.sidebar.button(st.session_state.button)
msg_placeholder = st.empty()

if st.session_state.button == "START":
    st.session_state.button = "STOP"

else :
    st.session_state.button = "START"
    recording()
