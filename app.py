from login import login_signup_page
import streamlit as st
from streamlit_js_eval import streamlit_js_eval
from streamlit_folium import st_folium
import folium
from datetime import datetime
from db import insert_report
import pandas as pd
import cv2
import numpy as np
import mediapipe as mp

from db import get_connection

# ------------------------- SYMPTOM DETECTION PAGES -------------------------- #

def disclaimer_page():
    st.title("Welcome to VISHioN")
    st.warning(
        "⚠️ This tool is for preliminary screening only. It does NOT replace professional medical diagnosis. "
        "Please consult a pediatrician for an accurate checkup."
    )
    if st.button("Next"):
        st.session_state.page = "instructions"

def instructions_page():
    st.title("Instructions for Taking Baby Photo")
    st.write("""
    Please follow these instructions carefully:
    - Use natural lighting, avoid shadows or colored light.
    - Capture the baby’s face, eyes, lips, hands, and feet clearly.
    - Avoid blurry or low-resolution photos.
    - Make sure the baby is calm and the photo is focused.
    """)
    if st.button("Next"):
        st.session_state.page = "color"

def skin_color_page():
    st.title("Observed Skin Color")
    skin_color_options = ["Yellow", "Blue", "Very Pale", "No discoloration"]
    selected_color = st.selectbox(
        "Select your baby's observed skin color:",
        ["-- Select --"] + skin_color_options,
        index=0
    )
    if selected_color != "-- Select --":
        st.session_state["skin_color"] = selected_color

    if st.button("Next"):
        if "skin_color" not in st.session_state:
            st.error("Please select a skin color to continue.")
        else:
            st.session_state.page = "questionnaire"

def symptoms_questionnaire_page():
    st.title("Symptom Questionnaire")

    if "skin_color" not in st.session_state or not st.session_state["skin_color"]:
        st.warning("Please select skin color first.")
        st.session_state.page = "color"
        st.stop()

    skin_color = st.session_state["skin_color"]

    urine_color = st.text_input("Urine color (e.g. normal, dark):", key="input_urine_color")
    stool_color = st.text_input("Stool color (e.g. normal, pale):", key="input_stool_color")

    if skin_color == "Yellow":
        st.subheader("Symptoms related to Jaundice")
        symptoms_labels = [
            "Yellow skin", "Yellow eyes", "Poor feeding", "Low energy levels", "Cries more"
        ]
    elif skin_color == "Blue":
        st.subheader("Symptoms related to Cyanosis")
        symptoms_labels = [
            "Blue skin", "Blue lips", "Blue tongue", "Tires easily", "Rapid breathing",
            "Poor feeding", "Becoming fussy", "Seizures"
        ]
    elif skin_color == "Very Pale":
        st.subheader("Symptoms related to Pallor")
        symptoms_labels = [
            "Pale skin", "Pale nails", "Poor feeding", "Low energy levels",
            "Difficulty in breathing", "Fainting", "Hands and feet are cold"
        ]
    else:
        st.subheader("Any unusual symptoms?")
        symptoms_labels = [
            "Poor feeding", "Low energy levels", "Rapid breathing", "Tires easily",
            "Becoming fussy", "Difficulty in breathing", "Fainting"
        ]

    prev_selected = st.session_state.get('symptoms', [])
    checked_symptoms = []
    for symptom in symptoms_labels:
        if st.checkbox(symptom, value=(symptom in prev_selected), key=f"checkbox_{symptom}"):
            checked_symptoms.append(symptom)

    if st.button("Next", key="questionnaire_next"):
        st.session_state['symptoms'] = checked_symptoms
        st.session_state.page = "upload"


# ------------------------- UPLOAD PAGE -------------------------- #
def enhance_image(region):
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced

def analyze_symptom(region, symptom):
    region = enhance_image(region)  # Enhance image before analysis
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    # rest of your existing code ...

    if symptom == "jaundice":
        lower, upper = np.array([20, 50, 80]), np.array([35, 255, 255])
    elif symptom == "cyanosis":
        lower, upper = np.array([85, 40, 40]), np.array([135, 255, 255])

    elif symptom == "pallor":
        lower, upper = np.array([0, 0, 200]), np.array([180, 60, 255])
    else:
        return 0
    mask = cv2.inRange(hsv, lower, upper)
    count = cv2.countNonZero(mask)
    total = region.shape[0] * region.shape[1]
    return (count / total) * 100 if total > 0 else 0

def extract_regions(image):
    mp_face = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    regions = {}

    with mp_face.FaceMesh(static_image_mode=True) as face_mesh:
        face_result = face_mesh.process(rgb)
        if face_result.multi_face_landmarks:
            face_landmarks = face_result.multi_face_landmarks[0]
            h, w, _ = image.shape
            face_points = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]
            x1, y1 = max(min(x for x, y in face_points) - 10, 0), max(min(y for x, y in face_points) - 10, 0)
            x2, y2 = min(max(x for x, y in face_points) + 10, w), min(max(y for x, y in face_points) + 10, h)
            face_region = image[y1:y2, x1:x2]
            if face_region.shape[0] > 50 and face_region.shape[1] > 50:
                regions["face"] = face_region
            else:
                st.warning("Face region too small or unclear. Please upload a clearer photo.")

    with mp_hands.Hands(static_image_mode=True) as hands:
        hand_result = hands.process(rgb)
        if hand_result.multi_hand_landmarks:
            h, w, _ = image.shape
            all_x = [int(pt.x * w) for hand in hand_result.multi_hand_landmarks for pt in hand.landmark]
            all_y = [int(pt.y * h) for hand in hand_result.multi_hand_landmarks for pt in hand.landmark]
            x1, y1 = max(min(all_x) - 10, 0), max(min(all_y) - 10, 0)
            x2, y2 = min(max(all_x) + 10, w), min(max(all_y) + 10, h)
            hand_region = image[y1:y2, x1:x2]
            if hand_region.shape[0] > 50 and hand_region.shape[1] > 50:
                regions["hands"] = hand_region
            else:
                st.warning("Hands region too small or unclear. Please upload a clearer photo.")

    if "face" not in regions:
        st.warning("Face not detected. Please upload a clear photo showing the face.")
    if "hands" not in regions:
        st.warning("Hands not detected. Please upload a clear photo showing the hands.")

    return regions

def combined_advice(skin_color, symptoms, image_scores):
    j_score = sum(s in {"Yellow skin", "Yellow eyes", "Poor feeding", "Low energy levels", "Cries more"} for s in symptoms)
    c_score = sum(s in {"Blue skin", "Blue lips", "Blue tongue", "Rapid breathing", "Tires easily"} for s in symptoms)

    if skin_color == "Blue" and c_score >= 2 and image_scores.get('cyanosis', 0) > 5:
        return "🔵 Cyanosis symptoms detected. Seek urgent care."
    p_score = sum(s in {"Pale skin", "Pale nails", "Cold hands", "Fainting", "Low energy levels"} for s in symptoms)

    if skin_color == "Yellow" and j_score >= 2 and image_scores.get('jaundice', 0) > 6:
        return "🟡 Jaundice likely. Consult a pediatrician."
    elif skin_color == "Blue" and c_score >= 2 and image_scores.get('cyanosis', 0) > 5:
        return "🔵 Cyanosis symptoms detected. Seek urgent care."
    elif skin_color == "Very Pale" and p_score >= 2 and image_scores.get('pallor', 0) > 7:
        return "⚪ Pallor symptoms suggest anemia. Doctor visit recommended."
    else:
        return "✅ No major signs detected. Monitor baby and consult if needed."

def upload_page():
    st.title("Upload Baby Photo")
    st.info("""
Please upload a clear photo of your baby:
- Use natural daylight, avoid shadows or colored lights.
- Show the face and hands clearly.
- Avoid blurry or low-resolution images.
- Ensure the baby is calm for better results.
""")

    uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

        regions = extract_regions(img)
        scores = {}
        if "face" in regions:
            scores["jaundice"] = analyze_symptom(regions["face"], "jaundice")
            scores["pallor"] = analyze_symptom(regions["face"], "pallor")
            scores["cyanosis"] = analyze_symptom(regions["face"], "cyanosis")

        if "hands" in regions:
            scores["cyanosis"] = analyze_symptom(regions["hands"], "cyanosis")

        st.subheader("🔬 Results:")
        for k, v in scores.items():
            st.write(f"{k.capitalize()}: {v:.2f}%")

        advice = combined_advice(
            st.session_state.get("skin_color", ""),
            st.session_state.get("symptoms", []),
            scores
        )
        st.info(advice)

        show_pediatrician_map()

import librosa
import joblib
import os

def extract_features_from_audio(audio_data, sr=22050):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def load_model():
    return joblib.load("cry_model.pkl")  # make sure this file is placed in the project root

def cry_analysis_page():
    st.title("🔊 Baby Cry Analysis")

    uploaded = st.file_uploader("Upload baby's cry audio file (.mp3/.wav)", type=["mp3", "wav"])
    
    if uploaded:
        st.audio(uploaded, format='audio/mp3')

        try:
            y, sr = librosa.load(uploaded, sr=22050)
            features = extract_features_from_audio(y, sr).reshape(1, -1)

            model = load_model()
            prediction = model.predict(features)[0]

            label_map = {
                0: "Belly Pain",
                1: "Burp",
                2: "Discomfort",
                3: "Tired"
            }

            st.success(f"🎧 Predicted Cry Type: **{label_map.get(prediction, 'Unknown')}**")

        except Exception as e:
            st.error(f"❌ Error analyzing audio: {e}")


def show_pediatrician_map():
    st.subheader("📍 Nearby Pediatricians")
    loc = streamlit_js_eval(js_expressions="navigator.geolocation.getCurrentPosition", key="locate")
    if loc and "coords" in loc:
        lat, lon = loc["coords"]["latitude"], loc["coords"]["longitude"]
        m = folium.Map(location=[lat, lon], zoom_start=14)
        folium.Marker([lat, lon], tooltip="You are here", icon=folium.Icon(color="blue")).add_to(m)
        st_folium(m, width=700)
        st.info("You can manually search 'Pediatricians near me' on Google.")

# ------------------------- BABY HISTORY MODULE -------------------------- #

def load_reports_with_files(username):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, report_date, report_name, description, file FROM baby_reports WHERE username = %s", (username,))
    rows = cursor.fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=["ID", "Date", "Report Name", "Description", "File"])

def baby_history_page():
    st.title("👶 Baby History Page")
    reports = load_reports_with_files(st.session_state["username"])
    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input("Search by date or name")
    with col2:
        if st.button("➕ Add Report"):
            st.session_state.page = "add_report"

    if search:
        reports = reports[
            reports["Date"].astype(str).str.contains(search) |
            reports["Report Name"].str.contains(search, case=False)
        ]

    if not reports.empty:
        for _, row in reports.iterrows():
            st.markdown(f"**📅 {row['Date']}** - **{row['Report Name']}**")
            st.write(row["Description"])
            if row["File"]:
                st.download_button(
                    "📥 Download Attachment",
                    row["File"],
                    file_name=f"{row['Report Name'].replace(' ', '_')}.pdf"
                )
            st.markdown("---")
    else:
        st.info("No reports found.")

def add_report_page():
    st.title("📄 Add Baby Report")
    report_date = st.date_input("Date", value=datetime.today())
    report_name = st.text_input("Report Title")
    report_description = st.text_area("Report Description")
    uploaded_file = st.file_uploader("Attach File", type=["pdf", "jpg", "jpeg", "png"])

    if st.button("Save Report"):
        file_data = uploaded_file.read() if uploaded_file else None
        insert_report(
    username=st.session_state["username"],
    report_date=report_date.strftime("%Y-%m-%d"),
    title=report_name,
    description=report_description,
    file_blob=file_data
)

        st.success("✅ Report saved!")
        st.session_state.page = "baby_history"


import joblib
import librosa

def extract_features_from_audio(file):
    y, sr = librosa.load(file, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled.reshape(1, -1)

def baby_cry_analysis_page():
    st.title("Baby Cry Analysis 🎵")
    uploaded_audio = st.file_uploader("Upload Baby Cry MP3/WAV Audio", type=["mp3", "wav"])
    if uploaded_audio:
        st.audio(uploaded_audio, format='audio/mp3')

        # Load saved model and label encoder
        clf = joblib.load("baby_cry_classifier.joblib")
        le = joblib.load("label_encoder.joblib")

        features = extract_features_from_audio(uploaded_audio)
        prediction = clf.predict(features)
        predicted_label = le.inverse_transform(prediction)[0]

        st.success(f"Predicted Cry Type: {predicted_label}")


# ------------------------- MAIN NAVIGATION -------------------------- #

def show_options_page():
    st.title(f"Welcome, {st.session_state.get('username', '')} 👋")
    if st.button("🧪 Run Symptom Detection"):
        st.session_state.page = "detection"
    if st.button("📁 View Baby History"):
        st.session_state.page = "baby_history"
    if st.button("🔊 Analyze Baby Cry"):
     st.session_state.page = "cry_analysis"


def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login_signup_page()
        return

    if "page" not in st.session_state:
        st.session_state.page = "options"

    page = st.session_state.page
    if page == "options": show_options_page()
    elif page == "detection": disclaimer_page()
    elif page == "instructions": instructions_page()
    elif page == "color": skin_color_page()
    elif page == "questionnaire": symptoms_questionnaire_page()
    elif page == "upload": upload_page()
    elif page == "baby_history": baby_history_page()
    elif page == "add_report": add_report_page()
    elif page == "cry_analysis": cry_analysis_page()


if __name__ == "__main__":
    main()
