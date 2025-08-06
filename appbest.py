from login import login_signup_page

import streamlit as st
from streamlit_js_eval import streamlit_js_eval
from streamlit_folium import st_folium
import folium
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
import mediapipe as mp

DATA_FILE = "baby_reports.csv"

# ----------- Pages and Functions -----------

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
    if "skin_color" not in st.session_state:
        st.session_state["skin_color"] = None

    selected_color = st.selectbox(
        "Select your baby's observed skin color:",
        ["-- Select --"] + skin_color_options,
        index=0,
        key="skin_color_selectbox"
    )
    if selected_color != "-- Select --":
        st.session_state["skin_color"] = selected_color

    if st.button("Next"):
        if st.session_state["skin_color"] is None:
            st.error("Please select a skin color to continue.")
        else:
            st.success(f"Skin color selected: {st.session_state['skin_color']}")
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

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

def detect_baby_parts(image):
    results = {}
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb_image)
        if face_results.multi_face_landmarks:
            results['face'] = True
            results['eyes'] = True
            results['lips'] = True
        else:
            results['face'] = False
            results['eyes'] = False
            results['lips'] = False

    with mp_hands.Hands(static_image_mode=True) as hands:
        hand_results = hands.process(rgb_image)
        if hand_results.multi_hand_landmarks:
            results['hands'] = True
        else:
            results['hands'] = False

    return results

def extract_regions(image):
    regions = {}
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        face_result = face_mesh.process(rgb)
        if face_result.multi_face_landmarks:
            face_landmarks = face_result.multi_face_landmarks[0]
            h, w, _ = image.shape
            face_points = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]
            x_coords, y_coords = zip(*face_points)
            x1, y1, x2, y2 = max(min(x_coords)-10, 0), max(min(y_coords)-10, 0), min(max(x_coords)+10, w), min(max(y_coords)+10, h)
            regions["face"] = image[y1:y2, x1:x2]
            lip_ids = list(range(61, 291))
            lip_points = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in lip_ids]
            lx, ly = zip(*lip_points)
            lx1, ly1, lx2, ly2 = max(min(lx)-10, 0), max(min(ly)-10, 0), min(max(lx)+10, w), min(max(ly)+10, h)
            regions["lips"] = image[ly1:ly2, lx1:lx2]
    with mp_hands.Hands(static_image_mode=True) as hands:
        hand_result = hands.process(rgb)
        if hand_result.multi_hand_landmarks:
            h, w, _ = image.shape
            all_x, all_y = [], []
            for hand_landmarks in hand_result.multi_hand_landmarks:
                for pt in hand_landmarks.landmark:
                    all_x.append(int(pt.x * w))
                    all_y.append(int(pt.y * h))
            hx1, hy1 = max(min(all_x)-10, 0), max(min(all_y)-10, 0)
            hx2, hy2 = min(max(all_x)+10, w), min(max(all_y)+10, h)
            regions["hands"] = image[hy1:hy2, hx1:hx2]
    return regions

def analyze_symptom(region, symptom):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    if symptom == "jaundice":
        lower = np.array([20, 50, 80])
        upper = np.array([35, 255, 255])
    elif symptom == "cyanosis":
        lower = np.array([90, 50, 50])
        upper = np.array([130, 255, 255])
    elif symptom == "pallor":
        lower = np.array([0, 0, 200])
        upper = np.array([180, 60, 255])
    else:
        return 0
    mask = cv2.inRange(hsv, lower, upper)
    count = cv2.countNonZero(mask)
    total = region.shape[0] * region.shape[1]
    return (count / total) * 100 if total > 0 else 0

def combined_advice(skin_color, symptoms, image_results):
    msg = ""
    jaundice_symptoms = {"Yellow skin", "Yellow eyes", "Poor feeding", "Low energy levels", "Cries more"}
    cyanosis_symptoms = {"Blue skin", "Blue lips", "Blue tongue", "Rapid breathing", "Tires easily"}
    pallor_symptoms = {"Pale skin", "Pale nails", "Cold hands", "Fainting", "Low energy levels"}
    j_score = sum(1 for s in symptoms if s in jaundice_symptoms)
    c_score = sum(1 for s in symptoms if s in cyanosis_symptoms)
    p_score = sum(1 for s in symptoms if s in pallor_symptoms)
    jaundice_detected = skin_color == "Yellow" and j_score >= 2 and image_results.get('jaundice', 0) > 6
    cyanosis_detected = skin_color == "Blue" and c_score >= 2 and image_results.get('cyanosis', 0) > 5
    pallor_detected = skin_color == "Very Pale" and p_score >= 2 and image_results.get('pallor', 0) > 7

    if jaundice_detected:
        if j_score >= 4 or image_results.get('jaundice', 0) > 10:
            msg = "⚠️ Severe Jaundice likely. Immediate pediatric consultation advised."
        else:
            msg = "🟡 Mild to moderate jaundice signs. Please consult a pediatrician soon."
    elif cyanosis_detected:
        msg = "🔵 Cyanosis signs detected. Seek urgent medical attention."
    elif pallor_detected:
        msg = "⚪ Pallor symptoms observed. Could be anemia. Consult a doctor."
    elif skin_color == "No discoloration" and len(symptoms) > 3:
        msg = "🩺 Multiple symptoms noted without discoloration. Monitor and consult pediatrician."
    else:
        msg = "✅ No significant symptoms found. Monitor baby and consult if concerned."
    return msg

def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var

def upload_page():
    st.title("Upload Baby Photo")
    uploaded_file = st.file_uploader("Upload a clear photo of your baby", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Baby Photo", use_column_width=True)
        blurry, variance = is_blurry(img)
        st.write(f"Image Sharpness (Laplacian Variance): {variance:.2f}")
        if blurry:
            st.error("⚠️ The image is blurry. Please upload a clearer photo.")
            return

        parts = detect_baby_parts(img)
        regions = extract_regions(img)

        if not regions:
            st.error("❌ Could not extract required regions for analysis.")
            return

        st.subheader("🧪 Symptom Analysis:")
        if "face" in regions:
            jaundice_score = analyze_symptom(regions["face"], "jaundice")
            pallor_score = analyze_symptom(regions["face"], "pallor")
            st.write(f"🟡 Jaundice (face): {jaundice_score:.2f}%")
            st.write(f"⚪ Pallor (face): {pallor_score:.2f}%")
            if jaundice_score > 6:
                st.warning("⚠️ Possible signs of jaundice. Consult pediatrician.")
            if pallor_score > 7:
                st.warning("⚠️ Possible pallor. Baby may be anemic.")

        if "lips" in regions:
            cyanosis_score = analyze_symptom(regions["lips"], "cyanosis")
            st.write(f"🔵 Cyanosis (lips): {cyanosis_score:.2f}%")
            if cyanosis_score > 5:
                st.warning("⚠️ Possible cyanosis (low oxygen levels).")

        if "hands" in regions:
            pallor_hands = analyze_symptom(regions["hands"], "pallor")
            cyanosis_hands = analyze_symptom(regions["hands"], "cyanosis")
            st.write(f"⚪ Pallor (hands): {pallor_hands:.2f}%")
            st.write(f"🔵 Cyanosis (hands): {cyanosis_hands:.2f}%")

        image_results = {
            'jaundice': jaundice_score,
            'cyanosis': cyanosis_score,
            'pallor': pallor_score
        }

        advice = combined_advice(
            st.session_state.get('skin_color', ''),
            st.session_state.get('symptoms', []),
            image_results
        )
        st.info(f"🔍 Final Diagnosis:\n\n{advice}")

        missing_parts = [part for part, present in parts.items() if not present]
        if missing_parts:
            st.error(f"⚠️ Could not detect: {', '.join(missing_parts)}. Please upload a photo showing these clearly.")
            return
        st.success("✅ All required baby parts detected!")

    show_pediatrician_map()

def show_pediatrician_map():
    st.subheader("📍 Nearby Pediatricians")
    loc = streamlit_js_eval(js_expressions="navigator.geolocation.getCurrentPosition", key="locate")
    if not loc or "coords" not in loc:
        st.warning("📌 Location not available yet. Allow browser location access.")
        return

    lat = loc["coords"]["latitude"]
    lon = loc["coords"]["longitude"]
    st.success(f"Your Location: {lat:.4f}, {lon:.4f}")

    m = folium.Map(location=[lat, lon], zoom_start=14)
    folium.Marker([lat, lon], tooltip="You are here", icon=folium.Icon(color="blue")).add_to(m)
    st_folium(m, width=700)
    st.info("🩺 This map shows your location. Pediatricians nearby can be searched manually on Google.")
def show_options_page():
    st.title(f"Welcome, {st.session_state.get('username', '')} 👋")
    st.write("Choose an option:")

    if st.button("🧪 Run Symptom Detection"):
        st.session_state.page = "detection"

    if st.button("📁 View Baby History"):
        st.session_state.page = "baby_history"

def load_reports():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        return pd.DataFrame(columns=["Date", "Report Name", "Description"])

def save_reports(df):
    df.to_csv(DATA_FILE, index=False)

from db import insert_report, fetch_reports

from db import get_reports
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
        st.subheader("🗂️ Reports")
        for _, row in reports.iterrows():
            st.markdown(f"**📅 {row['Date']}** - **{row['Report Name']}**")
            st.write(row['Description'])

            if row["File"]:
                st.download_button(
                    label="📥 Download Attachment",
                    data=row["File"],
                    file_name=f"{row['Report Name'].replace(' ', '_')}.pdf" if row['File'][0:4] == b"%PDF" else "report_image.jpg"
                )

            st.markdown("---")
    else:
        st.info("No reports found.")

from db import add_report
def add_report_page():
    st.title("📄 Add Baby Report")

    report_date = st.date_input("Date of Report", value=datetime.today())
    report_name = st.text_input("Report Title")
    report_description = st.text_area("Report Description")
    uploaded_file = st.file_uploader("Attach report file (PDF/Image)", type=["pdf", "jpg", "jpeg", "png"])

    if st.button("Save Report"):
        file_data = None
        if uploaded_file is not None:
            file_data = uploaded_file.read()

        # Save to MySQL using db.py helper
        from db import insert_report
        insert_report(
            username=st.session_state["username"],
            date=report_date.strftime("%Y-%m-%d"),
            title=report_name,
            description=report_description,
            file_blob=file_data
        )

        st.success("✅ Report saved successfully!")
        st.session_state.page = "baby_history"
from db import get_connection

def load_reports_with_files(username):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, date, report_name, description, file
        FROM baby_reports
        WHERE username = %s
    """, (username,))
    rows = cursor.fetchall()
    conn.close()

    # Create dataframe
    df = pd.DataFrame(rows, columns=["ID", "Date", "Report Name", "Description", "File"])
    return df

def show_baby_data_page():
    st.title("Manage Baby Data (Coming Soon)")
    st.info("This feature is under development.")
    if st.button("Back to Options"):
        st.session_state.page = "options"

# ----------- Main Function -----------

def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login_signup_page()
        return

    if "page" not in st.session_state:
        st.session_state.page = "options"

    if st.session_state.page == "options":
        show_options_page()
    elif st.session_state.page == "detection":
        disclaimer_page()
    elif st.session_state.page == "instructions":
        instructions_page()
    elif st.session_state.page == "color":
        skin_color_page()
    elif st.session_state.page == "questionnaire":
        symptoms_questionnaire_page()
    elif st.session_state.page == "upload":
        upload_page()
    elif st.session_state.page == "baby_history":
        baby_history_page()
    elif st.session_state.page == "add_report":
        add_report_page()
    elif st.session_state.page == "data":
        show_baby_data_page()
    else:
        st.error("Unknown page! Resetting to options.")
        st.session_state.page = "options"

if __name__ == "__main__":
    main()
