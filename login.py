import streamlit as st
import json
import os

USER_DB = "users.json"

def load_users():
    if not os.path.exists(USER_DB):
        return {}
    with open(USER_DB, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f)

def login_signup_page():
    st.title("👶 VISHioN - Login or Signup")

    users = load_users()

    tab1, tab2 = st.tabs(["🔑 Login", "🆕 Signup"])

    with tab1:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if username in users and users[username] == password:
                st.success("Logged in successfully!")
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state.page = "options"
            else:
                st.error("Invalid username or password.")

    with tab2:
        new_username = st.text_input("Choose a Username", key="signup_username")
        new_password = st.text_input("Choose a Password", type="password", key="signup_password")
        if st.button("Signup"):
            if new_username in users:
                st.warning("Username already exists.")
            else:
                users[new_username] = new_password
                save_users(users)
                st.success("Signup successful! Please log in.")
