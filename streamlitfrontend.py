import streamlit as st
import requests

st.set_page_config(page_title="Fitbot 1.0 ",page_icon='💪🏽')

st.title("Fitbot 1.0 💪🏽")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display conversation
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"**YOU:** {msg['content']}")
    else:
        st.markdown(f"**FITBOT:** {msg['content']}")

# Function to send message
def send_message():
    user_message = st.session_state.user_input.strip()
    if not user_message:
        return

    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_message})

    # Clear input box
    st.session_state.user_input = ""

    # Call backend
    with st.spinner("FitBot is thinking..."):
        try:
            response = requests.post(
                "http://127.0.0.1:5000/chat",
                json={"message": user_message},
                timeout=15
            )
            if response.status_code == 200:
                bot_reply = response.json()["response"]
                st.session_state.messages.append({"role": "bot", "content": bot_reply})
            else:
                st.error(f"Backend error: {response.text}")
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")

# Form for input (Enter key submits automatically)
with st.form(key="chat_form", clear_on_submit=False):
    st.text_input("Type your message here:", key="user_input")
    submit_button = st.form_submit_button("Send", on_click=send_message)
