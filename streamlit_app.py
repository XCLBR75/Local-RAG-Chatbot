import os
import streamlit as st
from PIL import Image
from agent_script import agent, llm, close_resources


# === Page Config ===
st.set_page_config(page_title="RAG Chatbot", layout="centered")

# === Load images once ===
if "avatars" not in st.session_state:
    st.session_state.avatars = {
        "user": Image.open("images/user.jpg").resize((40, 40)),
        "bot": Image.open("images/chatbot.jpg").resize((40, 40)),
    }

# === Initialize history ===
if "messages" not in st.session_state:
    st.session_state.messages = []  

# === Helper to add messages ===
def add_message(sender, text):
    st.session_state.messages.append({"sender": sender, "text": text})

# === Run agent and update response ===
def get_bot_response(query):
    max_retries = 1
    attempt = 0
    response = ""

    while attempt < max_retries:
        try:
            response = agent.run(query)
            if response and not response.strip().lower().startswith("error"):
                break
        except Exception as e:
            response = f"Error: {str(e)}"
        attempt += 1

    if response.strip().lower().startswith("error") or response.strip().lower().startswith("could not parse llm output"):
        try:
            raw_response = llm.invoke(query)
            response = raw_response if isinstance(raw_response, str) else str(raw_response)
        except Exception as fallback_error:
            response = f"Total failure: {fallback_error}"

    return response

# === Chat display ===
st.title("🤖 RAG Chatbot")
prev_sender = None

for msg in st.session_state.messages:
    is_user = (msg["sender"] == "user")
    avatar_img = st.session_state.avatars["user"] if is_user else st.session_state.avatars["bot"]
    bubble_color = "#DCF8C6" if is_user else "#E6E6E6"
    extra_margin_top = "20px" if (is_user and prev_sender == "bot") else "5px"

    cols = st.columns([1, 6, 1])

    with cols[0]:
        if not is_user:
            st.image(avatar_img, width=40, use_container_width=False)

    with cols[1]:
        st.markdown(
            f"""
            <div style="
                display:table;
                background-color:{bubble_color};
                padding:10px;
                border-radius:10px;
                color:black;
                max-width: 100%;
                word-wrap: break-word;
                margin-left: { 'auto' if is_user else '0' };
                margin-right: { '0' if is_user else 'auto' };
                margin-top: {extra_margin_top};
                ">
                {msg['text']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cols[2]:
        if is_user:
            st.image(avatar_img, width=40, use_container_width=False)




# === User input ===
query = st.chat_input("Type your message...")
if query:
    add_message("user", query)
    add_message("bot", "...")  
    st.session_state.pending_query = query  
    st.rerun()

# Check pending query on rerun
if "pending_query" in st.session_state:
    query_to_process = st.session_state.pop("pending_query")
    st.session_state.messages[-1]["text"] = get_bot_response(query_to_process)
    st.rerun()


# === Shutdown hook ===
def shutdown():
    try:
        close_resources()
    except OSError:
        pass
    except Exception as e:
        st.write(f"Error during shutdown: {e}")
    finally:
        os._exit(0)  

st.sidebar.button("🔴 Close App", on_click=shutdown)

