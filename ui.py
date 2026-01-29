import streamlit as st
import requests

# ================================
# CONFIG
# ================================

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="🎥",
    layout="centered"
)

# ================================
# UI HEADER
# ================================

st.title("🎥 YouTube RAG Assistant")
st.write(
    "Ask questions from a YouTube video using AI. "
    "The answer is generated only from the video transcript."
)

st.divider()

# ================================
# INPUTS
# ================================

video_id = st.text_input(
    "YouTube Video ID",
    placeholder="Example: yKeNBjo_lJU"
)

question = st.text_area(
    "Your Question",
    placeholder="Example: What is FAISS?",
    height=100
)

# ================================
# BUTTON ACTION
# ================================

if st.button("Ask Question 🚀"):
    if not video_id or not question:
        st.warning("Please enter both Video ID and Question")
    else:
        with st.spinner("Thinking... 🤔"):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "video_id": video_id,
                        "question": question
                    },
                    timeout=120
                )

                if response.status_code == 200:
                    answer = response.json()["answer"]
                    st.success("Answer:")
                    st.write(answer)

                else:
                    error_msg = response.json().get("detail", "Something went wrong")
                    st.error(error_msg)

            except requests.exceptions.RequestException:
                st.error("Cannot connect to backend API")

# ================================
# FOOTER
# ================================

st.divider()
st.caption("Built with FastAPI + LangChain + FAISS + Streamlit")
