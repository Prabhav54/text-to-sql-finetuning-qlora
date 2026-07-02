"""
Streamlit demo app for the fine-tuned model.

Run with:
    streamlit run app/streamlit_app.py

This is the deployable artifact you link on your resume/GitHub README —
a live demo is worth far more to a recruiter than a notebook screenshot.
"""

import sys
from pathlib import Path

import streamlit as st

# Allow running `streamlit run app/streamlit_app.py` from repo root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config_utils import load_config
from src.model_utils import load_model_for_inference
from src.inference import generate


st.set_page_config(page_title="Domain LLM — Fine-Tuned Demo", page_icon="🤖")
st.title("🤖 Fine-Tuned Domain Assistant")
st.caption("Llama 3.1 8B fine-tuned with QLoRA — ask a domain question below.")


@st.cache_resource
def get_model():
    cfg = load_config("config/config.yaml")
    adapter_path = cfg["training"]["final_adapter_dir"]
    model, tokenizer = load_model_for_inference(cfg, adapter_path)
    return model, tokenizer, cfg


model, tokenizer, cfg = get_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Generating..."):
            response = generate(model, tokenizer, user_input, cfg)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
