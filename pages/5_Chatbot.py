# =============================================================
# Module: 5_Chatbot.py
# Project Area: AI Toolkit Hub
# Purpose: Implements the runtime logic for this project component.
# Notes: Keep this file focused on one responsibility so future
# maintenance remains straightforward.
# =============================================================

import streamlit as st
from openai import OpenAI

# Page configuration
st.set_page_config(
    page_title="ChatGPT Assistant",
    page_icon="💬",
    layout="wide"
)

# Check authentication
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("Please log in first!")
    st.stop()

# Initialize API client (Supports NVIDIA NIM or OpenAI)
api_key = (
    st.secrets.get("NVIDIA_API_KEY")
    or st.secrets.get("Nvidia_api_key")
    or st.secrets.get("nvidia_api_key")
    or st.secrets.get("OPENAI_API_KEY")
    or st.secrets.get("LLM_API_KEY")
)

is_nvidia = bool(
    st.secrets.get("NVIDIA_API_KEY")
    or st.secrets.get("Nvidia_api_key")
    or st.secrets.get("nvidia_api_key")
    or (api_key and api_key.startswith("nvapi-"))
)
default_base_url = "https://integrate.api.nvidia.com/v1" if is_nvidia else "https://api.openai.com/v1"
base_url = st.secrets.get("NVIDIA_BASE_URL") or st.secrets.get("OPENAI_BASE_URL", default_base_url)

client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0)

# Configurable chat model
default_model = "meta/llama-3.2-11b-vision-instruct" if is_nvidia else "gpt-4o-mini"
CHAT_MODEL = st.secrets.get("CHAT_MODEL", default_model)

# Title
st.title("💬 AI Assistant")
st.caption("Chat with AI models with live per-message model tracking")

# Domain-specific system prompts
DOMAIN_PROMPTS = {
    "Cybersecurity": """You are a cybersecurity expert assistant.
Analyze incidents, threats, and provide technical guidance.""",
    
    "Data Science": """You are a data science expert assistant.
Help with analysis, visualization, and statistical insights.""",
    
    "IT Operations": """You are an IT operations expert assistant.
Help troubleshoot issues, optimize systems, and manage tickets."""
}

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'selected_domain' not in st.session_state:
    st.session_state.selected_domain = "Cybersecurity"

# Sidebar with controls
with st.sidebar:
    st.subheader("User Info")
    st.write(f"👤 {st.session_state.username}")
    st.write(f"🔑 {st.session_state.role.upper()}")
    
    st.markdown("---")
    st.subheader("Domain Selection")
    
    # Domain selector
    domain = st.selectbox(
        "Choose Domain",
        ["Cybersecurity", "Data Science", "IT Operations"],
        index=["Cybersecurity", "Data Science", "IT Operations"].index(st.session_state.selected_domain)
    )
    
    # Update domain if changed
    if domain != st.session_state.selected_domain:
        st.session_state.selected_domain = domain
        st.session_state.messages = []  # Clear chat when domain changes
        st.success(f"Switched to {domain} domain")
        st.rerun()
    
    # Display current system prompt
    with st.expander("View System Prompt"):
        st.code(DOMAIN_PROMPTS[domain], language="text")
    
    st.markdown("---")
    st.subheader("Chat Controls")
    
    # Display message count
    message_count = len([m for m in st.session_state.messages if m["role"] != "system"])
    st.metric("Messages", message_count)
    
    # Clear chat button
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.subheader("Model Selection")
    
    if is_nvidia:
        nvidia_models = [
            "meta/llama-3.2-11b-vision-instruct",
            "nvidia/nemotron-3-super-120b-a12b",
            "nvidia/ising-calibration-1.5-31b",
            "Custom Model..."
        ]
        chosen = st.selectbox("Choose AI Model", nvidia_models, index=0)
        if chosen == "Custom Model...":
            model = st.text_input("Enter Model ID", value=CHAT_MODEL)
        else:
            model = chosen
    else:
        openai_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "Custom Model..."]
        chosen = st.selectbox("Choose AI Model", openai_models, index=0)
        if chosen == "Custom Model...":
            model = st.text_input("Enter Model ID", value=CHAT_MODEL)
        else:
            model = chosen
            
    st.info(f"🤖 **Active Model:** `{model}`")

# Display chat history with model badges and switch indicators
prev_msg_model = None
for message in st.session_state.messages:
    msg_model = message.get("model")
    
    # Render model switch transition banner if the model changed mid-conversation
    if msg_model and prev_msg_model and msg_model != prev_msg_model:
        st.markdown(
            f"""<div style="background: rgba(59, 130, 246, 0.1); border-left: 4px solid #3B82F6; padding: 6px 12px; border-radius: 4px; margin: 12px 0; font-size: 0.85rem; color: #93C5FD;">
            🔄 <b>Model Switched:</b> <code>{prev_msg_model}</code> ➔ <code style="color: #60A5FA; font-weight: 600;">{msg_model}</code>
            </div>""",
            unsafe_allow_html=True
        )
    if msg_model:
        prev_msg_model = msg_model
        
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            if msg_model:
                st.caption(f"🎯 **Prompt sent to:** `{msg_model}`")
            st.markdown(message["content"])
        else:
            if msg_model:
                st.caption(f"🤖 **Response generated by:** `{msg_model}`")
            st.markdown(message["content"])

# Get user input
prompt = st.chat_input("Say something...")

if prompt:
    # Check if this prompt switches models compared to the previous turn
    if prev_msg_model and model != prev_msg_model:
        st.markdown(
            f"""<div style="background: rgba(59, 130, 246, 0.1); border-left: 4px solid #3B82F6; padding: 6px 12px; border-radius: 4px; margin: 12px 0; font-size: 0.85rem; color: #93C5FD;">
            🔄 <b>Model Switched:</b> <code>{prev_msg_model}</code> ➔ <code style="color: #60A5FA; font-weight: 600;">{model}</code>
            </div>""",
            unsafe_allow_html=True
        )
        
    # Display user message
    with st.chat_message("user"):
        st.caption(f"🎯 **Prompt sent to:** `{model}`")
        st.markdown(prompt)
    
    # Add user message to session state with model tag
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "model": model
    })
    
    # Prepare API payload (only role and content fields for API compliance)
    api_messages = [
        {"role": "system", "content": DOMAIN_PROMPTS[st.session_state.selected_domain]}
    ] + [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    
    # Call AI API with fallback and error handling
    with st.chat_message("assistant"):
        with st.spinner(f"Thinking with {model}..."):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=api_messages,
                    stream=False
                )
                full_reply = completion.choices[0].message.content or "No response content returned."
                st.caption(f"🤖 **Response generated by:** `{model}`")
                st.markdown(full_reply)
                
                # Save assistant response with model tag
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_reply,
                    "model": model
                })
            except Exception as e:
                error_msg = str(e)
                st.error(f"⚠️ API Response Error ({model}): {error_msg}")
                if "503" in error_msg or "ResourceExhausted" in error_msg:
                    st.warning(f"⚡ NVIDIA's GPU server for `{model}` is currently overloaded. Switch to **`meta/llama-3.2-11b-vision-instruct`** in the sidebar for instant responses.")
                elif "504" in error_msg or "timeout" in error_msg.lower():
                    st.warning(f"⏱️ The model `{model}` timed out. Try switching to **`meta/llama-3.2-11b-vision-instruct`**.")
                elif "404" in error_msg or "410" in error_msg or "403" in error_msg or "access" in error_msg.lower():
                    st.info(f"💡 The model `{model}` is not accessible on your API key. Try switching model in the sidebar.")
