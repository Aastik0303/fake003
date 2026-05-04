import streamlit as st
import os
from chat_agent import get_chat_agent
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="DeepDetect AI Chat", layout="wide")

st.title("🛡️ DeepDetect AI: Forensic Chat")
st.markdown("Upload an image and chat with the AI to analyze its authenticity using Vision Transformers.")

# Initialize chat history and Agent
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "agent" not in st.session_state:
    st.session_state.agent = get_chat_agent()

# Sidebar for Image Upload
with st.sidebar:
    st.header("1. Upload Target")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    image_path = None
    if uploaded_file is not None:
        # Save temp image for the ViT tool to access
        os.makedirs("temp", exist_ok=True)
        image_path = os.path.join("temp", uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(uploaded_file, caption="Target Image", use_column_width=True)
        st.success(f"Image cached at: {image_path}")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("Ask me to analyze the uploaded image..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process with LangGraph Agent
    with st.chat_message("assistant"):
        with st.spinner("🧠 Orchestrating Agents & Running ViT Inference..."):
            
            # Inject the image path into the prompt if an image is uploaded
            full_prompt = prompt
            if image_path:
                full_prompt += f"\n[System Context: The user has uploaded an image at path: {image_path}. Use this path if you need to run analysis.]"
            
            # Run the LangGraph Agent
            inputs = {"messages": [HumanMessage(content=full_prompt)]}
            response = st.session_state.agent.invoke(inputs)
            
            # Extract the final AI response
            final_ai_message = response["messages"][-1].content
            
            st.markdown(final_ai_message)
            st.session_state.messages.append({"role": "assistant", "content": final_ai_message})
