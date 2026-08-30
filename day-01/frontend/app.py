import streamlit as st
import requests

st.set_page_config(page_title="AI Engineering Roadmap - Day 01 Frontend", layout="centered")

st.title("💬 AI Chat Interface")
st.subheader("Day 01: Simple LLM Service")

# Backend URL
BACKEND_URL = "http://127.0.0.1:8000/chat"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is on your mind?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call the FastAPI backend
    try:
        with st.spinner("Thinking..."):
            response = requests.post(
                BACKEND_URL,
                json={"message": prompt}
            )
            response.raise_for_status()
            data = response.json()
            
            answer = data["answer"]
            model = data["model"]
            usage = data["usage"]

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(answer)
                st.caption(f"Model: {model} | Tokens: In {usage['input_tokens']}, Out {usage['output_tokens']}")
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend. Please make sure the FastAPI server is running at http://127.0.0.1:8000")
    except Exception as e:
        st.error(f"An error occurred: {e}")