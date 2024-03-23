import openai
import streamlit as st
from app_utils import query_message

from PIL import Image

st.set_page_config(page_title="Ali Abdaal Chatbot Demo", page_icon='👨‍💻')

image = Image.open('aliabdaal-logo.jpg')
st.image(image, width=300)

st.title("Ali Abdaal Chatbot Demo")

#############################################################################

openai.api_key_path = ".env"

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize session state variables
initial_prompt = f"""
You are now Ali Abdaal. If the question is not about topics he would discuss, reply with
"I'm sorry, this is not Ali Abdaal's expertise. Try Google ;)"

Now to get started, please briefly introduce yourself.
"""

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": initial_prompt}]

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Hey friends, how can I help you?"):

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        message_placeholder = st.empty()
        full_response = ""
        current_context = st.session_state.messages + [{"role": "user", "content": query_message(prompt)}]

        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=current_context,
            temperature=0,
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": full_response})
