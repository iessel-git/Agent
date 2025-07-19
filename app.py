import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Free AI Agent", page_icon="🤖")
st.title("🤖 Free AI Agent – FAQ & Recommendations (No OpenAI)")

# Load a lightweight Hugging Face model (cached for faster reloads)
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="distilgpt2")

generator = load_model()

FAQ_CONTEXT = """FAQ:
Q: What is your return policy?
A: You can return products within 30 days.

Q: Do you offer discounts?
A: Yes, we provide seasonal discounts.

Q: How can I contact support?
A: Email us at support@company.com.

Now, answer the following question briefly:
Q: """

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle user input
if user_input := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            prompt = f"{FAQ_CONTEXT}{user_input}\nA:"
            raw_response = generator(
                prompt,
                max_length=len(prompt.split()) + 50,  # limit output tokens roughly
                do_sample=False,
                temperature=0.3,
            )[0]["generated_text"]

            # Extract the answer portion by removing the prompt from output
            answer_part = raw_response[len(prompt):].strip()

            # Cut off at first newline or punctuation for a concise answer
            for sep in ['\n', '.', '?', '!']:
                if sep in answer_part:
                    answer_part = answer_part.split(sep)[0] + sep
                    break

            response = answer_part

            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
