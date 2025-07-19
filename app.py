import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Free AI Agent", page_icon="🤖")
st.title("🤖 Free AI Agent – FAQ & Recommendations (No OpenAI)")

# Load a lightweight Hugging Face model (cached for faster reloads)
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="distilgpt2")

generator = load_model()

FAQ_CONTEXT = """
You are a helpful assistant that answers FAQs:

1. What is your return policy? -> You can return products within 30 days.
2. Do you offer discounts? -> Yes, we provide seasonal discounts.
3. How can I contact support? -> Email us at support@company.com.

If the question is unrelated to the FAQ, politely say: "I’m sorry, I only answer FAQs for now."
"""

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
            raw_response = generator(
                f"{FAQ_CONTEXT}\nQuestion: {user_input}\nAnswer:",
                max_length=120,
                do_sample=True,
                temperature=0.3,
                top_p=0.9
            )[0]["generated_text"]

            # Extract answer after "Answer:"
            response = raw_response.split("Answer:")[-1].strip()
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
