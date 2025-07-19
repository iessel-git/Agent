import os
import re
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from rapidfuzz import process

st.set_page_config(page_title="AI Agent Demo", page_icon="🤖")
st.title("Hi, I am Rose, your Assitant 🤖 ")

# Get API key from environment variable or Streamlit secrets
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY is not set in environment variables!")
    st.stop()

# FAQ dictionary for exact or fuzzy matching
FAQ_DICT = {
    "what is your return policy": "You can return products within 30 days.",
    "do you offer discounts": "Yes, we provide seasonal discounts.",
    "how can i contact support": "Email us at support@company.com."
}

# Prompt template for OpenAI fallback
template = """
You are a helpful assistant that answers FAQs.

FAQ:
1. What is your return policy? -> You can return products within 30 days.
2. Do you offer discounts? -> Yes, we provide seasonal discounts.
3. How can I contact support? -> Email us at support@company.com.

If the question is unrelated to the FAQ, politely say: "I’m sorry, I only answer FAQs for now."

Question: {question}
Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo")
chain = LLMChain(llm=llm, prompt=prompt)

def normalize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def get_closest_match(query, choices, threshold=75):
    match, score, _ = process.extractOne(query, choices)
    if score >= threshold:
        return match
    return None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    norm_question = normalize(user_input)
    matched_key = get_closest_match(norm_question, FAQ_DICT.keys())

    if matched_key:
        response = FAQ_DICT[matched_key]
    else:
        with st.spinner("Thinking..."):
            response = chain.run(question=user_input)

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
