import os
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

st.set_page_config(page_title="AI Agent Demo", page_icon="🤖")
st.title("🤖 AI Agent – FAQ & Recommendations")

# Get API key from Streamlit secrets
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY is not set in Streamlit Secrets!")
    st.stop()

# Define a simple prompt template for FAQ
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
llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4o")
chain = LLMChain(llm=llm, prompt=prompt)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.run(question=user_input)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
