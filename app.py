import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document

st.set_page_config(page_title="AI Agent Demo", page_icon="🤖")
st.title("🤖 AI Agent – FAQ & Recommendations")

# Build FAQ knowledge base dynamically
@st.cache_resource
def build_faq_chain():
    faqs = [
        {"question": "What is your return policy?", "answer": "You can return products within 30 days."},
        {"question": "Do you offer discounts?", "answer": "Yes, we provide seasonal discounts."},
        {"question": "How can I contact support?", "answer": "Email us at support@company.com."}
    ]
    docs = [Document(page_content=f"Q: {f['question']} A: {f['answer']}") for f in faqs]
    embedding = OpenAIEmbeddings()
    faq_index = FAISS.from_documents(docs, embedding)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-4o"),
        retriever=faq_index.as_retriever()
    )
    return qa_chain

qa_chain = build_faq_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain({"question": user_input, "chat_history": []})
            response = result["answer"]
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
