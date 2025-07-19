import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Assistant", page_icon="🤖")
st.header("Hi, I am Judy, your Assistant 🤖")


# Load the model
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="distilgpt2")

generator = load_model()

# FAQ dictionary with exact answers
FAQ_DICT = {
    "what is your return policy": "You can return products within 30 days.",
    "do you offer discounts": "Yes, we provide seasonal discounts.",
    "how can i contact support": "Email us at support@company.com."
}

# Prompt template for fallback
FAQ_CONTEXT = """FAQ:
Q: What is your return policy?
A: You can return products within 30 days.

Q: Do you offer discounts?
A: Yes, we provide seasonal discounts.

Q: How can I contact support?
A: Email us at support@company.com.

If the question is unrelated to the FAQs above, respond politely: "I'm sorry, I only answer FAQs for now."

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

    # Normalize user question to lower case and strip punctuation for dictionary lookup
    question_key = user_input.lower().strip().rstrip("?")

    # Check dictionary first
    if question_key in FAQ_DICT:
        response = FAQ_DICT[question_key]
    else:
        # Fallback to distilgpt2 generation
        prompt = f"{FAQ_CONTEXT}{user_input}\nA:"
        raw_response = generator(
            prompt,
            max_length=len(prompt.split()) + 50,
            do_sample=False,
            temperature=0.3,
        )[0]["generated_text"]

        answer_part = raw_response[len(prompt):].strip()
        for sep in ['\n', '.', '?', '!']:
            if sep in answer_part:
                answer_part = answer_part.split(sep)[0] + sep
                break

        response = answer_part

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
