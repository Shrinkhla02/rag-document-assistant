import streamlit as st
from rag_chain import get_answer

st.title("RAG AI Assistant")

query = st.text_input("Ask a question from your documents")

if query:
    with st.spinner("Thinking..."):
        answer = get_answer(query)
        st.write(answer)