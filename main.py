import streamlit as st
from model import create_vector_db,get_qa_chain

st.title("Mental Health Q and A 🧠👨🏻‍⚕️")

btn = st.button("create Knowledgebase")

if btn:
    pass


question=st.text_input("Ask your queries related to mental health:")

if question:
    chain=get_qa_chain()
    response=chain(question)

    st.header("Answer: ")
    st.write(response["result"])

if __name__ == "__main__":
    print("hii")