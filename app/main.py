import streamlit as st
from router import router
from faq import ingest_faq_data,faq_chain
from pathlib import Path
from sql import sql_chain
faqs_path = Path(__file__).parent / "resource/faq_data.csv"
ingest_faq_data(faqs_path)
def ask(query):
    route = router(query).name
    if route=="faq":
        return faq_chain(query)
    elif route=="sql":
        return sql_chain(query)
    else:
        return f"Route {route} not implemented yet"


st.title("E-Commerce ChatBot")

query = st.chat_input("Write your query")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Handle user input
if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state["messages"].append({"role": "user", "content": query})

    # Dummy response for now
    response = ask(query)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state["messages"].append({"role": "assistant", "content": response})
