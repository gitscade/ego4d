import streamlit as st

st.title("Interactive Python App in Browser")
user_input = st.text_input("Enter text here:")


if st.button("Print"):
    st.write(f"You entered: {user_input}")