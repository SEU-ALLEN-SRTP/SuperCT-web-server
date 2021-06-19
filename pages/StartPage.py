import streamlit as st


def page(state):
    with open('documentation/welcome.md') as f:
        st.markdown(f.read())
