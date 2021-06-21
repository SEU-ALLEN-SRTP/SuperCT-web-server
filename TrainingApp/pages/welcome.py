import streamlit as st


@st.cache()
def load_welcome():
    with open('documentation/welcome.md') as f:
        out = f.read()
    return out


def page(state):
    st.markdown(load_welcome())
