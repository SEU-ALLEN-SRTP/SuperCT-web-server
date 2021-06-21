import streamlit as st


@st.cache()
def load_introduction():
    with open('documentation/introduction.md') as f:
        out = f.read()
    return out


def page(state):
    st.markdown('# Welcome to SuperCT Web App! :tada::zap::boom:'
                '\n---')
    c1, c2, c3 = st.beta_columns([6, 4, 7])
    with c1:
        st.image('documentation/model.png', use_column_width='always')
    with c2:
        st.image('documentation/outperform.png', use_column_width='always')
    with c3:
        st.image('documentation/eg.png', use_column_width='always')
    st.markdown(load_introduction())
