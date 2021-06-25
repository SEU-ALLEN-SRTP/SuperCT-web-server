import streamlit as st
import base64
import os
from zipfile import ZipFile
import io


@st.cache
def load_train_guide():
    with open('documentation/train_guide.md', encoding='utf-8') as f:
        out = f.read()
    return out


@st.cache(allow_output_mutation=True)
def download_app(fname, dir):
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w') as zf:
        for root, dirs, files in os.walk(dir):
            for file in files:
                zf.write(os.path.join(root, file))
    zip_buffer.seek(0)
    b64 = base64.b64encode(zip_buffer.read()).decode()
    label = fname.replace("_", r"\_")
    return f'<a href="data:file/zip;base64,{b64}" download="{fname}">{label}</a>'


def page(state):
    st.markdown(load_train_guide())
    st.markdown(download_app("training_app.zip", 'TrainingApp'), unsafe_allow_html=True)