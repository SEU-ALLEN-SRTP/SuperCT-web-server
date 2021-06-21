import streamlit as st
import base64
import os
import zipfile
import io

@st.cache
def load_train_guide():
    with open('documentation/train_guide.md', encoding='utf-8') as f:
        out = f.read()
    return out


@st.cache(allow_output_mutation=True)
def download_app():
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        for root, dirs, files in os.walk('TrainingApp'):
            for file in files:
                with open(os.path.join(root, file), 'br') as f:
                    zf.writestr(f.read())
    return zip_buffer

    b64 = base64.b64encode(text.encode()).decode()
    label = fname.replace("_", r"\_")
    return f'<a href="data:file/csv;base64,{b64}" download="{fname}">{label}</a>'


def page(state):
    st.markdown(load_train_guide())
    st.markdown(download_app(, "cell_type_prediction_{}.csv".format(time_str)),
                    unsafe_allow_html=True)