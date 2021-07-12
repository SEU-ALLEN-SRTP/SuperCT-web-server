FROM gpuci/miniconda-cuda:10.1-runtime-ubuntu18.04
EXPOSE 8501
COPY .condarc /root/.condarc
WORKDIR /app
RUN conda install -y keras tensorflow==2.3.0 matplotlib pandas scikit-learn plotly && conda install -y -c conda-forge streamlit umap-learn
COPY . .
CMD streamlit run main.py
