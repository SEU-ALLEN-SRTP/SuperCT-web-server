FROM gpuci/miniconda-cuda:10.1-runtime-ubuntu18.04
EXPOSE 8501
WORKDIR /app
COPY env.yml ./env.yml
RUN conda env update -n base --file env.yml
COPY . .
CMD streamlit run main.py