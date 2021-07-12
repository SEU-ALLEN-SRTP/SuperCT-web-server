FROM tensorflow/tensorflow:2.3.0
EXPOSE 8501
WORKDIR /app
COPY ./requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt -i https://mirrors.bfsu.edu.cn/pypi/web/simple
COPY . .
CMD streamlit run main.py
