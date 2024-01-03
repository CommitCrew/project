FROM python:3.10
ENV USERNAME=commitcrew
RUN mkdir -p /home/dockerdemo
WORKDIR /home/dockerdemo
COPY . /home/dockerdemo
RUN pip install --upgrade pip
EXPOSE 8080
RUN pip install -r requirements.txt
CMD ["python","app.py"]