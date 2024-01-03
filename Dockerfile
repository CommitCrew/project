FROM python:3.10
ENV USERNAME=commitcrew
RUN mkdir -p /home/dockerdemo
WORKDIR /home/dockerdemo
COPY . /home/dockerdemo
EXPOSE 5000
RUN pip install flask
RUN pip install -r requirements.txt
CMD ["python","app.py"]