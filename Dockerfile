FROM pytorch/pytorch:latest

RUN mkdir /app
WORKDIR /app

COPY app.py /app/app.py

CMD [ "python", "app.py" ]