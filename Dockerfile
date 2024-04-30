
FROM python:3.11-alpine

WORKDIR .

COPY ./xyz /xyz

COPY ./requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
