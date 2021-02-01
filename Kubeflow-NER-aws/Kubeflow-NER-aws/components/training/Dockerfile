FROM python:3.7
RUN pip install --upgrade pip
RUN python3 -m pip install spacy==2.0.18
RUN python3 -m pip install spacy download en
RUN python3 -m pip install boto3
RUN python3 -m pip install pyyaml
COPY ./src /pipelines/component/src
