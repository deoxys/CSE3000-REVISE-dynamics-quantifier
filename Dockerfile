# syntax=docker/dockerfile:1

FROM python:3.7

WORKDIR /code

# Get most recent CARLA library build and build from source 
COPY CARLA CARLA
WORKDIR /code/CARLA
RUN pip install -U pip setuptools wheel
RUN pip install -e .
RUN pip install kneed

WORKDIR /code
ENTRYPOINT [ "python3", "main.py" ]