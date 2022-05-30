# syntax=docker/dockerfile:1

FROM python:3.7

WORKDIR /code

# Get most recent CARLA library build and build from source 
COPY CARLA CARLA
WORKDIR /code/CARLA
RUN pip install -U pip setuptools wheel
RUN pip install -e .
RUN pip install kneed

RUN git clone https://github.com/skorch-dev/skorch.git
WORKDIR /skorch
# create and activate a virtual environment
RUN python -m pip install -r requirements.txt
# install pytorch version for your system (see below)
RUN python -m pip install .

WORKDIR /code
ENTRYPOINT [ "python3", "main.py" ]