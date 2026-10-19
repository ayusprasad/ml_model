# Modular ML Model Lab

[![CI](https://github.com/ayusprasad/ml_model/actions/workflows/ci.yml/badge.svg)](https://github.com/ayusprasad/ml_model/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Demo%20API-000000?logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white)

A modular machine-learning workspace with data preparation and model pipeline examples plus a small Flask health API.

## Architecture

~~~text
app.py                 Flask health API
src/                   ingestion, cleaning, feature and model modules
steps/ and pipelines/  pipeline building blocks
data/ and artifacts/  project data and generated outputs
~~~

The API currently exposes a lightweight readiness page; the ML pipeline modules are explored and run separately.

## Run locally

### Prerequisites

- Python 3.11 or newer
- Docker for the container example
- Input datasets for pipeline runs

### Install

~~~bash
git clone https://github.com/ayusprasad/ml_model.git
cd ml_model
python -m venv .venv
python -m pip install -r requirements-app.txt
python app.py
~~~

Visit http://127.0.0.1:5000. The broader experiments may require requirements.txt and data artifacts.

### Docker

~~~bash
docker build -t ml-model-api .
docker run --rm -p 5000:5000 ml-model-api
~~~

## Checks

~~~bash
pytest -q
ruff check --select E4,E7,E9,F app.py tests
~~~

CI runs lint, API tests, compilation, and a Docker build. No environment variables are required for the health API.
