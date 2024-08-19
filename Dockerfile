
FROM python:3.10 as requirements-stage

WORKDIR /tmp

RUN pip install poetry

COPY ./pyproject.toml ./poetry.lock* /tmp/


RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

FROM python:3.10

WORKDIR /code
RUN pip install -U pip setuptools wheel
COPY --from=requirements-stage /tmp/requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN python -m spacy download en_core_web_trf


#modify rpunct use cuda
RUN pip install --no-cache-dir -e /code/rpunct


COPY . /code/

# Heroku uses PORT, Azure App Services uses WEBSITES_PORT, Fly.io uses 8080 by default
CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-${WEBSITES_PORT:-8080}}"]
