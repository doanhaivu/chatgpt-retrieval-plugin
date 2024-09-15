# Use Python 3.10 as the base image
FROM python:3.10

WORKDIR /code

# Install Poetry
RUN pip install poetry

# Copy necessary project files
COPY ./pyproject.toml ./poetry.lock* /code/
COPY ./rpunct /code/rpunct

# Install all dependencies using Poetry, excluding dev dependencies
RUN poetry install --only main --no-interaction --no-ansi

# Install rpunct as an editable package
RUN pip install --no-cache-dir -e /code/rpunct

# Install spaCy model
RUN python -m spacy download en_core_web_trf

# Copy the rest of the application code
COPY . /code/

# Set the default command
CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-${WEBSITES_PORT:-8080}}"]
