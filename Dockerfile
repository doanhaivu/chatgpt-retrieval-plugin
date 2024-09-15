# Stage 1: Build the base with Poetry and all dependencies
FROM python:3.10 as requirements-stage

WORKDIR /tmp

# Install Poetry
RUN pip install poetry

# Copy project files that are needed for dependency resolution
COPY ./pyproject.toml ./poetry.lock* /tmp/
# Copy the rpunct directory at this stage
COPY ./rpunct /tmp/rpunct

# Install all dependencies using Poetry, excluding dev dependencies
RUN poetry install --only main --no-interaction --no-ansi

# Stage 2: Build the final image
FROM python:3.10

WORKDIR /code

# Install basic Python packages
RUN pip install -U pip setuptools wheel

# Copy rpunct directory first
COPY ./rpunct /code/rpunct

# Copy the application code
COPY . /code/

# Copy the virtual environment from the first stage
COPY --from=requirements-stage /root/.cache/pypoetry/virtualenvs /code/.venv

# Activate the virtual environment by default in the container
ENV PATH="/code/.venv/chatgpt-retrieval-plugin-6WcazSRI-py3.10/bin:$PATH"

# Install rpunct as an editable package
RUN pip install --no-cache-dir -e /code/rpunct

# Install spaCy model
RUN python -m spacy download en_core_web_trf

# Set the default command
CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-${WEBSITES_PORT:-8080}}"]
