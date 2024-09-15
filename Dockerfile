# Use Python 3.10 as the base image
FROM python:3.10

WORKDIR /code

# Copy the pre-downloaded .whl files into the Docker image
COPY ./wheels /code/wheels

# Install pip dependencies
COPY ./requirements.txt /code/

# Install dependencies using the pre-downloaded .whl files
RUN pip install --no-cache-dir --find-links=/code/wheels -r requirements.txt

# Copy necessary project files
COPY ./rpunct /code/rpunct

# Install rpunct as an editable package
RUN pip install --no-cache-dir -e /code/rpunct

# Install spaCy model
RUN python -m spacy download en_core_web_trf

# Copy the rest of the application code
COPY . /code/

# Set the default command
CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-${WEBSITES_PORT:-8080}}"]
