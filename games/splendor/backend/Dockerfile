FROM python:3.12-slim

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry add fastapi-cli \
    && poetry install --no-dev --no-root

COPY ./app ./app

ENV PYTHONPATH=/app
ENV PORT=8080
EXPOSE 8080

CMD ["python", "-m", "fastapi", "run", "app/api.py", "--port", "8080", "--host", "0.0.0.0"]
