FROM python:3.12-slim

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create true \
    && poetry install --no-dev --no-root

COPY ./app ./app

ENV PYTHONPATH=/app
ENV PORT=8080
EXPOSE 8080

CMD ["poetry", "run", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]
