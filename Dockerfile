FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY scour ./scour
RUN pip install --no-cache-dir -e ".[api]"
# Build the benchmark corpus into the image so the demo has data on first start.
RUN python -m scour.eval.cranfield --out data/cranfield
ENV SCOUR_DATA=data/cranfield SCOUR_NEURAL=0 PORT=8000
CMD uvicorn scour.api:app --host 0.0.0.0 --port ${PORT}
