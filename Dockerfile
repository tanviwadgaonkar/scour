FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY scour ./scour
RUN pip install --no-cache-dir ".[api]"
# Build the benchmark corpus into the image so the demo has data on first start.
# The download is checked against a pinned SHA-256 (see scour/eval/cranfield.py).
RUN python -m scour.eval.cranfield --out data/cranfield
RUN useradd --create-home --uid 1000 scour && chown -R scour /app
USER scour
ENV SCOUR_DATA=data/cranfield SCOUR_NEURAL=0 PORT=8000
EXPOSE 8000
CMD uvicorn scour.api:app --host 0.0.0.0 --port ${PORT}
