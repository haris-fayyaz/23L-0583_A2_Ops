FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=5000

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

# Packages first: this layer is reused until requirements change
COPY requirements-web.txt .
RUN pip install -r requirements-web.txt

# Source last: it changes often
COPY --chown=appuser:appuser app ./app
RUN mkdir -p /app/instance && chown appuser:appuser /app/instance

USER appuser
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3)"

CMD ["python", "-m", "app.main"]