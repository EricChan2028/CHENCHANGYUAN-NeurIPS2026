# Executive summary: CPU-only offline forecast executable. Build on linux/amd64.
FROM python:3.13.13-slim-bookworm
LABEL qfbench2.interface_version="2.0" \
      qfbench2.track="forecasting" \
      qfbench2.verb="forecast"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY forecast.py /app/forecast.py
COPY LICENSE PROVENANCE.md /app/
RUN printf '#!/bin/sh\nexec python3 /app/forecast.py "$@"\n' > /usr/local/bin/forecast \
    && chmod +x /usr/local/bin/forecast \
    && useradd --no-create-home --uid 1000 runner
USER runner
CMD ["forecast", "--help"]
