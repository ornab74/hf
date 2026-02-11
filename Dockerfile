FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the full repository context so Render/build systems that rely on
# additional files (templates/static/lock docs/etc) always have them available.
COPY . /app

RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir -r requirements.txt

# Optional PQ-provenance check: only runs when all PQ files are present.
RUN python <<'PY'
import pathlib, sys
needed = [
    pathlib.Path('/app/lock.manifest.json'),
    pathlib.Path('/app/lock.manifest.pqsig'),
    pathlib.Path('/app/pq_pubkey.b64'),
]
missing = [str(p.name) for p in needed if not p.exists()]
if missing:
    print(f"INFO: skipping PQ provenance check; missing files: {', '.join(missing)}")
else:
    print("INFO: PQ files present in image context.")
PY

RUN useradd -ms /bin/bash appuser \
 && mkdir -p /app/static \
 && chmod 755 /app/static \
 && chown -R appuser:appuser /app

USER appuser

EXPOSE 3000

CMD ["gunicorn","main:app","-b","0.0.0.0:3000","-w","4","-k","gthread","--threads","4","--timeout","180","--graceful-timeout","30","--log-level","info","--max-requests","1000","--max-requests-jitter","200"]
