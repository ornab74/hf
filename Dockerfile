FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OQS_INSTALL_PATH=/usr/local \
    LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    cmake \
    ninja-build \
    build-essential \
    pkg-config \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PQ provenance inputs (must exist in repo root)
COPY requirements.txt lock.manifest.json lock.manifest.pqsig pq_pubkey.b64 ./

ARG LIBOQS_VERSION=0.14.0
ARG LIBOQS_TARBALL_SHA256=5b0df6138763b3fc4e385d58dbb2ee7c7c508a64a413d76a917529e3a9a207ea

# Build liboqs C library
RUN set -euo pipefail \
 && curl -fsSL -o /tmp/liboqs.tar.gz \
      "https://github.com/open-quantum-safe/liboqs/archive/refs/tags/${LIBOQS_VERSION}.tar.gz" \
 && echo "${LIBOQS_TARBALL_SHA256}  /tmp/liboqs.tar.gz" | sha256sum -c - \
 && mkdir -p /tmp/liboqs-src \
 && tar -xzf /tmp/liboqs.tar.gz -C /tmp/liboqs-src --strip-components=1 \
 && cmake -S /tmp/liboqs-src -B /tmp/liboqs-src/build \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DBUILD_SHARED_LIBS=ON \
      -DOQS_USE_OPENSSL=OFF \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -G Ninja \
 && cmake --build /tmp/liboqs-src/build --parallel \
 && cmake --install /tmp/liboqs-src/build \
 && ldconfig \
 && rm -rf /tmp/liboqs.tar.gz /tmp/liboqs-src

RUN python -m pip install --upgrade pip

# Install oqs python bindings (needed only for verification step)
RUN set -euo pipefail \
 && python -m pip install --no-cache-dir liboqs-python==0.14.1

# PQ authenticity verification
RUN python <<'PY'
import base64, binascii, hashlib, json, sys, pathlib
import oqs

manifest_path = pathlib.Path("lock.manifest.json")
sig_path = pathlib.Path("lock.manifest.pqsig")
pub_path = pathlib.Path("pq_pubkey.b64")
req_path = pathlib.Path("requirements.txt")

for p in (manifest_path, sig_path, pub_path, req_path):
    if not p.exists():
        print(f"ERROR: missing required file: {p}", file=sys.stderr)
        sys.exit(2)

def load_raw_or_b64(path: pathlib.Path) -> bytes:
    data = path.read_bytes().strip()
    try:
        txt = data.decode("ascii").strip()
    except UnicodeDecodeError:
        return data
    txt_compact = "".join(txt.split())
    try:
        decoded = base64.b64decode(txt_compact, validate=True)
    except (binascii.Error, ValueError):
        return data
    return decoded if decoded else data

# The manifest file is canonical JSON bytes (signed), but we still canonicalize
# after parsing to protect against trailing newline differences.
manifest_bytes = manifest_path.read_bytes()
try:
    manifest_obj = json.loads(manifest_bytes.decode("utf-8"))
except UnicodeDecodeError:
    print("ERROR: lock.manifest.json is not valid UTF-8 JSON", file=sys.stderr)
    sys.exit(6)

canonical = json.dumps(manifest_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

sig = load_raw_or_b64(sig_path)
pub = load_raw_or_b64(pub_path)

alg = manifest_obj.get("pq_alg") or "Dilithium2"
with oqs.Signature(alg) as v:
    ok = v.verify(canonical, sig, pub)

if not ok:
    print("ERROR: PQ signature verification FAILED for lock.manifest.json", file=sys.stderr)
    sys.exit(3)

expected = (manifest_obj.get("requirements_txt_sha256") or "").lower().strip()
if not expected or len(expected) != 64:
    print("ERROR: manifest missing requirements_txt_sha256", file=sys.stderr)
    sys.exit(4)

actual = hashlib.sha256(req_path.read_bytes()).hexdigest().lower()
if actual != expected:
    print("ERROR: requirements.txt SHA256 mismatch", file=sys.stderr)
    print(f" expected: {expected}", file=sys.stderr)
    print(f"   actual: {actual}", file=sys.stderr)
    sys.exit(5)

print("OK: PQ signature + requirements.txt SHA256 verified.")
PY

# Full dependency install (hash-locked)
RUN python -m pip install --no-cache-dir --require-hashes -r requirements.txt

# App
COPY . .

RUN useradd -ms /bin/bash appuser \
 && mkdir -p /app/static \
 && chmod 755 /app/static \
 && chown -R appuser:appuser /app

USER appuser

EXPOSE 3000

CMD ["gunicorn","main:app","-b","0.0.0.0:3000","-w","4","-k","gthread","--threads","4","--timeout","180","--graceful-timeout","30","--log-level","info","--preload","--max-requests","1000","--max-requests-jitter","200"]
