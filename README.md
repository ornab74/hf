# Heartflow

Heartflow is a Flask app for structured analysis, encrypted storage, and markdown-based report pages with MathJax and Mermaid.
![screenshot](https://github.com/ornab74/hf/blob/main/screenshot2.png)


![screenshot2](https://github.com/ornab74/hf/blob/main/screenshot3.png)
## Quickstart

```bash
python main.py
```

## What it includes
- Home analyzer dashboard
- About and Creators pages with markdown, MathJax, and Mermaid rendering
- AES-GCM encrypted SQLite storage
- CSRF protection, rate limiting, and hardened security headers
- Optional OpenAI-backed structured analysis

## Architecture

```mermaid
flowchart TD
  U[User] --> W[Heartflow Web App]
  W --> M[Markdown + MathJax + Mermaid Pages]
  W --> A[Analysis Pipeline]
  A --> T[Public Tweet Fetch]
  A --> Q[Quantum-RAG / Deterministic Fallbacks]
  A --> S[Encrypted SQLite Storage]
  A --> O[Structured Result Rendering]
```

## Render Flow

```mermaid
sequenceDiagram
  participant U as User
  participant W as Web App
  participant A as Analysis Engine
  participant R as Render Layer
  U->>W: Submit handle
  W->>A: Analyze handle
  A->>A: Build RAG + quantum packet
  A->>R: Return structured result
  R->>U: Render charts, math, and diagrams
```

## Life Optimization Loop

```mermaid
flowchart LR
  S[State] --> C[Constraints]
  C --> L[Leverage]
  L --> A[Action]
  A --> V[Verification]
  V --> S
```

## Safety Scanner States

```mermaid
stateDiagram-v2
  [*] --> Low
  Low --> Medium: load rises or pressure increases
  Medium --> High: sustained overload
  High --> Medium: recovery window
  Medium --> Low: stable cadence
```

## Local assets
- MathJax is vendored in `static/vendor/mathjax/`
- Orbitron is vendored in `static/vendor/orbitron/`
- Mermaid is vendored in `static/vendor/mermaid/`

## Configuration
Required:
- `ENCRYPTION_PASSPHRASE`

Optional:
- `ENCRYPTION_SALT_B64`
- `ENCRYPTION_BOOT_NONCE_B64`
- `OPENAI_API_KEY`
- `HF_OPENAI_MODEL`
- `HF_OPENAI_BASE_URL`
- `TWITTER_BEARER_TOKEN`
- `X_COMPLIANCE_STRICT`
- `FLASK_SECRET_KEY`
- `HF_DB_PATH`
- `PORT`

## Production example

```bash
gunicorn main:app -b 0.0.0.0:${PORT:-3000} -w ${WEB_CONCURRENCY:-2} -k gthread --threads 4
```

## Notes
- In strict mode, X API access requires `TWITTER_BEARER_TOKEN`.
- The database defaults to `/var/data/hf_secure.db`.
- About and Creators pages now support fenced `mermaid` blocks, so diagrams render directly from markdown.
- GitHub will render the Mermaid fences in this README automatically.
