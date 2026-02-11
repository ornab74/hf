# HeartFlow Web App

A Flask-based HeartFlow experience with advanced quantum-inspired scoring, long-horizon self-development outlooks, and a glassmorphism UI.

## Features

- **Beautiful web UI** with a central “Find your Heartflow” glass card.
- **Twitter handle analysis flow** (`@handle` input).
- **Advanced HeartFlow scoring** across six axes with deterministic signal shaping.
- **Quantum simulation layer** (coherence, entropy bits, entanglement bits, dominant modes, trajectory).
- **Human improvement roadmaps** with 1-year, 5-year, and 10-year outlook cards.
- **Trips to become a better human** with challenge-oriented growth prompts.
- **CSRF protection** using per-session cryptographic tokens.
- **SRI enabled** for Bootstrap CSS/JS CDN assets.
- Security response headers (`X-Frame-Options`, `X-Content-Type-Options`, etc.).

## Setup

```bash
pip install -r requirements.txt
```

Set env variables:

- `TWITTER_BEARER_TOKEN` (optional but recommended for live tweet fetch)
- `FLASK_SECRET_KEY` (recommended in production)
- `SESSION_COOKIE_SECURE=1` (recommended behind HTTPS)

## Run

```bash
python main.py
```

Then open: `http://localhost:5000`

## Quick checks

- `GET /healthz` returns `{"ok": true}`.
- POSTing to `/analyze` without valid CSRF token returns HTTP 400.
