# HeartFlow Web App

A Flask-based HeartFlow experience with advanced quantum-inspired scoring, long-horizon self-development outlooks, and a glassmorphism UI.
<img width="1510" height="1764" alt="copyImage" src="https://github.com/user-attachments/assets/e7d33297-7f27-4c7d-a096-260ae81d6ef4" />

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
# HeartFlow

HeartFlow is a Textual-based terminal app that fetches X/Twitter content and infers a 6-axis HeartFlow vector from recent posts.

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

Set env variables:

- `TWITTER_BEARER_TOKEN` (optional but recommended for live tweet fetch)
- `FLASK_SECRET_KEY` (recommended in production)
- `SESSION_COOKIE_SECURE=1` (recommended behind HTTPS)

## Run
2. Ensure these environment variables are set:

- `TWITTER_CLIENT_ID`
- `TWITTER_BEARER_TOKEN`
- `OPENAI_API_KEY`
- optional: `TWITTER_ACCESS_TOKEN`, `TWITTER_REFRESH_TOKEN`

3. Run the app:

```bash
python main.py
```

Then open: `http://localhost:5000`

## Quick checks

- `GET /healthz` returns `{"ok": true}`.
- POSTing to `/analyze` without valid CSRF token returns HTTP 400.
## HeartFlow demo (`elonmusk`)

The following is a real end-to-end demo run against live APIs for user `elonmusk`.

### API activity log

```text
2026-02-11 03:13:00,199 INFO HTTP Request: GET https://api.twitter.com/2/users/by/username/elonmusk?user.fields=id%2Cusername%2Cname "HTTP/1.1 200 OK"
2026-02-11 03:13:00,466 INFO HTTP Request: GET https://api.twitter.com/2/users/44196397/tweets?max_results=20&tweet.fields=created_at%2Cpublic_metrics%2Clang&exclude=retweets%2Creplies "HTTP/1.1 200 OK"
2026-02-11 03:13:06,659 INFO HTTP Request: POST https://api.openai.com:18080/responses "HTTP/1.1 200 OK"
2026-02-11 03:13:10,945 INFO HTTP Request: POST https://api.openai.com:18080/responses "HTTP/1.1 200 OK"
2026-02-11 03:13:14,129 INFO HTTP Request: POST https://api.openai.com:18080/responses "HTTP/1.1 200 OK"
2026-02-11 03:13:17,214 INFO HTTP Request: POST https://api.openai.com:18080/responses "HTTP/1.1 200 OK"
2026-02-11 03:13:19,554 INFO HTTP Request: POST https://api.openai.com:18080/responses "HTTP/1.1 200 OK"
2026-02-11 03:13:22,052 INFO HTTP Request: POST https://api.openai.com:18080/responses "HTTP/1.1 200 OK"
```

### Result JSON

```json
{
  "username": "elonmusk",
  "user_id": "44196397",
  "tweets_used": 20,
  "vector": {
    "SR": 0.525,
    "CT": 0.575,
    "CF": 0.575,
    "GDI_INV": 0.525,
    "CAP": 0.5,
    "HCS": 0.5125
  },
  "confidence": 0.6,
  "risk": 0.1,
  "flags": [
    "removed_segments=20",
    "CAP_clamped",
    "HCS_clamped"
  ],
  "reasoning_preview": "Evidence was unequal across axes, leading to adjustments. CAP lacked coverage, pulling it to neutral. Evidence limited confidence."
}
```

