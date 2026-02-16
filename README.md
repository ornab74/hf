# HeartFlow Onefile Secure Trainer

This app now runs as a **single-file `main.py`** Flask application with inline UI/CSS/JS.

## Security additions
- About and Creators pages with markdown + MathJax equation rendering
- AES-GCM encrypted SQLite storage (`hf_secure.db`)
- Boot-time key derivation from `ENCRYPTION_PASSPHRASE` + PBKDF2 salt
- Extra entropy from `psutil` for key diversification
- CSRF protection + hardened security headers

## Required env vars
- `ENCRYPTION_PASSPHRASE` (**required**)

## Optional env vars
- `ENCRYPTION_SALT_B64` (if absent, generated at boot for process env)
- `ENCRYPTION_BOOT_NONCE_B64` (if absent, generated at boot for process env)
- `OPENAI_API_KEY`, `HF_OPENAI_MODEL`, `HF_OPENAI_BASE_URL`
- `TWITTER_BEARER_TOKEN` (required in default strict mode for compliant X API access)
- `X_COMPLIANCE_STRICT` (`1` default; keep enabled to block non-compliant tweet sources)
- `FLASK_SECRET_KEY`

## Run
```bash
python main.py
```

## Production
```bash
gunicorn main:app -b 0.0.0.0:${PORT:-3000} -w ${WEB_CONCURRENCY:-2} -k gthread --threads 4
```

## Persistent storage
- Default encrypted DB path: `/var/data/hf_secure.db`
- Override with `HF_DB_PATH` if needed


## X API compliance
- The app only pulls tweets from official X API v2 endpoints.
- In strict mode (`X_COMPLIANCE_STRICT=1`), analyze requests are rejected unless `TWITTER_BEARER_TOKEN` is set.
- No scraping or third-party mirrors are used in compliant mode.


## Pages
- `/` Home analyzer dashboard
- `/about` markdown + equations
- `/creators` markdown + equations
