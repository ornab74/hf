# HeartFlow Onefile Secure Trainer

This app now runs as a **single-file `main.py`** Flask application with inline UI/CSS/JS.

## Security additions
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
- `TWITTER_BEARER_TOKEN` (optional; if unset, app falls back to public Nitter RSS for tweet pull)
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
