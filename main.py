import asyncio
import hashlib
import hmac
import math
import os
import re
import secrets
from typing import Any, Dict, List, Tuple

import httpx
from dotenv import load_dotenv
from flask import Flask, make_response, render_template, request, session

load_dotenv()

TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
SECRET_KEY = os.getenv("FLASK_SECRET_KEY") or secrets.token_urlsafe(32)
HF_REQUEST_TIMEOUT = float(os.getenv("HF_REQUEST_TIMEOUT", "20"))

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config.update(
    SECRET_KEY=SECRET_KEY,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("SESSION_COOKIE_SECURE", "0") == "1",
)

BOOTSTRAP_CSS = {
    "href": "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css",
    "integrity": "sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH",
}

BOOTSTRAP_JS = {
    "src": "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js",
    "integrity": "sha384-YvpcrYf0tY3lHB60NNkmXc5s9fDVZLESaAA55NDzOxhy9GkcIdslK1eN7N6jIeHz",
}

HANDLE_RE = re.compile(r"^[A-Za-z0-9_]{1,15}$")
AXIS_TERMS = {
    "SR": ["build", "mission", "future", "planet", "scale", "infrastructure"],
    "CT": ["thanks", "love", "help", "support", "care", "community"],
    "CF": ["new", "launch", "design", "prototype", "idea", "creative"],
    "GDI_INV": ["open", "share", "fair", "public", "transparency"],
    "CAP": ["risk", "hard", "challenge", "truth", "fight", "bold"],
    "HCS": ["together", "align", "peace", "respect", "team", "bridge"],
}


@app.after_request
def set_security_headers(resp):
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    resp.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self' https://cdn.jsdelivr.net; script-src 'self' https://cdn.jsdelivr.net; img-src 'self' data:; font-src 'self' https://cdn.jsdelivr.net; connect-src 'self'; frame-ancestors 'none'"
    return resp


def get_csrf_token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


def verify_csrf(token: str) -> bool:
    saved = session.get("csrf_token", "")
    return bool(saved and token) and hmac.compare_digest(saved, token)


def sanitize_handle(raw: str) -> str:
    clean = (raw or "").strip().lstrip("@").strip()
    if not HANDLE_RE.match(clean):
        raise ValueError("Twitter handle must be 1-15 chars (letters, numbers, underscore).")
    return clean


async def fetch_recent_texts(handle: str, limit: int = 20) -> List[str]:
    if not TWITTER_BEARER_TOKEN:
        return []

    headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}
    async with httpx.AsyncClient(timeout=HF_REQUEST_TIMEOUT) as client:
        user_resp = await client.get(
            f"https://api.twitter.com/2/users/by/username/{handle}",
            headers=headers,
            params={"user.fields": "id"},
        )
        user_resp.raise_for_status()
        uid = user_resp.json().get("data", {}).get("id")
        if not uid:
            return []

        tweets_resp = await client.get(
            f"https://api.twitter.com/2/users/{uid}/tweets",
            headers=headers,
            params={
                "max_results": min(max(limit, 5), 100),
                "exclude": "retweets,replies",
                "tweet.fields": "lang,created_at",
            },
        )
        tweets_resp.raise_for_status()
        rows = tweets_resp.json().get("data", [])
        return [r.get("text", "") for r in rows if r.get("text")]




def sanitize_display_text(value: str, max_len: int = 280) -> str:
    raw = str(value or "")
    cleaned = "".join(ch for ch in raw if ch == "\n" or 32 <= ord(ch) <= 126)
    cleaned = cleaned.replace("<", "").replace(">", "").replace("`", "")
    cleaned = cleaned.replace("javascript:", "")
    cleaned = cleaned.strip()[:max_len]
    return cleaned


def sanitize_result_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    safe = dict(result)
    safe["handle"] = sanitize_display_text(result.get("handle", ""), 15)
    safe["vibe"] = sanitize_display_text(result.get("vibe", ""), 30)
    safe["signature"] = sanitize_display_text(result.get("signature", ""), 32)

    safe_quantum = dict(result.get("quantum", {}))
    safe_modes = []
    for mode in safe_quantum.get("dominant_modes", [])[:6]:
        safe_modes.append({
            "axis": sanitize_display_text(mode.get("axis", ""), 16),
            "weight": float(mode.get("weight", 0.0)),
        })
    safe_quantum["dominant_modes"] = safe_modes
    safe["quantum"] = safe_quantum

    safe_outlooks = []
    for o in result.get("outlooks", [])[:3]:
        safe_outlooks.append({
            "horizon": sanitize_display_text(o.get("horizon", ""), 24),
            "title": sanitize_display_text(o.get("title", ""), 64),
            "focus": sanitize_display_text(o.get("focus", ""), 280),
            "actions": [sanitize_display_text(a, 220) for a in o.get("actions", [])[:5]],
            "milestone": sanitize_display_text(o.get("milestone", ""), 220),
        })
    safe["outlooks"] = safe_outlooks

    safe_trips = []
    for t in result.get("human_trips", [])[:4]:
        safe_trips.append({
            "name": sanitize_display_text(t.get("name", ""), 80),
            "why": sanitize_display_text(t.get("why", ""), 220),
            "challenge": sanitize_display_text(t.get("challenge", ""), 220),
        })
    safe["human_trips"] = safe_trips

    safe_axes = {}
    for k, v in result.get("axes", {}).items():
        kk = sanitize_display_text(k, 20)
        safe_axes[kk] = max(0.0, min(1.0, float(v)))
    safe["axes"] = safe_axes
    return safe

def _term_ratio(text_blob: str, terms: List[str]) -> float:
    text = text_blob.lower()
    if not text.strip():
        return 0.5
    hits = sum(text.count(t) for t in terms)
    return min(1.0, hits / 8.0)


def _quantum_simulation(seed: int, axes: Dict[str, float]) -> Dict[str, Any]:
    axis_keys = list(axes.keys())
    amps: List[complex] = []
    for idx, k in enumerate(axis_keys):
        mag = max(0.1, axes[k])
        phase = ((seed >> (idx * 5)) & 0xFF) / 255.0 * math.tau
        amps.append(complex(mag * math.cos(phase), mag * math.sin(phase)))

    couples: List[Tuple[int, int, float]] = [
        (0, 2, 0.18),  # SR<->CF
        (1, 5, 0.16),  # CT<->HCS
        (2, 4, 0.14),  # CF<->CAP
        (3, 0, 0.12),  # GDI_INV<->SR
        (4, 1, 0.12),  # CAP<->CT
        (5, 3, 0.10),  # HCS<->GDI_INV
    ]

    trajectory: List[float] = []
    for t in range(48):
        phase_kick = ((seed >> (t % 16)) & 0x1F) / 31.0 * 0.09
        new_amps = amps[:]
        for i, j, c in couples:
            spin = complex(math.cos(phase_kick + t * 0.03), math.sin(phase_kick + t * 0.03))
            new_amps[i] += amps[j] * c * spin
            new_amps[j] += amps[i] * c * spin.conjugate()
        norm = math.sqrt(sum(abs(a) ** 2 for a in new_amps)) or 1.0
        amps = [a / norm for a in new_amps]
        trajectory.append(round(sum(abs(a) for a in amps) / len(amps), 4))

    probs = [abs(a) ** 2 for a in amps]
    p_sum = sum(probs) or 1.0
    probs = [p / p_sum for p in probs]

    entropy = -sum(p * math.log2(max(p, 1e-12)) for p in probs)
    coherence = sum(abs(a) for a in amps) / len(amps)
    entanglement_bits = 2.0 * entropy / len(amps)

    dominant = sorted(zip(axis_keys, probs), key=lambda x: x[1], reverse=True)[:3]
    return {
        "coherence": round(coherence, 4),
        "entropy_bits": round(entropy, 4),
        "entanglement_bits": round(entanglement_bits, 4),
        "trajectory": trajectory[::6],
        "dominant_modes": [{"axis": k, "weight": round(v, 4)} for k, v in dominant],
    }


def _build_outlooks(axes: Dict[str, float], quantum: Dict[str, Any]) -> List[Dict[str, Any]]:
    dominant_axis = quantum["dominant_modes"][0]["axis"] if quantum["dominant_modes"] else "SR"
    coherence = quantum["coherence"]

    base_focus = {
        "SR": "Stewardship & systems thinking",
        "CT": "Compassion through service",
        "CF": "Creative ship velocity",
        "GDI_INV": "Integrity, fairness, generosity",
        "CAP": "Courageous truth under pressure",
        "HCS": "Harmony and bridge-building",
    }[dominant_axis]

    return [
        {
            "horizon": "1-year",
            "title": "Foundation Loop",
            "focus": f"Stabilize {base_focus.lower()} with weekly measurable habits.",
            "actions": [
                "Ship one meaningful act/week that helps people beyond your circle.",
                "Run a monthly reflection on decisions, tradeoffs, and impact.",
                "Track mood + attention + contribution in a simple journal.",
            ],
            "milestone": f"Target coherence >= {max(0.55, coherence):.2f} with consistent weekly cadence.",
        },
        {
            "horizon": "5-year",
            "title": "Compounding Character Arc",
            "focus": "Convert strengths into community-positive systems and mentorship.",
            "actions": [
                "Build a small circle that practices radical honesty and service.",
                "Create a public artifact (tool/course/community) that outlives short-term trends.",
                "Take one difficult ethical stand each year and document lessons.",
            ],
            "milestone": "A repeatable personal operating system others can adopt.",
        },
        {
            "horizon": "10-year",
            "title": "Legacy & Human Uplift",
            "focus": "Design for legacy impact: better institutions, not just better outcomes.",
            "actions": [
                "Sponsor or found mission-driven initiatives that increase human dignity.",
                "Train successors; make your best frameworks open and teachable.",
                "Invest in reconciliation: bridge groups, generations, and viewpoints.",
            ],
            "milestone": "Your work improves trust, capability, and cooperation at scale.",
        },
    ]


def _build_human_trips(axes: Dict[str, float]) -> List[Dict[str, str]]:
    top_axes = sorted(axes.items(), key=lambda x: x[1], reverse=True)
    low_axes = sorted(axes.items(), key=lambda x: x[1])

    return [
        {
            "name": "Service Pilgrimage Sprint",
            "why": f"Amplify {top_axes[0][0]} into lived compassion.",
            "challenge": "Spend 3 weekends in the next 90 days volunteering where outcomes are measurable.",
        },
        {
            "name": "Silence + Systems Retreat",
            "why": f"Reduce noise and strengthen weaker axis {low_axes[0][0]}.",
            "challenge": "Do a 48-hour no-social retreat and design one life-system improvement.",
        },
        {
            "name": "Bridge Builder Expedition",
            "why": "Grow harmony and courage together through difficult conversations.",
            "challenge": "Host 6 structured dialogues between people who disagree, with clear listening rules.",
        },
    ]


def score_heartflow(handle: str, texts: List[str]) -> Dict[str, Any]:
    blob = "\n".join(texts)
    seed = int(hashlib.sha256((handle + blob[:3000]).encode()).hexdigest(), 16)

    axes = {k: _term_ratio(blob, terms) for k, terms in AXIS_TERMS.items()}

    jitter = ((seed % 1000) / 1000.0 - 0.5) * 0.06
    for k in axes:
        axes[k] = max(0.0, min(1.0, axes[k] + jitter))

    quantum = _quantum_simulation(seed, axes)
    coherence_boost = (quantum["coherence"] - 0.5) * 0.08
    for k in axes:
        axes[k] = max(0.0, min(1.0, axes[k] + coherence_boost))

    overall = round(sum(axes.values()) / len(axes) * 100, 1)
    vibe = "Harmonic" if overall >= 66 else "Emergent" if overall >= 45 else "Chaotic"

    rgb = ((seed >> 8) & 255, (seed >> 16) & 255, (seed >> 24) & 255)
    glass = f"linear-gradient(135deg, rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.34), rgba(120, 180, 255, 0.18))"

    return {
        "handle": handle,
        "axes": {k: round(v, 3) for k, v in axes.items()},
        "overall": overall,
        "vibe": vibe,
        "tweets_used": len(texts),
        "signature": f"{seed % 10**8:08d}",
        "glass": glass,
        "quantum": quantum,
        "outlooks": _build_outlooks(axes, quantum),
        "human_trips": _build_human_trips(axes),
    }


@app.get("/")
def index():
    return render_template(
        "index.html",
        csrf_token=get_csrf_token(),
        result=None,
        error=None,
        bootstrap_css=BOOTSTRAP_CSS,
        bootstrap_js=BOOTSTRAP_JS,
    )


@app.post("/analyze")
def analyze():
    token = request.form.get("csrf_token", "")
    if not verify_csrf(token):
        return make_response("CSRF validation failed", 400)

    try:
        handle = sanitize_handle(request.form.get("handle", ""))
    except ValueError as exc:
        return render_template(
            "index.html",
            csrf_token=get_csrf_token(),
            result=None,
            error=str(exc),
            bootstrap_css=BOOTSTRAP_CSS,
            bootstrap_js=BOOTSTRAP_JS,
        )

    try:
        texts = asyncio.run(fetch_recent_texts(handle))
    except Exception:
        texts = []

    result = sanitize_result_payload(score_heartflow(handle, texts))

    return render_template(
        "index.html",
        csrf_token=get_csrf_token(),
        result=result,
        error=None,
        bootstrap_css=BOOTSTRAP_CSS,
        bootstrap_js=BOOTSTRAP_JS,
    )


@app.get("/healthz")
def healthz():
    return {"ok": True}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
