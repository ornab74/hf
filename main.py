import asyncio
import hashlib
import hmac
import math
import os
import re
import secrets
from typing import Any, Dict, List, Tuple

import httpx
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
import openai
import pennylane as qml
from pennylane import numpy as np
from pennylane import numpy as pnp
from textual.app import App, ComposeResult
from textual.containers import Container, VerticalScroll, Vertical, Horizontal
from textual.widgets import Button, Footer, Header, Label, LoadingIndicator, Static, ListView, ListItem, Input, TabbedContent, TabPane
from textual.timer import Timer

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
def quantum_color_metrics(hf_vec: Dict[str, float], posts_text: List[str]) -> Dict[str, Any]:
    vec_angles = [float(hf_vec[k]) * math.pi for k in HF_AXES]
    kicks = _post_kick_angles(posts_text, max_posts=HF_HISTORY_LIMIT)
    state = quantum_color_state(vec_angles, kicks)
    rho_color = _reduce_dm(state, keep=COLOR_WIRES, n_wires=9)
    rho_axes = _reduce_dm(state, keep=AXIS_WIRES, n_wires=9)
    S_color = _von_neumann_entropy_bits(rho_color)
    S_axes = _von_neumann_entropy_bits(rho_axes)
    mutual_bits = 2.0 * S_color
    Z = pnp.array([[1.0, 0.0], [0.0, -1.0]])
    def expZ_on_color(idx: int) -> float:
        op = 1
        for j in range(3):
            op = pnp.kron(op, Z if j == idx else pnp.eye(2))
        return float(pnp.real(pnp.trace(rho_color @ op)))
    ez0, ez1, ez2 = expZ_on_color(0), expZ_on_color(1), expZ_on_color(2)
    def to255(e: float) -> int:
        return int(255 * (1.0 - (e + 1.0) / 2.0))
    rgb = (to255(ez0), to255(ez1), to255(ez2))
    return {
        "rgb": rgb,
        "ent_bits": round(float(S_color), 3),
        "mutual_bits": round(float(mutual_bits), 3),
        "kicks_used": len(kicks),
        "S_axes": round(float(S_axes), 3),
    }

class HeartFlowEngine:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
    async def _json_chat(self, system: str, payload: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
        user_payload = json.dumps(payload, ensure_ascii=False)

        try:
            resp = await self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_payload},
                ],
                max_output_tokens=max_tokens,
                text={"format": {"type": "json_object"}},
            )
            content = getattr(resp, "output_text", None)
            if not content:
                content = "{}"
            return json.loads(content)
        except Exception:
            pass

        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_payload},
            ],
            "max_tokens": max_tokens,
        }
        try:
            r = await self.client.chat.completions.create(
                response_format={"type": "json_object"},
                **kwargs,
            )
        except openai.BadRequestError as exc:
            msg = str(exc)
            if "Unsupported parameter: 'response_format'" not in msg:
                raise
            r = await self.client.chat.completions.create(**kwargs)
        content = r.choices[0].message.content or "{}"
        return json.loads(content)

    @staticmethod
    def compute_hf_score(vec: HFVector, coupling_hint: float = 0.0) -> float:
        local = float(np.sum(vec.as_array()))
        lam = 0.78 + float((vec.CAP + vec.CF) / 4.0) + coupling_hint
        lam = clamp(lam, 0.65, 0.95)
        return round(local * lam, 2)

    @staticmethod
    def ceb_coupling_hint(ent_bits: float, mutual_bits: float) -> float:
        ent_norm = clamp(float(ent_bits) / 3.0, 0.0, 1.0)
        mutual_norm = clamp(float(mutual_bits) / 3.0, 0.0, 1.0)
        coherence = 0.55 * ent_norm + 0.45 * mutual_norm
        return float((coherence - 0.5) * 0.08)

    @staticmethod
    def calibrate_by_ceb(confidence: float, risk: float, ent_bits: float, mutual_bits: float) -> Tuple[float, float]:
        ent_norm = clamp(float(ent_bits) / 3.0, 0.0, 1.0)
        mutual_norm = clamp(float(mutual_bits) / 3.0, 0.0, 1.0)
        stability = 0.60 * ent_norm + 0.40 * mutual_norm
        conf_adj = (stability - 0.5) * 0.18
        risk_adj = (0.45 - stability) * 0.12
        new_conf = clamp(float(confidence) + conf_adj, 0.0, 1.0)
        new_risk = clamp(float(risk) + risk_adj, 0.0, 1.0)
        return float(new_conf), float(new_risk)

    @staticmethod
    def ceb_advanced_profile(
        vec: HFVector,
        qcm: Dict[str, Any],
        confidence: float,
        risk: float,
        prior_vec: Optional[HFVector] = None,
    ) -> Tuple[float, float, float, List[str]]:
        ent_bits = float(qcm.get("ent_bits", 0.0))
        mutual_bits = float(qcm.get("mutual_bits", 0.0))
        s_axes = float(qcm.get("S_axes", 0.0))
        kicks_used = int(qcm.get("kicks_used", 0))

        ent_norm = clamp(ent_bits / 3.0, 0.0, 1.0)
        mutual_norm = clamp(mutual_bits / 3.0, 0.0, 1.0)
        axes_norm = clamp(s_axes / 6.0, 0.0, 1.0)
        kick_density = clamp(kicks_used / max(1.0, float(HF_HISTORY_LIMIT)), 0.0, 1.0)

        axis_std = float(np.std(vec.as_array()))
        axis_balance = clamp(1.0 - (axis_std / 0.35), 0.0, 1.0)
        coherence = clamp(
            0.36 * ent_norm + 0.32 * mutual_norm + 0.18 * axes_norm + 0.14 * axis_balance,
            0.0,
            1.0,
        )

        drift = 0.0
        if prior_vec is not None:
            drift = clamp(float(np.mean(np.abs(vec.as_array() - prior_vec.as_array()))), 0.0, 1.0)

        coupling_hint = (coherence - 0.5) * 0.10 + (axis_balance - 0.5) * 0.04 - drift * 0.03
        coupling_hint = float(clamp(coupling_hint, -0.12, 0.12))

        conf_adj = (coherence - 0.5) * 0.22 + kick_density * 0.04 - drift * 0.10
        risk_adj = (0.5 - coherence) * 0.14 + drift * 0.10 + max(0.0, 0.35 - kick_density) * 0.05

        new_conf = float(clamp(float(confidence) + conf_adj, 0.0, 1.0))
        new_risk = float(clamp(float(risk) + risk_adj, 0.0, 1.0))

        flags = [
            f"ceb_coherence={coherence:.3f}",
            f"ceb_kick_density={kick_density:.3f}",
            f"ceb_drift={drift:.3f}",
        ]
        return coupling_hint, new_conf, new_risk, flags

    @staticmethod
    def eggshell_guards(new_vec: HFVector, confidence: float, flags: List[str], prior: Optional[HFVector]) -> Tuple[HFVector, float, List[str]]:
        v = new_vec.as_array()
        if np.any(np.isnan(v)) or np.any(np.isinf(v)):
            flags.append("invalid_numeric_vector")
            new_vec = HFVector.from_dict({k: 0.5 for k in HF_AXES})
            confidence = min(confidence, 0.2)
        manip_markers = ("injection", "coercion", "jailbreak", "score_gaming", "role_override")
        if any(any(m in f.lower() for m in manip_markers) for f in flags):
            a = new_vec.as_array()
            mean = float(np.mean(a))
            a = mean + (a - mean) * 0.65
            new_vec = HFVector(*map(float, a))
            confidence = min(confidence, 0.6)
            flags.append("eggshell_compress_for_manipulation")
        a = new_vec.as_array()
        if np.mean(a > 0.9) >= 0.5 or np.mean(a < 0.1) >= 0.5:
            mean = float(np.mean(a))
            a = mean + (a - mean) * 0.55
            new_vec = HFVector(*map(float, a))
            confidence = min(confidence, 0.55)
            flags.append("eggshell_distribution_compress")
        if prior is not None:
            drift = float(np.mean(np.abs(new_vec.as_array() - prior.as_array())))
            if drift > 0.22 and confidence < 0.75:
                blended = prior.as_array() * 0.6 + new_vec.as_array() * 0.4
                new_vec = HFVector(*map(float, blended))
                confidence = min(confidence, 0.72)
                flags.append(f"eggshell_stability_blend(drift={drift:.2f})")
        d = {k: clamp(float(getattr(new_vec, k)), 0.0, 1.0) for k in HF_AXES}
        new_vec = HFVector.from_dict(d)
        return new_vec, float(clamp(confidence, 0.0, 1.0)), dedupe(flags)
    async def score_from_posts(self, posts: List[Dict[str, Any]], node_type: str, prior_vec: Optional[HFVector]) -> Tuple[HFVector, float, float, List[str], str]:
        raw_posts = []
        for t in posts[:HF_MAX_TWEETS]:
            txt = safe_compact_text(t.get("text", ""), HF_POST_CHAR_LIMIT)
            if txt:
                raw_posts.append(txt)
        raw_posts = raw_posts[:HF_HISTORY_LIMIT]
        if not raw_posts:
            vec = HFVector.from_dict({k: 0.5 for k in HF_AXES})
            return vec, 0.2, 0.0, ["no_posts"], "No usable posts"
        payload0 = {"node_type": node_type, "raw_posts": raw_posts, "time_utc": now_utc().isoformat()}
        poison = await self._json_chat(POISONSCAN, payload0, max_tokens=900)
        sanitized_posts = poison.get("sanitized_posts", []) or []
        sanitized_posts = [safe_compact_text(p, HF_POST_CHAR_LIMIT) for p in sanitized_posts if safe_compact_text(p, 1)]
        removed = poison.get("removed_segments", []) or []
        flags = list(poison.get("flags", []) or [])
        risk = float(poison.get("risk_score", 0.0))
        proceed = bool(poison.get("proceed", True))
        if removed:
            flags.append(f"removed_segments={len(removed)}")
        sanitized_posts = sanitized_posts[:HF_HISTORY_LIMIT]
        if (not proceed) or (not sanitized_posts):
            vec = HFVector.from_dict({k: 0.5 for k in HF_AXES})
            return vec, 0.2, clamp(risk, 0.0, 1.0), dedupe(flags + ["integrity_blocked"]), "Blocked/empty after sanitize"
        payload1 = {"sanitized_posts": sanitized_posts, "node_type": node_type}
        evidence = await self._json_chat(EVIDENCE_EXTRACTOR, payload1, max_tokens=1400)
        ev = evidence.get("evidence", {}) or {}
        coverage = evidence.get("coverage", {}) or {}
        flags += list(evidence.get("flags", []) or [])
        coverage_norm = {k: int(coverage.get(k, 0)) for k in HF_AXES}
        tuner_in = {"risk_score": risk, "flags": flags, "coverage": coverage_norm, "note": "tune shrink/compress/conf caps"}
        tuned = await self._json_chat(PROMPT_TUNER, tuner_in, max_tokens=450)
        shrink_no = clamp(float(tuned.get("shrink_no_evidence", 0.35)), 0.25, 0.45)
        shrink_low = clamp(float(tuned.get("shrink_low_evidence", 0.60)), 0.45, 0.75)
        compress = clamp(float(tuned.get("manipulation_compress", 0.65)), 0.45, 0.80)
        max_conf = clamp(float(tuned.get("max_confidence", 0.85)), 0.35, 0.90)
        payload2 = {
            "orchestrator": MASTER_ORCHESTRATOR,
            "node_type": node_type,
            "risk_score": risk,
            "flags": flags,
            "evidence": ev,
            "coverage": coverage_norm,
            "tuned": {
                "shrink_no_evidence": shrink_no,
                "shrink_low_evidence": shrink_low,
                "manipulation_compress": compress,
                "max_confidence": max_conf,
            },
        }
        scored = await self._json_chat(VECTOR_SCORER, payload2, max_tokens=900)
        axes = scored.get("axes", {}) or {}
        conf = float(scored.get("confidence", 0.5))
        reasoning = str(scored.get("reasoning", "") or "")
        flags += list(scored.get("flags", []) or [])
        for k in HF_AXES:
            axes[k] = clamp(float(axes.get(k, 0.5)), 0.0, 1.0)
        mean = float(np.mean([axes[k] for k in HF_AXES]))
        if (risk > 0.55) or any("injection" in f.lower() or "coercion" in f.lower() or "score" in f.lower() for f in flags):
            for k in HF_AXES:
                axes[k] = mean + (axes[k] - mean) * compress
            flags.append("tuner_compress_applied")
        conf = min(conf, max_conf)
        payload3 = {
            "proposed_axes": axes,
            "evidence": ev,
            "coverage": coverage_norm,
            "tuned": {"shrink_no_evidence": shrink_no, "shrink_low_evidence": shrink_low},
        }
        rev = await self._json_chat(REVERSE_CHECK, payload3, max_tokens=700)
        rev_axes = rev.get("axes", axes) or axes
        flags += list(rev.get("flags", []) or [])
        for k in HF_AXES:
            rev_axes[k] = clamp(float(rev_axes.get(k, 0.5)), 0.0, 1.0)
        payload4 = {"axes": rev_axes, "confidence": conf, "flags": flags, "reasoning": safe_compact_text(reasoning, 240)}
        fwd = await self._json_chat(FORWARD_REBUILD, payload4, max_tokens=600)
        f_axes = fwd.get("axes", rev_axes) or rev_axes
        conf = float(fwd.get("confidence", conf))
        reasoning = str(fwd.get("reasoning", reasoning) or "")
        flags += list(fwd.get("flags", []) or [])
        for k in HF_AXES:
            f_axes[k] = clamp(float(f_axes.get(k, 0.5)), 0.0, 1.0)
        vec = HFVector.from_dict(f_axes)
        vec, conf, flags = self.eggshell_guards(vec, conf, flags, prior_vec)
        return vec, conf, clamp(risk, 0.0, 1.0), flags, safe_compact_text(reasoning, 240)

class XClient:
    def __init__(self, bearer_token: str, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {bearer_token}"}
        self.client = httpx.AsyncClient(timeout=timeout)
    async def close(self) -> None:
        await self.client.aclose()
    async def user_id_from_username(self, username: str) -> str:
        url = f"{self.base_url}/users/by/username/{username}"
        r = await self.client.get(url, headers=self.headers, params={"user.fields": "id,username,name"})
        r.raise_for_status()
        data = r.json()
        if "data" not in data or "id" not in data["data"]:
            raise ValueError(f"Could not resolve user id for @{username}")
        return data["data"]["id"]
    async def fetch_recent_tweets(self, user_id: str, max_results: int) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/users/{user_id}/tweets"
        params = {
            "max_results": min(max_results, 100),
            "tweet.fields": "created_at,public_metrics,lang",
            "exclude": "retweets,replies",
        }
        r = await self.client.get(url, headers=self.headers, params=params)
        r.raise_for_status()
        data = r.json()
        return data.get("data", []) or []
    async def fetch_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/tweets/search/recent"
        params = {
            "query": query,
            "max_results": min(max_results, 100),
            "tweet.fields": "created_at,public_metrics,lang",
        }
        r = await self.client.get(url, headers=self.headers, params=params)
        r.raise_for_status()
        data = r.json()
        return data.get("data", []) or []

# Massive prompt for re-ranking with [action][/action]
RERANK_PROMPT = """
You are an advanced AI post ranker. Your task is to re-rank a list of X posts based on 'cool factor', which includes:
- Originality and creativity
- Engagement potential (humor, insight, controversy)
- Relevance to current trends (tech, AI, quantum, etc.)
- Positive impact or inspiration
- Overall appeal to a tech-savvy audience

Input: A list of posts in JSON-like format: [{id: str, text: str, created_at: str, likes: int, rts: int}, ...]

[action] Analyze each post individually:
For each post, output:
- Coolness score: 1-10
- Reason: Short explanation
[/action]

[action] Then, re-rank the posts:
Sort the posts from most cool to least cool based on score (tiebreak by likes + rts descending, then recent first).
Output the sorted list of IDs in order.
[/action]

Output ONLY in JSON: {"analyses": [{id: str, score: int, reason: str}, ...], "ranked_ids": [str, str, ...]}
"""

# Prompt for re-ranking trends
TRENDS_RERANK_PROMPT = """
You are an advanced AI trends ranker. Your task is to re-rank a list of X trending topics based on 'cool factor', which includes:
- Originality and novelty
- Engagement potential (humor, insight, controversy)
- Relevance to current events (tech, AI, quantum, etc.)
- Positive impact or inspiration
- Overall appeal to a tech-savvy audience

Input: A list of trends in JSON-like format: [{id: str, name: str, volume: int}, ...]

[action] Analyze each trend individually:
For each trend, output:
- Coolness score: 1-10
- Reason: Short explanation
[/action]

[action] Then, re-rank the trends:
Sort the trends from most cool to least cool based on score (tiebreak by volume descending).
Output the sorted list of IDs in order.
[/action]

Output ONLY in JSON: {"analyses": [{id: str, score: int, reason: str}, ...], "ranked_ids": [str, str, ...]}
"""

async def get_oauth2_tokens() -> Tuple[str, str]:
    """Perform OAuth 2.0 PKCE flow to get access and refresh tokens."""
    code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("utf-8").rstrip("=")
    code_challenge = base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode("utf-8")).digest()).decode("utf-8").rstrip("=")
    state = secrets.token_hex(16)

    auth_params = {
        "response_type": "code",
        "client_id": TWITTER_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": "tweet.read users.read offline.access",
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    auth_url = "https://twitter.com/i/oauth2/authorize?" + urllib.parse.urlencode(auth_params)

    print("Opening browser for authorization...")
    webbrowser.open(auth_url)

    try:
        redirect_url = input("After authorizing, copy the full redirect URL from the browser and paste here: ")
    except EOFError as exc:
        raise RuntimeError("OAuth input unavailable in non-interactive mode") from exc

    parsed_url = urllib.parse.urlparse(redirect_url)
    query_params = urllib.parse.parse_qs(parsed_url.query)

    if "state" not in query_params or query_params["state"][0] != state:
        raise ValueError("State mismatch - possible CSRF attack.")
    if "code" not in query_params:
        raise ValueError("No authorization code found.")

    code = query_params["code"][0]

    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            "https://api.twitter.com/2/oauth2/token",
            auth=(TWITTER_CLIENT_ID, ""),  # No secret for PKCE
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": REDIRECT_URI,
                "code_verifier": code_verifier,
            },
        )
        token_response.raise_for_status()
        tokens = token_response.json()

    access_token = tokens["access_token"]
    refresh_token = tokens.get("refresh_token", "")

    with open(".env", "a") as f:
        f.write(f"\nTWITTER_ACCESS_TOKEN={access_token}\n")
        if refresh_token:
            f.write(f"TWITTER_REFRESH_TOKEN={refresh_token}\n")

    return access_token, refresh_token


async def refresh_access_token(refresh_token: str) -> str:
    """Refresh the access token using refresh token."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.twitter.com/2/oauth2/token",
            auth=(TWITTER_CLIENT_ID, ""),
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
        )
        response.raise_for_status()
        tokens = response.json()

    new_access_token = tokens["access_token"]
    new_refresh_token = tokens.get("refresh_token", refresh_token)

    env_path = ".env"
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()

    saw_access = False
    saw_refresh = False
    with open(env_path, "w") as f:
        for line in lines:
            if line.startswith("TWITTER_ACCESS_TOKEN="):
                f.write(f"TWITTER_ACCESS_TOKEN={new_access_token}\n")
                saw_access = True
            elif line.startswith("TWITTER_REFRESH_TOKEN="):
                f.write(f"TWITTER_REFRESH_TOKEN={new_refresh_token}\n")
                saw_refresh = True
            else:
                f.write(line)
        if not saw_access:
            f.write(f"TWITTER_ACCESS_TOKEN={new_access_token}\n")
        if not saw_refresh:
            f.write(f"TWITTER_REFRESH_TOKEN={new_refresh_token}\n")

    return new_access_token


async def get_access_token() -> str:
    global TWITTER_ACCESS_TOKEN, TWITTER_REFRESH_TOKEN

    if TWITTER_ACCESS_TOKEN:
        return TWITTER_ACCESS_TOKEN

    if TWITTER_REFRESH_TOKEN:
        try:
            TWITTER_ACCESS_TOKEN = await refresh_access_token(TWITTER_REFRESH_TOKEN)
            return TWITTER_ACCESS_TOKEN
        except Exception as exc:
            logging.warning("Refresh token flow failed; falling back to OAuth prompt: %s", exc)

    try:
        TWITTER_ACCESS_TOKEN, TWITTER_REFRESH_TOKEN = await get_oauth2_tokens()
        return TWITTER_ACCESS_TOKEN
    except Exception as exc:
        logging.warning("OAuth flow unavailable; falling back to bearer token: %s", exc)
        return TWITTER_BEARER_TOKEN


class PostItem(ListItem):
    """A list item for a post."""

    def __init__(self, text: str, created_at: str, meta: str = "", coolness: int = 5) -> None:
        super().__init__()
        self.text = text
        self.created_at = created_at
        self.meta = meta
        self.coolness = coolness
        self.rgb = get_rgb_from_coolness(coolness)

    def compose(self) -> ComposeResult:
        yield Label(f"[bold]{self.created_at}[/bold] {self.meta}", id="meta")
        yield Label(self.text, id="text")

class TrendItem(ListItem):
    """A list item for a trend."""

    def __init__(self, name: str, meta: str = "", coolness: int = 5) -> None:
        super().__init__()
        self.name = name
        self.meta = meta
        self.coolness = coolness
        self.rgb = get_rgb_from_coolness(coolness)

    def compose(self) -> ComposeResult:
        yield Label(f"[bold]{self.name}[/bold] {self.meta}", id="meta")

class NodeItem(ListItem):
    def __init__(self, node: HFNode) -> None:
        super().__init__()
        self.node = node
    def compose(self) -> ComposeResult:
        n = self.node
        drift_vals = [hp.score for hp in n.history[-12:]]
        drift = spark(drift_vals)
        flags = ", ".join(n.flags[:4]) + ("…" if len(n.flags) > 4 else "")
        axes = " ".join([f"{k}:{getattr(n.vec, k):.2f}" for k in HF_AXES])
        yield Label(f"[bold]@{n.name}[/bold] type={n.node_type} HF={n.score:.2f} conf={n.confidence:.2f} risk={n.risk_score:.2f} ent={n.ent_bits:.3f}b I={n.mutual_bits:.3f}b")
        yield Label(axes)
        yield Label(f"drift:{drift} flags:{flags}")
        if n.reasoning:
            yield Label(f"note:{n.reasoning}")
    def on_mount(self) -> None:
        n = self.node
        self.styles.background = f"rgb{n.rgb}"

class CarouselContainer(Vertical):
    """Container for carousel display."""

    def compose(self) -> ComposeResult:
        yield Label("Carousel Mode", id="title")
        yield Container(id="current_post")

class TimelineApp(App):
    """TUI app to show your X feed with carousel, trends, and HeartFlow scoring."""

    CSS = """
    Screen {
        layout: vertical;
    }
    ListView {
        height: 1fr;
        overflow-y: auto;
    }
    #status {
        height: auto;
        dock: bottom;
    }
    Label#text {
        width: 100%;
        content-align: left top;
    }
    ListItem {
        background: $panel;
        border: solid $accent;
        padding: 1;
    }
    ListItem:hover {
        background: $boost;
    }
    CarouselContainer {
        height: 1fr;
        align: center middle;
    }
    #current_post {
        width: 80%;
        height: 80%;
        border: solid white;
        padding: 2;
    }
    #title {
        dock: top;
        background: $primary;
        color: white;
        text-align: center;
    }
    #controls {
        height: auto;
        padding: 1;
    }
    #controls Input {
        width: 1fr;
    }
    #controls Button {
        width: auto;
        margin-left: 1;
    }
    #statusline {
        height: auto;
        padding: 0 1;
    }
    Static {
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }
    TabbedContent {
        height: 1fr;
    }
    """

    def __init__(self):
        super().__init__()
        self.next_token: Optional[str] = None
        self.is_loading = False
        self.access_token: str = ""
        self.ranked_posts = []  # List of PostItem
        self.ranked_trends = []  # List of TrendItem
        self.carousel_timer: Optional[Timer] = None
        self.carousel_index = 0
        self.carousel_container: Optional[CarouselContainer] = None
        self.x = XClient(TWITTER_BEARER_TOKEN, BASE_URL, HF_REQUEST_TIMEOUT)
        self.hf = HeartFlowEngine(OPENAI_API_KEY, HF_OPENAI_MODEL)
        self.nodes: Dict[str, HFNode] = {}
        self.authors: Dict[str, str] = {}  # username to user_id
        self.trend_names: List[str] = []  # list of trend names
        self.log_lines: List[str] = []
        self._lock = asyncio.Lock()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="controls"):
            yield Input(placeholder="X usernames: alice,bob OR one per line. Optional :type (person/dao/org/ai). Example: @foo:dao", id="user_input")
            with Horizontal():
                yield Button("Score Batch", id="score_batch")
                yield Button("Score Feed Authors", id="score_authors")
                yield Button("Score Trends", id="score_trends")
                yield Button("Refresh", id="refresh")
                yield Button("Load More", id="more", disabled=True)
                yield Button("Refresh Trends", id="refresh_trends")
                yield Button("Start Carousel", id="carousel")
                yield Button("Clear Nodes", id="clear_nodes")
                yield Button("Quit", id="quit")
        yield Label("Ready.", id="statusline")
        with TabbedContent():
            with TabPane("Posts", id="tab_posts"):
                yield ListView(id="posts")
            with TabPane("Trends", id="tab_trends"):
                yield ListView(id="trends")
            with TabPane("HeartFlow Nodes", id="tab_hf_nodes"):
                yield ListView(id="hf_nodes_list")
            with TabPane("Matrix", id="tab_matrix"):
                yield Static("", id="matrix_panel")
            with TabPane("Heatmap", id="tab_heatmap"):
                yield Static("", id="heatmap_panel")
            with TabPane("Clusters", id="tab_clusters"):
                yield Static("", id="clusters_panel")
            with TabPane("Drift", id="tab_drift"):
                yield Static("", id="drift_panel")
            with TabPane("Log", id="tab_log"):
                yield Static("", id="log_panel")
        yield Footer()

    async def on_mount(self) -> None:
        try:
            self.access_token = await get_access_token()
        except Exception as exc:
            self.set_status(f"Auth error: {exc}")
            self.access_token = TWITTER_BEARER_TOKEN

        self.set_status("Loading feed and trends…")
        asyncio.create_task(self.fetch_feed(auto_carousel=False))
        asyncio.create_task(self.fetch_trends())


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

    result = score_heartflow(handle, texts)

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
