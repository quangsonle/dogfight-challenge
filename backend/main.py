import os
import time
import hmac
import json
import math
import base64
import hashlib
import asyncio
from collections import defaultdict
import numpy as np
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from dogfight_core import (
    HIT_RADIUS, BULLET_DAMAGE,
    Jet, modular_dx, extract_rl_obs, FastNumpyPolicy
)

MAX_STEPS = 2400
GAME_PASSWORD = os.getenv("GAME_PASSWORD", "pass_dogfight")
AUTH_SECRET = os.getenv("AUTH_SECRET") or hashlib.sha256(f"{GAME_PASSWORD}:df_secret_v1".encode()).hexdigest()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO = os.getenv("GITHUB_REPO", "")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
GITHUB_FILE_PATH = os.getenv("GITHUB_FILE_PATH", "data/leaderboard.json")
LOCAL_FALLBACK = "local_leaderboard.json"

app = FastAPI(title="AlphaDogfight 2D Web Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rl_policy = FastNumpyPolicy("rl_champion.npz")
lb_lock = asyncio.Lock()
leaderboard_cache = []
rate_buckets = defaultdict(list)

def get_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def check_rate_limit(key: str, max_calls: int, window_sec: int, min_cooldown_sec: int = 0):
    now = time.time()
    history = [t for t in rate_buckets[key] if now - t < window_sec]

    if min_cooldown_sec > 0 and history and (now - history[-1]) < min_cooldown_sec:
        wait = int(math.ceil(min_cooldown_sec - (now - history[-1])))
        raise HTTPException(status_code=429, detail=f"Too fast! Wait {wait}s before submitting again.")

    if len(history) >= max_calls:
        wait = int(math.ceil(window_sec - (now - history[0])))
        raise HTTPException(status_code=429, detail=f"Rate limit reached. Try again in {wait}s.")

    history.append(now)
    rate_buckets[key] = history

    if len(rate_buckets) > 1000:
        stale = [k for k, v in rate_buckets.items() if not v or (now - v[-1]) > 600]
        for k in stale:
            del rate_buckets[k]

def make_token() -> str:
    ts = str(int(time.time()))
    sig = hmac.new(AUTH_SECRET.encode(), ts.encode(), hashlib.sha256).hexdigest()
    return f"{ts}.{sig}"

def verify_token(token: str):
    if not token or "." not in token:
        raise HTTPException(status_code=401, detail="Missing or invalid pilot access token.")
    try:
        ts_str, sig = token.split(".", 1)
        ts = int(ts_str)
    except ValueError:
        raise HTTPException(status_code=401, detail="Malformed access token.")

    if time.time() - ts > 86400 * 7:
        raise HTTPException(status_code=401, detail="Session expired. Please re-enter passcode.")

    expected = hmac.new(AUTH_SECRET.encode(), ts_str.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        raise HTTPException(status_code=401, detail="Invalid pilot access token.")

def decode_action(ch: str):
    idx = "0123456789ab".find(ch)
    if idx == -1:
        raise HTTPException(status_code=400, detail="Invalid action encoding.")
    return (idx // 4) - 1, (idx // 2) % 2, idx % 2

def verify_and_score_replay(p1_actions: str, p2_actions: str):
    if len(p1_actions) != len(p2_actions) or len(p1_actions) < 10 or len(p1_actions) > MAX_STEPS:
        raise HTTPException(status_code=400, detail="Invalid action log length.")

    p1 = Jet(380.0, 220.0, math.pi * 0.5, (52, 152, 219), "PLAYER")
    p2 = Jet(820.0, 220.0, math.pi * 0.5, (231, 76, 60), "RL")
    bullets = []
    exact_matches = 0
    steps = len(p1_actions)

    for t in range(steps):
        t1, th1, f1 = decode_action(p1_actions[t])
        t2, th2, f2 = decode_action(p2_actions[t])

        # 1. P1 updates FIRST (matches docs/game.js execution order!)
        p1.update(t1, th1, f1, bullets)

        # 2. P2 observes updated (p2, p1, bullets) and verifies logits
        obs = np.clip(np.array(extract_rl_obs(p2, p1, bullets), dtype=np.float32), -2.5, 2.5)
        h1 = np.tanh(obs @ rl_policy.w1 + rl_policy.b1)
        h2 = np.tanh(h1 @ rl_policy.w2 + rl_policy.b2)
        logits = h2 @ rl_policy.w3 + rl_policy.b3
        l_turn, l_thr, l_fire = logits[0:3], logits[3:5], logits[5:7]

        gap = max(
            float(np.max(l_turn) - l_turn[t2 + 1]),
            float(np.max(l_thr) - l_thr[th2]),
            float(np.max(l_fire) - l_fire[f2]),
        )
        if gap > 0.25:
            raise HTTPException(status_code=400, detail=f"Anti-cheat: AI divergence at frame {t} (gap={gap:.2f}).")

        if int(np.argmax(l_turn)) == (t2 + 1) and int(np.argmax(l_thr)) == th2 and int(np.argmax(l_fire)) == f2:
            exact_matches += 1

        # 3. P2 updates
        p2.update(t2, th2, f2, bullets)

        surviving = []
        for b in bullets:
            if not b.update():
                continue
            hit = False
            for target in [p1, p2]:
                if b.owner is not target and target.alive:
                    if math.hypot(modular_dx(b.x, target.x), target.y - b.y) < HIT_RADIUS:
                        target.hp = max(0, target.hp - BULLET_DAMAGE)
                        hit = True
                        if target.hp == 0:
                            target.alive = False
                        break
            if not hit:
                surviving.append(b)
        bullets = surviving

        if not p1.alive or not p2.alive:
            if t < steps - 1:
                raise HTTPException(status_code=400, detail="Invalid trailing actions after KO.")
            break

    if (exact_matches / max(1, steps)) < 0.85:
        raise HTTPException(status_code=400, detail="Anti-cheat: Replay failed AI verification.")

    won = (p1.alive and not p2.alive) or (p1.alive and p2.alive and p1.hp > p2.hp)
    dmg_dealt = 100 - p2.hp
    if won:
        speed_bonus = max(0, (MAX_STEPS - steps) // 5)
        score = 1000 + (p1.hp * 10) + speed_bonus
    else:
        score = (dmg_dealt * 5) + (steps // 60)

    return {"won": bool(won), "score": int(score), "dmg_dealt": int(dmg_dealt), "hp_left": int(p1.hp)}

async def load_leaderboard():
    global leaderboard_cache
    if GITHUB_TOKEN and GITHUB_REPO:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}?ref={GITHUB_BRANCH}"
        headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, headers=headers)
            if r.status_code == 200:
                leaderboard_cache = json.loads(base64.b64decode(r.json()["content"]).decode("utf-8"))
                return
    if os.path.exists(LOCAL_FALLBACK):
        with open(LOCAL_FALLBACK, "r") as f:
            leaderboard_cache = json.load(f)

async def save_leaderboard():
    payload_str = json.dumps(leaderboard_cache, indent=2)
    if GITHUB_TOKEN and GITHUB_REPO:
        base_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"
        headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
        async with httpx.AsyncClient(timeout=10.0) as client:
            for _ in range(2):
                r_get = await client.get(f"{base_url}?ref={GITHUB_BRANCH}", headers=headers)
                sha = r_get.json()["sha"] if r_get.status_code == 200 else None
                body = {
                    "message": "Update verified dogfight leaderboard",
                    "content": base64.b64encode(payload_str.encode("utf-8")).decode("utf-8"),
                    "branch": GITHUB_BRANCH,
                }
                if sha:
                    body["sha"] = sha
                r_put = await client.put(base_url, headers=headers, json=body)
                if r_put.status_code in (200, 201):
                    return
    with open(LOCAL_FALLBACK, "w") as f:
        f.write(payload_str)

@app.on_event("startup")
async def on_startup():
    await load_leaderboard()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/unlock")
async def unlock_game(request: Request):
    ip = get_client_ip(request)
    check_rate_limit(f"unlock:{ip}", max_calls=8, window_sec=60)

    try:
        data = json.loads((await request.body()).decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid request format.")

    pw = str(data.get("password", ""))
    if not pw or not hmac.compare_digest(pw, GAME_PASSWORD):
        raise HTTPException(status_code=401, detail="Invalid pilot passcode.")

    return {"token": make_token()}

@app.get("/leaderboard")
def get_leaderboard(request: Request, token: str = ""):
    verify_token(token)
    ip = get_client_ip(request)
    check_rate_limit(f"lb:{ip}", max_calls=30, window_sec=60)
    return {"entries": leaderboard_cache[:50]}

@app.post("/submit")
async def submit_score(request: Request):
    ip = get_client_ip(request)
    check_rate_limit(f"submit:{ip}", max_calls=8, window_sec=600, min_cooldown_sec=15)

    try:
        data = json.loads((await request.body()).decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid request format.")

    verify_token(str(data.get("token", "")))

    name = str(data.get("name", "")).strip()
    if not name or len(name) > 28:
        raise HTTPException(status_code=400, detail="Name must be 1–28 characters.")

    li = str(data.get("linkedin_url", "")).strip()
    if len(li) > 120 or (li and not (li.startswith("https://") or li.startswith("http://"))):
        raise HTTPException(status_code=400, detail="LinkedIn URL must start with https://")

    p1_actions = str(data.get("p1_actions", ""))
    p2_actions = str(data.get("p2_actions", ""))

    result = verify_and_score_replay(p1_actions, p2_actions)
    entry = {
        "name": name,
        "linkedin_url": li,
        "won": result["won"],
        "score": result["score"],
        "dmg_dealt": result["dmg_dealt"],
        "hp_left": result["hp_left"],
    }

    async with lb_lock:
        leaderboard_cache.append(entry)
        leaderboard_cache.sort(key=lambda x: (x["won"], x["score"]), reverse=True)
        del leaderboard_cache[50:]
        await save_leaderboard()

    return result
