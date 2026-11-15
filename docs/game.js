// =====================================================================
// CONFIGURATION: REPLACE WITH YOUR LIVE RENDER HTTPS URL (NO TRAILING SLASH!)
// =====================================================================
const RAW_API_URL = "https://YOUR-RENDER-SERVICE.onrender.com";
const API_BASE_URL = RAW_API_URL.replace(/\/+$/, "");

// =====================================================================
// EXACT AERODYNAMIC CONSTANTS (1:1 MATCH WITH dogfight_core.py)
// =====================================================================
const WIDTH = 1200;
const HEIGHT = 700;
const GROUND_Y = 45.0;
const CEILING_Y = HEIGHT - 45.0;

const GRAVITY = 0.13;
const THRUST = 0.27;
const TURN_RATE = 0.065;
const V_STALL = 2.8;
const V_CRUISE_LIFT = 4.3;
const V_MAX = 14.5;
const PARASITIC_DRAG = 0.0018;
const INDUCED_DRAG = 0.024;

const BULLET_SPEED = 14.5;
const BULLET_LIFE = 36;
const FIRE_COOLDOWN = 6;
const MAX_HEAT = 100.0;
const HEAT_PER_SHOT = 9.0;
const COOL_RATE = 0.90;
const BULLET_DAMAGE = 15;
const HIT_RADIUS = 18.0;
const MAX_MATCH_STEPS = 2400;

// =====================================================================
// MATH HELPERS
// =====================================================================
function modX(x) { return ((x % WIDTH) + WIDTH) % WIDTH; }
function wrapAngle(a) {
  let r = (a + Math.PI) % (2.0 * Math.PI);
  if (r < 0) r += 2.0 * Math.PI;
  return r - Math.PI;
}
function modularDx(x1, x2) {
  let dx = x2 - x1;
  if (dx > WIDTH / 2) dx -= WIDTH;
  else if (dx < -WIDTH / 2) dx += WIDTH;
  return dx;
}
function rotatePt(px, py, angle) {
  const ca = Math.cos(angle), sa = Math.sin(angle);
  return [px * ca - py * sa, px * sa + py * ca];
}
function encodeAction(turn, thr, fire) {
  const code = (turn + 1) * 4 + thr * 2 + fire;
  return "0123456789ab"[code];
}

// =====================================================================
// AUTHENTICATION, SERVER PRE-WARM & WEIGHT LOADING (ZERO PREFLIGHT)
// =====================================================================
let authToken = "";
let rlWeights = null;

async function wakeAndCheckHealth() {
  const healthEl = document.getElementById("serverHealth");
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (res.ok) {
      healthEl.style.color = "#6ee7b7";
      healthEl.textContent = "Verification server online";
    } else {
      healthEl.style.color = "#fbbf24";
      healthEl.textContent = `Server waking (${res.status})...`;
    }
  } catch (err) {
    healthEl.style.color = "#f87171";
    healthEl.textContent = `Cannot reach server (${API_BASE_URL})`;
  }
}

async function ensureWeightsLoaded() {
  if (rlWeights) return;
  const res = await fetch("rl_weights.json?v=4");
  if (!res.ok) throw new Error("Failed to load neural weights file.");
  rlWeights = await res.json();
}

async function onUnlocked(token) {
  authToken = token;
  const status = document.getElementById("authStatus");
  status.style.color = "#fbbf24";
  status.textContent = "Loading 54M-step neural weights...";

  await ensureWeightsLoaded();
  try { sessionStorage.setItem("df_token", token); } catch {}

  document.getElementById("authOverlay").classList.add("hidden");
  document.getElementById("menuOverlay").classList.remove("hidden");
  fetchLeaderboard();
}

function forceRelock(message) {
  authToken = "";
  try { sessionStorage.removeItem("df_token"); } catch {}
  appState = "LOCKED";
  document.getElementById("menuOverlay").classList.add("hidden");
  document.getElementById("gameOverOverlay").classList.add("hidden");
  document.getElementById("authOverlay").classList.remove("hidden");
  document.getElementById("authStatus").style.color = "#f87171";
  document.getElementById("authStatus").textContent = message;
}

document.getElementById("authForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = document.getElementById("btnUnlock");
  const status = document.getElementById("authStatus");
  const pw = document.getElementById("passcodeInput").value;
  btn.disabled = true;
  status.style.color = "#fbbf24";
  status.textContent = "Verifying passcode...";

  try {
    let res = null;
    for (let attempt = 1; attempt <= 3; attempt++) {
      try {
        // Content-Type: text/plain avoids CORS OPTIONS preflight completely!
        res = await fetch(`${API_BASE_URL}/unlock`, {
          method: "POST",
          headers: { "Content-Type": "text/plain" },
          body: JSON.stringify({ password: pw })
        });
        break;
      } catch (netErr) {
        if (attempt === 3) throw netErr;
        status.textContent = `Waking server, retrying (${attempt + 1}/3)...`;
        await new Promise(r => setTimeout(r, 4000));
      }
    }
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Access denied");
    await onUnlocked(data.token);
  } catch (err) {
    status.style.color = "#f87171";
    status.textContent = `${err.message} (${API_BASE_URL})`;
  } finally {
    btn.disabled = false;
  }
});

// =====================================================================
// MOBILE ORIENTATION GUARD
// =====================================================================
function checkOrientation() {
  const isTouchDevice = ("ontouchstart" in window) || navigator.maxTouchPoints > 0;
  const isPortrait = window.innerHeight > window.innerWidth;
  const rotEl = document.getElementById("rotateOverlay");

  if (isTouchDevice && isPortrait && window.innerWidth < 900) {
    rotEl.classList.remove("hidden");
  } else {
    rotEl.classList.add("hidden");
  }
}
window.addEventListener("resize", checkOrientation);
window.addEventListener("orientationchange", checkOrientation);

// =====================================================================
// 256x256 PURE JS NEURAL NETWORK INFERENCE
// =====================================================================
function rlPredict(jet, opp, bullets) {
  if (!rlWeights) return [0, 1, 0];
  const obs = extractRlObs(jet, opp, bullets);
  const { w1, b1, w2, b2, w3, b3 } = rlWeights;

  const h1 = new Float64Array(256);
  for (let j = 0; j < 256; j++) {
    let sum = b1[j];
    for (let i = 0; i < 28; i++) sum += obs[i] * w1[i][j];
    h1[j] = Math.tanh(sum);
  }
  const h2 = new Float64Array(256);
  for (let j = 0; j < 256; j++) {
    let sum = b2[j];
    for (let i = 0; i < 256; i++) sum += h1[i] * w2[i][j];
    h2[j] = Math.tanh(sum);
  }
  const logits = new Float64Array(7);
  for (let j = 0; j < 7; j++) {
    let sum = b3[j];
    for (let i = 0; i < 256; i++) sum += h2[i] * w3[i][j];
    logits[j] = sum;
  }

  let turnIdx = 0;
  if (logits[1] > logits[turnIdx]) turnIdx = 1;
  if (logits[2] > logits[turnIdx]) turnIdx = 2;
  return [turnIdx - 1, logits[4] > logits[3] ? 1 : 0, logits[6] > logits[5] ? 1 : 0];
}

function extractRlObs(jet, opp, bullets) {
  const alt_norm = 2.0 * (jet.y - GROUND_Y) / (CEILING_Y - GROUND_Y) - 1.0;
  const ca = Math.cos(jet.angle), sa = Math.sin(jet.angle);
  const spd = jet.speed();
  const vx_norm = jet.vx / V_MAX, vy_norm = jet.vy / V_MAX, spd_norm = spd / V_MAX;
  const heat_norm = jet.heat / MAX_HEAT, overheat_flag = jet.overheated ? 1.0 : 0.0, hp_norm = jet.hp / 100.0;

  const pred_y_30 = jet.y + jet.vy * 30.0 - 0.5 * GRAVITY * 900.0;
  const ground_danger = Math.max(0.0, Math.min(1.0, (GROUND_Y + 135.0 - pred_y_30) / 135.0));
  const pred_y_ceil = jet.y + jet.vy * 15.0;
  const ceiling_danger = Math.max(0.0, Math.min(1.0, (pred_y_ceil - (CEILING_Y - 50.0)) / 80.0));

  const dx = modularDx(jet.x, opp.x), dy = opp.y - jet.y, dist = Math.hypot(dx, dy);
  const rel_fwd = (dx * ca + dy * sa) / (WIDTH * 0.5);
  const rel_left = (-dx * sa + dy * ca) / (WIDTH * 0.5);

  const t_hit = Math.min(35.0, dist / BULLET_SPEED);
  const lead_dx = modularDx(jet.x, opp.x + opp.vx * t_hit - jet.vx * t_hit * 0.35);
  const lead_dy = (opp.y + opp.vy * t_hit - jet.vy * t_hit * 0.35) - jet.y;
  const lead_dist = Math.max(1.0, Math.hypot(lead_dx, lead_dy));
  const lead_dir_fwd = (lead_dx * ca + lead_dy * sa) / lead_dist;
  const lead_dir_left = (-lead_dx * sa + lead_dy * ca) / lead_dist;

  const opp_v_fwd = (opp.vx * ca + opp.vy * sa) / V_MAX;
  const opp_v_left = (-opp.vx * sa + opp.vy * ca) / V_MAX;
  const rel_ang = wrapAngle(opp.angle - jet.angle);
  const cos_rel = Math.cos(rel_ang), sin_rel = Math.sin(rel_ang);

  const dist_norm = Math.min(1.0, dist / 800.0), opp_hp_norm = opp.hp / 100.0;
  const opp_to_me_ang = Math.atan2(-dy, -dx);
  const opp_aim_cos = Math.cos(wrapAngle(opp_to_me_ang - opp.angle));
  const opp_overheat = opp.overheated ? 1.0 : 0.0;

  let b_fwd = 0.0, b_left = 0.0, b_vfwd = 0.0, b_vleft = 0.0, b_threat = 0.0;
  let min_b_dist = 750.0;
  for (const b of bullets) {
    if (b.owner !== jet) {
      const bdx = modularDx(jet.x, b.x), bdy = b.y - jet.y, bdist = Math.hypot(bdx, bdy);
      if (bdist < min_b_dist) {
        min_b_dist = bdist;
        b_fwd = (bdx * ca + bdy * sa) / 750.0;
        b_left = (-bdx * sa + bdy * ca) / 750.0;
        b_vfwd = ((b.vx - jet.vx) * ca + (b.vy - jet.vy) * sa) / (BULLET_SPEED + V_MAX);
        b_vleft = (-(b.vx - jet.vx) * sa + (b.vy - jet.vy) * ca) / (BULLET_SPEED + V_MAX);
        b_threat = 1.0 - (bdist / 750.0);
      }
    }
  }

  return [
    alt_norm, ca, sa, vx_norm, vy_norm, spd_norm,
    heat_norm, overheat_flag, hp_norm,
    ground_danger, ceiling_danger,
    rel_fwd, rel_left, lead_dir_fwd, lead_dir_left,
    opp_v_fwd, opp_v_left, cos_rel, sin_rel,
    dist_norm, opp_hp_norm, opp_aim_cos, opp_overheat,
    b_fwd, b_left, b_vfwd, b_vleft, b_threat
  ].map(v => Math.max(-2.5, Math.min(2.5, v)));
}

// =====================================================================
// HARD RULE-BASED BOT (MODE 2)
// =====================================================================
function hardNaiveAiControl(ai, opp) {
  const speed = ai.speed();
  const dx = modularDx(ai.x, opp.x), dy = opp.y - ai.y, dist = Math.hypot(dx, dy);

  const pred_y = ai.y + ai.vy * 28.0 - 0.5 * GRAVITY * 784.0;
  if (pred_y < GROUND_Y + 95.0 || (ai.y < GROUND_Y + 145.0 && ai.vy < -1.5)) {
    const err = wrapAngle(Math.PI * 0.5 - ai.angle);
    return [err > 0.04 ? 1 : (err < -0.04 ? -1 : 0), 1, 0];
  }
  if (ai.y > CEILING_Y - 65.0 && ai.vy > 0.5) {
    const err = wrapAngle((ai.vx >= 0 ? -0.35 : -Math.PI + 0.35) - ai.angle);
    return [err > 0.04 ? 1 : (err < -0.04 ? -1 : 0), 0, 0];
  }
  if (speed < V_STALL + 0.9 && ai.y > GROUND_Y + 190.0) {
    const err = wrapAngle((Math.cos(ai.angle) >= 0 ? -0.35 : -Math.PI + 0.35) - ai.angle);
    return [err > 0.04 ? 1 : (err < -0.04 ? -1 : 0), 1, 0];
  }
  const opp_aim_err = Math.abs(wrapAngle(opp.angle - Math.atan2(-dy, -dx)));
  if (dist < 350.0 && opp_aim_err < 0.32 && Math.abs(wrapAngle(ai.angle - opp.angle)) < 0.85) {
    return [ai.y < HEIGHT * 0.65 ? 1 : -1, 1, 0];
  }

  const t_hit = Math.min(35.0, dist / BULLET_SPEED);
  const lead_dx = modularDx(ai.x, opp.x + opp.vx * t_hit - ai.vx * t_hit * 0.35);
  const lead_dy = (opp.y + opp.vy * t_hit - ai.vy * t_hit * 0.35) - ai.y;
  const angle_err = wrapAngle(Math.atan2(lead_dy, lead_dx) - ai.angle);
  const turn = angle_err > 0.035 ? 1 : (angle_err < -0.035 ? -1 : 0);
  const throttle = (Math.abs(angle_err) > 1.2 && speed > 8.5) ? 0 : 1;
  const fire = (Math.abs(angle_err) < 0.14 && dist < 560.0 && !ai.overheated && ai.heat < 88.0) ? 1 : 0;
  return [turn, throttle, fire];
}

// =====================================================================
// ENTITIES & SIMULATION (EXACT MATCH WITH dogfight_core.py)
// =====================================================================
class Jet {
  constructor(x, y, angle, color, name) {
    this.x = x; this.y = y; this.angle = angle;
    this.vx = 6.0 * Math.cos(angle);
    this.vy = 6.0 * Math.sin(angle);
    this.color = color; this.name = name;
    this.hp = 100; this.alive = true; this.crashed = false;
    this.fireTimer = 0; this.heat = 0.0; this.overheated = false;
  }
  speed() { return Math.hypot(this.vx, this.vy); }
  update(turnInput, throttleInput, fireInput, bullets, particles, safeFloor) {
    if (!this.alive) return false;
    let fired = false;
    const speed = this.speed();

    if (speed >= V_STALL) {
      this.angle = wrapAngle(this.angle + turnInput * TURN_RATE);
    } else {
      this.angle = wrapAngle(this.angle + turnInput * TURN_RATE * 0.15);
      const downErr = wrapAngle(-Math.PI * 0.5 - this.angle);
      this.angle = wrapAngle(this.angle + downErr * 0.04);
    }

    const velAngle = speed > 0.05 ? Math.atan2(this.vy, this.vx) : this.angle;
    if (speed > 0.05) {
      const alpha = wrapAngle(this.angle - velAngle);
      const grip = speed >= V_STALL ? 0.18 * Math.min(1.0, speed / 5.0) : 0.02;
      const newVelAngle = wrapAngle(velAngle + alpha * grip);
      const dragI = INDUCED_DRAG * (Math.abs(Math.sin(alpha)) ** 1.5) * speed;
      const dragP = PARASITIC_DRAG * (speed * speed);
      const newSpeed = Math.max(0.0, speed - dragI - dragP);
      this.vx = newSpeed * Math.cos(newVelAngle);
      this.vy = newSpeed * Math.sin(newVelAngle);

      if (Math.abs(alpha) > 0.25 && speed > 5.0 && Math.random() < 0.6) {
        particles.push({ x: this.x, y: this.y, vx: 0, vy: 0, color: "rgba(180,200,220,", life: 16, maxLife: 16, r: 2 });
      }
    }

    const airDensity = Math.max(0.05, Math.min(1.0, (CEILING_Y - this.y + 30.0) / 90.0));
    if (throttleInput) {
      this.vx += THRUST * airDensity * Math.cos(this.angle);
      this.vy += THRUST * airDensity * Math.sin(this.angle);
      particles.push({
        x: this.x - 16 * Math.cos(this.angle), y: this.y - 16 * Math.sin(this.angle),
        vx: this.vx - 3.5 * Math.cos(this.angle) + (Math.random() - 0.5) * 1.2,
        vy: this.vy - 3.5 * Math.sin(this.angle) + (Math.random() - 0.5) * 1.2,
        color: "rgba(255,160,40,", life: 10, maxLife: 10, r: 3
      });
    }

    const wingLift = GRAVITY * airDensity * Math.min(1.0, (speed / V_CRUISE_LIFT) ** 2) * Math.abs(Math.cos(velAngle));
    this.vy -= (GRAVITY - wingLift);

    const finalSpd = this.speed();
    if (finalSpd > V_MAX) {
      const s = V_MAX / finalSpd;
      this.vx *= s; this.vy *= s;
    }

    this.x = modX(this.x + this.vx);
    this.y += this.vy;

    if (this.y > CEILING_Y) {
      this.vy -= GRAVITY * 1.4;
      if (this.y > HEIGHT - 5.0) { this.y = HEIGHT - 5.0; this.vy = Math.min(this.vy, -0.5); }
    }

    if (this.y <= GROUND_Y) {
      if (safeFloor) {
        this.y = GROUND_Y + 5.0;
        this.vy = Math.max(3.5, Math.abs(this.vy) * 0.6);
        if (Math.sin(this.angle) < 0) this.angle = Math.cos(this.angle) >= 0 ? Math.PI * 0.22 : Math.PI * 0.78;
        spawnBurst(this.x, this.y, particles, "rgba(100,220,255,", 8);
      } else {
        this.y = GROUND_Y; this.hp = 0; this.alive = false; this.crashed = true;
        spawnBurst(this.x, this.y, particles, "rgba(255,120,40,", 40);
        return false;
      }
    }

    if (this.fireTimer > 0) this.fireTimer--;
    this.heat = Math.max(0.0, this.heat - COOL_RATE);
    if (this.overheated && this.heat < 25.0) this.overheated = false;

    if (fireInput && this.fireTimer === 0 && !this.overheated) {
      this.fireTimer = FIRE_COOLDOWN;
      this.heat += HEAT_PER_SHOT;
      if (this.heat >= MAX_HEAT) { this.heat = MAX_HEAT; this.overheated = true; }
      bullets.push({
        x: modX(this.x + 20 * Math.cos(this.angle)),
        y: this.y + 20 * Math.sin(this.angle),
        vx: this.vx + BULLET_SPEED * Math.cos(this.angle),
        vy: this.vy + BULLET_SPEED * Math.sin(this.angle),
        color: this.color, owner: this, life: BULLET_LIFE
      });
      fired = true;
    }

    if (this.hp <= 40 && Math.random() < 0.4) {
      particles.push({
        x: this.x, y: this.y,
        vx: (Math.random() - 0.5) * 0.8, vy: 0.2 + Math.random() * 0.6,
        color: "rgba(110,110,120,", life: 25, maxLife: 25, r: 4
      });
    }
    return fired;
  }
}

function spawnBurst(x, y, particles, rgbaPrefix, count) {
  for (let i = 0; i < count; i++) {
    const a = Math.random() * 2 * Math.PI, s = 1 + Math.random() * 5.5, life = 15 + Math.floor(Math.random() * 20);
    particles.push({ x, y, vx: s * Math.cos(a), vy: s * Math.sin(a), color: rgbaPrefix, life, maxLife: life, r: 2 + Math.floor(Math.random() * 3) });
  }
}

const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");

let appState = "LOCKED";
let gameMode = 3;
let waitingLaunch = true;
let mouseSteering = true;

let p1 = null, p2 = null, targetObj = null;
let bullets = [], particles = [];
let targetsHit = 0, shotsFired = 0, roundSteps = 0;
let p1ActionLog = "", p2ActionLog = "";
let lastMatchResult = null;

const keys = {};
let mouseX = 600, mouseY = 350, mouseLeft = false, mouseRight = false;

let isUsingTouch = false;
let lastTouchTime = 0;
let stickTouchId = null, stickOrigX = 0, stickOrigY = 0, stickCurrX = 0, stickCurrY = 0;
let fireTouchId = null, touchFire = false;

function respawnTarget() {
  for (let i = 0; i < 50; i++) {
    const tx = 100 + Math.random() * (WIDTH - 200);
    const ty = GROUND_Y + 110 + Math.random() * (CEILING_Y - GROUND_Y - 190);
    if (Math.hypot(modularDx(p1.x, tx), ty - p1.y) > 260) {
      targetObj = { x: tx, y: ty, r: 22 };
      return;
    }
  }
  targetObj = { x: modX(p1.x + WIDTH / 2), y: HEIGHT * 0.55, r: 22 };
}

function startMatch(mode) {
  if (!authToken) return;
  gameMode = mode;
  appState = "PLAY";
  waitingLaunch = true;
  bullets = []; particles = [];
  targetsHit = 0; shotsFired = 0; roundSteps = 0;
  p1ActionLog = ""; p2ActionLog = "";
  stickTouchId = null; fireTouchId = null; touchFire = false;

  p1 = new Jet(380, 220, Math.PI * 0.5, "#3498db", "YOU (BLUE)");
  if (mode === 1) {
    p2 = null;
    respawnTarget();
  } else {
    targetObj = null;
    p2 = new Jet(820, 220, Math.PI * 0.5, "#e74c3c", mode === 2 ? "HARD RULE-BOT" : "RL CHAMPION (54M)");
  }

  document.getElementById("menuOverlay").classList.add("hidden");
  document.getElementById("gameOverOverlay").classList.add("hidden");
  document.getElementById("btnMenu").classList.remove("hidden");
}

function showMenu() {
  if (!authToken) return;
  appState = "MENU";
  stickTouchId = null; fireTouchId = null; touchFire = false;
  document.getElementById("menuOverlay").classList.remove("hidden");
  document.getElementById("gameOverOverlay").classList.add("hidden");
  document.getElementById("btnMenu").classList.add("hidden");
}

function stepSimulation() {
  if (appState !== "PLAY" || waitingLaunch) {
    for (let i = particles.length - 1; i >= 0; i--) {
      particles[i].x = modX(particles[i].x + particles[i].vx);
      particles[i].y += particles[i].vy;
      if (--particles[i].life <= 0) particles.splice(i, 1);
    }
    return;
  }

  roundSteps++;
  let p1Turn = 0, p1Thr = 0, p1Fire = 0;

  if (isUsingTouch) {
    if (stickTouchId !== null) {
      const sdx = stickCurrX - stickOrigX;
      const sdy = stickOrigY - stickCurrY;
      const sdist = Math.hypot(sdx, sdy);
      if (sdist > 14) {
        const err = wrapAngle(Math.atan2(sdy, sdx) - p1.angle);
        p1Turn = err > 0.06 ? 1 : (err < -0.06 ? -1 : 0);
        p1Thr = sdist >= 38 ? 1 : 0;
      }
    }
    p1Fire = touchFire ? 1 : 0;
  } else if (mouseSteering) {
    const err = wrapAngle(Math.atan2((HEIGHT - mouseY) - p1.y, mouseX - p1.x) - p1.angle);
    p1Turn = err > 0.06 ? 1 : (err < -0.06 ? -1 : 0);
    p1Thr = (mouseRight || keys["KeyW"] || keys["ArrowUp"] || keys["ShiftLeft"]) ? 1 : 0;
    p1Fire = (mouseLeft || keys["Space"] || keys["KeyF"]) ? 1 : 0;
  } else {
    p1Turn = ((keys["ArrowLeft"] || keys["KeyA"]) ? 1 : 0) + ((keys["ArrowRight"] || keys["KeyD"]) ? -1 : 0);
    p1Thr = (keys["ArrowUp"] || keys["KeyW"]) ? 1 : 0;
    p1Fire = (keys["Space"] || keys["KeyF"]) ? 1 : 0;
  }

  p1ActionLog += encodeAction(p1Turn, p1Thr, p1Fire);
  const firedNow = p1.update(p1Turn, p1Thr, p1Fire, bullets, particles, gameMode === 1);
  if (gameMode === 1 && firedNow) shotsFired++;

  if (gameMode !== 1 && p2) {
    const [t2, th2, f2] = (gameMode === 2) ? hardNaiveAiControl(p2, p1) : rlPredict(p2, p1, bullets);
    p2ActionLog += encodeAction(t2, th2, f2);
    p2.update(t2, th2, f2, bullets, particles, false);
  }

  for (let i = bullets.length - 1; i >= 0; i--) {
    const b = bullets[i];
    b.x = modX(b.x + b.vx);
    b.y += b.vy;
    b.life--;
    if (b.life <= 0 || b.y <= GROUND_Y || b.y >= HEIGHT) { bullets.splice(i, 1); continue; }

    let hit = false;
    if (gameMode === 1 && targetObj) {
      if (Math.hypot(modularDx(b.x, targetObj.x), targetObj.y - b.y) < targetObj.r + 5) {
        targetsHit++;
        spawnBurst(targetObj.x, targetObj.y, particles, "rgba(255,200,60,", 30);
        respawnTarget();
        hit = true;
      }
    } else if (p2) {
      for (const target of [p1, p2]) {
        if (b.owner !== target && target.alive) {
          if (Math.hypot(modularDx(b.x, target.x), target.y - b.y) < HIT_RADIUS) {
            target.hp = Math.max(0, target.hp - BULLET_DAMAGE);
            spawnBurst(b.x, b.y, particles, "rgba(255,220,100,", 10);
            hit = true;
            if (target.hp === 0) {
              target.alive = false;
              spawnBurst(target.x, target.y, particles, target === p1 ? "rgba(52,152,219," : "rgba(231,76,60,", 50);
            }
            break;
          }
        }
      }
    }
    if (hit) bullets.splice(i, 1);
  }

  for (let i = particles.length - 1; i >= 0; i--) {
    particles[i].x = modX(particles[i].x + particles[i].vx);
    particles[i].y += particles[i].vy;
    if (--particles[i].life <= 0) particles.splice(i, 1);
  }

  if (gameMode !== 1 && (!p1.alive || !p2.alive || roundSteps >= MAX_MATCH_STEPS)) {
    finishMatch();
  }
}

function finishMatch() {
  appState = "GAMEOVER";
  stickTouchId = null; fireTouchId = null; touchFire = false;
  const dmgDealt = 100 - p2.hp;
  const durationSec = (roundSteps / 60).toFixed(1);
  const won = (p1.alive && !p2.alive) || (p1.alive && p2.alive && p1.hp > p2.hp);
  const score = won
    ? 1000 + (p1.hp * 10) + Math.max(0, Math.floor((MAX_MATCH_STEPS - roundSteps) / 5))
    : (dmgDealt * 5) + Math.floor(roundSteps / 60);

  lastMatchResult = { won, score, dmgDealt, hpLeft: p1.hp, durationSec };

  document.getElementById("resultHeadline").textContent = won ? "VICTORY! YOU DEFEATED THE AI!" : "DEFEATED BY THE AI!";
  document.getElementById("resultHeadline").style.color = won ? "#2ecc71" : "#f87171";
  document.getElementById("resultStats").textContent = `Time: ${durationSec}s | Damage Dealt: ${dmgDealt} HP | Your HP Left: ${p1.hp}`;
  document.getElementById("scoreDisplay").textContent = `Combat Score: ${score}`;

  const submitForm = document.getElementById("submitForm");
  if (gameMode === 3) {
    submitForm.classList.remove("hidden");
    document.getElementById("submitStatus").textContent = "";
    document.getElementById("btnSubmitScore").disabled = false;
  } else {
    submitForm.classList.add("hidden");
  }
  document.getElementById("gameOverOverlay").classList.remove("hidden");
}

function render() {
  ctx.fillStyle = "#121620";
  ctx.fillRect(0, 0, WIDTH, HEIGHT);

  ctx.strokeStyle = "#1c222e"; ctx.lineWidth = 1;
  ctx.fillStyle = "#475569"; ctx.font = "12px sans-serif";
  for (const alt of [150, 300, 450, 600]) {
    const gy = HEIGHT - alt;
    ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(WIDTH, gy); ctx.stroke();
    ctx.fillText(`${alt - GROUND_Y}m`, 8, gy - 5);
  }

  ctx.strokeStyle = "#2d374b"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(2, 0); ctx.lineTo(2, HEIGHT - GROUND_Y); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(WIDTH - 2, 0); ctx.lineTo(WIDTH - 2, HEIGHT - GROUND_Y); ctx.stroke();

  ctx.fillStyle = gameMode === 1 ? "#233746" : "#232a26";
  ctx.fillRect(0, HEIGHT - GROUND_Y, WIDTH, GROUND_Y);
  ctx.strokeStyle = gameMode === 1 ? "#46aad2" : "#50785a";
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(0, HEIGHT - GROUND_Y); ctx.lineTo(WIDTH, HEIGHT - GROUND_Y); ctx.stroke();

  if (!p1) return;

  for (const pt of particles) {
    ctx.fillStyle = `${pt.color}${(pt.life / pt.maxLife).toFixed(2)})`;
    ctx.beginPath(); ctx.arc(pt.x, HEIGHT - pt.y, pt.r, 0, Math.PI * 2); ctx.fill();
  }

  for (const b of bullets) {
    const sx = b.x, sy = HEIGHT - b.y;
    const ex = modX(b.x - b.vx * 0.7), ey = HEIGHT - (b.y - b.vy * 0.7);
    if (Math.abs(sx - ex) < 100) {
      ctx.strokeStyle = "#fff08c"; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke();
    }
    ctx.fillStyle = b.color;
    ctx.beginPath(); ctx.arc(sx, sy, 2.5, 0, Math.PI * 2); ctx.fill();
  }

  if (targetObj) {
    const tx = targetObj.x, ty = HEIGHT - targetObj.y;
    for (const [r, col] of [[targetObj.r, "#e64646"], [targetObj.r * 0.68, "#f5f5fa"], [targetObj.r * 0.36, "#e64646"], [4, "#ffd746"]]) {
      ctx.fillStyle = col;
      ctx.beginPath(); ctx.arc(tx, ty, r, 0, Math.PI * 2); ctx.fill();
    }
  }

  if (!isUsingTouch && mouseSteering && appState === "PLAY") {
    ctx.strokeStyle = "#78c8ff"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(mouseX, mouseY, 10, 0, Math.PI * 2); ctx.stroke();
  }

  drawJet(p1, gameMode === 1);
  if (p2) drawJet(p2, false);

  drawHud(p1, 20, false, gameMode === 1);
  if (p2) drawHud(p2, WIDTH - 20, true, false);

  if (isUsingTouch && appState === "PLAY") {
    drawMobileTouchOverlay();
  }

  if (appState === "PLAY" && waitingLaunch) {
    ctx.fillStyle = "rgba(22, 26, 36, 0.85)";
    ctx.fillRect(WIDTH / 2 - 270, HEIGHT / 2 - 40, 540, 68);
    ctx.strokeStyle = "#ffdc50"; ctx.lineWidth = 2;
    ctx.strokeRect(WIDTH / 2 - 270, HEIGHT / 2 - 40, 540, 68);
    ctx.fillStyle = "#ffdc50"; ctx.font = "bold 20px sans-serif"; ctx.textAlign = "center";
    ctx.fillText(isUsingTouch ? "TOUCH TO LAUNCH!" : "CLICK OR PRESS ANY KEY TO LAUNCH!", WIDTH / 2, HEIGHT / 2);
    ctx.textAlign = "left";
  }
}

function drawMobileTouchOverlay() {
  const sx = stickTouchId !== null ? stickOrigX : 140;
  const sy = stickTouchId !== null ? stickOrigY : HEIGHT - 135;
  const kx = stickTouchId !== null ? stickCurrX : sx;
  const ky = stickTouchId !== null ? stickCurrY : sy;
  const sdist = Math.hypot(kx - sx, ky - sy);
  const boosting = stickTouchId !== null && sdist >= 38;

  ctx.strokeStyle = boosting ? "rgba(255, 165, 40, 0.75)" : "rgba(120, 200, 255, 0.35)";
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.arc(sx, sy, 82, 0, Math.PI * 2); ctx.stroke();

  ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(sx, sy, 38, 0, Math.PI * 2); ctx.stroke();

  let dx = kx - sx, dy = ky - sy;
  if (sdist > 82) { dx = (dx / sdist) * 82; dy = (dy / sdist) * 82; }
  ctx.fillStyle = boosting ? "rgba(255, 165, 40, 0.7)" : "rgba(100, 200, 255, 0.6)";
  ctx.beginPath(); ctx.arc(sx + dx, sy + dy, 26, 0, Math.PI * 2); ctx.fill();

  ctx.fillStyle = "rgba(255, 255, 255, 0.55)";
  ctx.font = "bold 13px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(boosting ? "BOOST ON" : "STEER / BOOST", sx, sy + 102);

  const fx = WIDTH - 130, fy = HEIGHT - 135;
  ctx.fillStyle = touchFire ? "rgba(239, 68, 68, 0.55)" : "rgba(239, 68, 68, 0.22)";
  ctx.strokeStyle = touchFire ? "rgba(255, 120, 120, 0.9)" : "rgba(255, 120, 120, 0.45)";
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.arc(fx, fy, 64, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#fff";
  ctx.font = "bold 16px sans-serif";
  ctx.fillText("FIRE", fx, fy + 6);
  ctx.textAlign = "left";
}

function drawJet(jet, showGuides) {
  if (!jet.alive) return;
  const sx = jet.x, sy = HEIGHT - jet.y;

  if (showGuides) {
    ctx.fillStyle = "#5a8cb4";
    for (let d = 35; d < 320; d += 30) {
      const px = modX(jet.x + d * Math.cos(jet.angle)), py = jet.y + d * Math.sin(jet.angle);
      if (py > GROUND_Y && py < HEIGHT) { ctx.beginPath(); ctx.arc(px, HEIGHT - py, 2, 0, Math.PI * 2); ctx.fill(); }
    }
    const spd = jet.speed();
    if (spd > 1.0) {
      const fx = modX(jet.x + (jet.vx / spd) * 55), fy = HEIGHT - (jet.y + (jet.vy / spd) * 55);
      if (Math.abs(fx - sx) < 100) {
        ctx.strokeStyle = "#64e68c"; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(fx, fy, 5, 0, Math.PI * 2); ctx.stroke();
      }
    }
  }

  const pts = [[18, 0], [-12, 10], [-7, 0], [-12, -10]].map(([lx, ly]) => {
    const [rx, ry] = rotatePt(lx, ly, jet.angle);
    return [sx + rx, sy - ry];
  });

  ctx.fillStyle = jet.color; ctx.strokeStyle = "#f0f5ff"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath(); ctx.fill(); ctx.stroke();

  const [cx, cy] = rotatePt(5, 0, jet.angle);
  ctx.fillStyle = "#c8f0ff";
  ctx.beginPath(); ctx.arc(sx + cx, sy - cy, 3, 0, Math.PI * 2); ctx.fill();
}

function drawHud(jet, xPos, alignRight, isTraining) {
  const barW = 240, bx = alignRight ? xPos - barW : xPos;

  if (!isTraining) {
    ctx.fillStyle = "#282d37"; ctx.fillRect(bx, 18, barW, 18);
    ctx.fillStyle = jet.hp > 30 ? jet.color : "#e63c3c";
    ctx.fillRect(bx, 18, barW * Math.max(0, jet.hp) / 100, 18);
    ctx.strokeStyle = "#c8d2dc"; ctx.lineWidth = 1; ctx.strokeRect(bx, 18, barW, 18);
  }

  ctx.fillStyle = isTraining ? "#6ee6a0" : "#f0f5ff";
  ctx.font = "bold 16px sans-serif";
  ctx.textAlign = alignRight ? "right" : "left";
  ctx.fillText(isTraining ? `FLIGHT SCHOOL | Targets Hit: ${targetsHit} | Shots: ${shotsFired}` : `${jet.name}  HP: ${jet.hp}`, alignRight ? xPos : bx, isTraining ? 30 : 54);

  const spd = jet.speed();
  ctx.fillStyle = spd < V_STALL ? "#ff5050" : "#aabed2";
  ctx.font = "13px sans-serif";
  ctx.fillText(`Airspeed: ${Math.round(spd * 45)} kts | Alt: ${Math.round(jet.y - GROUND_Y)}m${spd < V_STALL ? " [STALL!]" : ""}`, alignRight ? xPos : bx, isTraining ? 52 : 74);

  const hbx = alignRight ? xPos - 140 : bx, hby = isTraining ? 64 : 86;
  ctx.fillStyle = "#282d37"; ctx.fillRect(hbx, hby, 140, 8);
  ctx.fillStyle = jet.overheated ? "#ff3c3c" : "#ffbe3c";
  ctx.fillRect(hbx, hby, 140 * (jet.heat / MAX_HEAT), 8);
  ctx.textAlign = "left";
}

// =====================================================================
// INPUTS
// =====================================================================
function toCanvasCoords(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  return [
    (clientX - rect.left) * (WIDTH / rect.width),
    (clientY - rect.top) * (HEIGHT / rect.height)
  ];
}

canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("mousemove", e => {
  if (performance.now() - lastTouchTime > 500) isUsingTouch = false;
  [mouseX, mouseY] = toCanvasCoords(e.clientX, e.clientY);
});
canvas.addEventListener("mousedown", e => {
  if (performance.now() - lastTouchTime > 500) isUsingTouch = false;
  [mouseX, mouseY] = toCanvasCoords(e.clientX, e.clientY);
  if (e.button === 0) mouseLeft = true;
  if (e.button === 2) mouseRight = true;
  if (appState === "PLAY" && waitingLaunch) waitingLaunch = false;
});
window.addEventListener("mouseup", e => {
  if (e.button === 0) mouseLeft = false;
  if (e.button === 2) mouseRight = false;
});

canvas.addEventListener("touchstart", e => {
  e.preventDefault();
  isUsingTouch = true;
  lastTouchTime = performance.now();
  if (appState === "PLAY" && waitingLaunch) waitingLaunch = false;

  for (const t of e.changedTouches) {
    const [cx, cy] = toCanvasCoords(t.clientX, t.clientY);
    if (cx < WIDTH * 0.5 && stickTouchId === null) {
      stickTouchId = t.identifier;
      stickOrigX = cx; stickOrigY = cy;
      stickCurrX = cx; stickCurrY = cy;
    } else if (cx >= WIDTH * 0.5 && fireTouchId === null) {
      fireTouchId = t.identifier;
      touchFire = true;
    }
  }
}, { passive: false });

canvas.addEventListener("touchmove", e => {
  e.preventDefault();
  lastTouchTime = performance.now();
  for (const t of e.changedTouches) {
    if (t.identifier === stickTouchId) {
      [stickCurrX, stickCurrY] = toCanvasCoords(t.clientX, t.clientY);
    }
  }
}, { passive: false });

function handleTouchRelease(e) {
  lastTouchTime = performance.now();
  for (const t of e.changedTouches) {
    if (t.identifier === stickTouchId) stickTouchId = null;
    if (t.identifier === fireTouchId) { fireTouchId = null; touchFire = false; }
  }
}
canvas.addEventListener("touchend", handleTouchRelease);
canvas.addEventListener("touchcancel", handleTouchRelease);

window.addEventListener("keydown", e => {
  if (["Space", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.code) && appState === "PLAY") e.preventDefault();
  keys[e.code] = true;
  if (appState === "PLAY" && waitingLaunch && !["KeyR", "KeyM", "KeyC"].includes(e.code)) waitingLaunch = false;
  if (e.code === "KeyM" && appState !== "MENU" && appState !== "LOCKED" && document.activeElement.tagName !== "INPUT") showMenu();
  if (e.code === "KeyR" && appState !== "MENU" && appState !== "LOCKED" && document.activeElement.tagName !== "INPUT") startMatch(gameMode);
  if (e.code === "KeyC" && appState !== "LOCKED" && document.activeElement.tagName !== "INPUT") toggleSteering();
});
window.addEventListener("keyup", e => { keys[e.code] = false; });

function toggleSteering() {
  mouseSteering = !mouseSteering;
  document.getElementById("btnToggleCtrl").textContent = mouseSteering
    ? "Controls: Mouse Fly-By-Wire [C]"
    : "Controls: Keyboard [C]";
}

document.querySelectorAll(".mode-btn").forEach(btn => btn.addEventListener("click", () => startMatch(Number(btn.dataset.mode))));
document.getElementById("btnToggleCtrl").addEventListener("click", toggleSteering);
document.getElementById("btnMenu").addEventListener("click", showMenu);
document.getElementById("btnBackMenu").addEventListener("click", showMenu);
document.getElementById("btnRematch").addEventListener("click", () => startMatch(gameMode));

document.getElementById("btnShare").addEventListener("click", () => {
  if (!lastMatchResult) return;
  const txt = lastMatchResult.won
    ? `I defeated the 54M-step Self-Play RL Fighter Pilot (Score: ${lastMatchResult.score} | ${lastMatchResult.hpLeft} HP left)! Can you beat the machine? ${window.location.href}`
    : `I dealt ${lastMatchResult.dmgDealt} HP damage against a 54M-step Self-Play RL Fighter Pilot (Score: ${lastMatchResult.score}). Think you can beat it? ${window.location.href}`;
  navigator.clipboard.writeText(txt);
  document.getElementById("btnShare").textContent = "Copied to Clipboard!";
  setTimeout(() => { document.getElementById("btnShare").textContent = "Copy LinkedIn Share Text"; }, 2000);
});

async function fetchLeaderboard() {
  if (!authToken) return;
  const listEl = document.getElementById("leaderboardList");
  try {
    const res = await fetch(`${API_BASE_URL}/leaderboard?token=${encodeURIComponent(authToken)}`);
    if (res.status === 401) { forceRelock("Session expired. Re-enter passcode."); return; }
    const data = await res.json();
    if (!data.entries || data.entries.length === 0) {
      listEl.innerHTML = '<div class="lb-loading">No verified runs yet. Be the first!</div>';
      return;
    }
    listEl.innerHTML = data.entries.map((e, idx) => {
      const nameHtml = e.linkedin_url
        ? `<a href="${escapeHtml(e.linkedin_url)}" target="_blank" rel="noopener">${escapeHtml(e.name)}</a>`
        : escapeHtml(e.name);
      const badge = e.won ? `<span class="lb-tag">SLAYER</span>` : "";
      return `<div class="lb-row ${e.won ? "slayer" : ""}">
        <div class="lb-name">#${idx + 1} ${nameHtml}${badge}</div>
        <div class="lb-score">${e.score}</div>
      </div>`;
    }).join("");
  } catch {
    listEl.innerHTML = '<div class="lb-loading">Leaderboard unreachable.</div>';
  }
}

document.getElementById("submitForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = document.getElementById("btnSubmitScore");
  const status = document.getElementById("submitStatus");
  btn.disabled = true;
  status.style.color = "#fbbf24";
  status.textContent = "Verifying physics replay on server...";

  try {
    const res = await fetch(`${API_BASE_URL}/submit`, {
      method: "POST",
      headers: { "Content-Type": "text/plain" },
      body: JSON.stringify({
        token: authToken,
        name: document.getElementById("playerName").value.trim(),
        linkedin_url: document.getElementById("linkedinUrl").value.trim(),
        p1_actions: p1ActionLog,
        p2_actions: p2ActionLog
      })
    });
    const out = await res.json();
    if (res.status === 401) { forceRelock("Session expired. Re-enter passcode."); return; }
    if (!res.ok) throw new Error(out.detail || "Verification failed");
    status.style.color = "#6ee7b7";
    status.textContent = `Verified! Official Score: ${out.score}`;
    fetchLeaderboard();
  } catch (err) {
    status.style.color = "#f87171";
    status.textContent = err.message;
    btn.disabled = false;
  }
});

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, m => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[m]));
}

const FIXED_DT = 1000 / 60;
let accumulator = 0, lastTime = performance.now();

function loop(now) {
  const frameTime = Math.min(now - lastTime, 100);
  lastTime = now;
  accumulator += frameTime;
  while (accumulator >= FIXED_DT) {
    stepSimulation();
    accumulator -= FIXED_DT;
  }
  render();
  requestAnimationFrame(loop);
}

checkOrientation();
wakeAndCheckHealth();
try {
  const savedToken = sessionStorage.getItem("df_token");
  if (savedToken) onUnlocked(savedToken).catch(() => forceRelock("Session expired. Re-enter passcode."));
} catch {}
requestAnimationFrame(loop);
