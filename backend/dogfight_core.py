# dogfight_core.py
import math
import random
import numpy as np

# =====================================================================
# ARENA & AERODYNAMIC CONSTANTS
# =====================================================================
WIDTH = 1200
HEIGHT = 700
GROUND_Y = 45.0
CEILING_Y = HEIGHT - 45.0

GRAVITY = 0.13
THRUST = 0.27
TURN_RATE = 0.065
V_STALL = 2.8
V_CRUISE_LIFT = 4.3
V_MAX = 14.5
PARASITIC_DRAG = 0.0018
INDUCED_DRAG = 0.024

BULLET_SPEED = 14.5
BULLET_LIFE = 36
FIRE_COOLDOWN = 6
MAX_HEAT = 100.0
HEAT_PER_SHOT = 9.0
COOL_RATE = 0.90
BULLET_DAMAGE = 15
HIT_RADIUS = 18.0

# =====================================================================
# MATH HELPERS
# =====================================================================
def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

def modular_dx(x1, x2):
    dx = x2 - x1
    if dx > WIDTH / 2.0:
        dx -= WIDTH
    elif dx < -WIDTH / 2.0:
        dx += WIDTH
    return dx

def rotate_point(px, py, angle):
    ca, sa = math.cos(angle), math.sin(angle)
    return px * ca - py * sa, px * sa + py * ca

# =====================================================================
# ULTRA-FAST PURE NUMPY INFERENCE FOR SELF-PLAY WORKERS
# =====================================================================
class FastNumpyPolicy:
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.w1, self.b1 = data["w1"], data["b1"]
        self.w2, self.b2 = data["w2"], data["b2"]
        self.w3, self.b3 = data["w3"], data["b3"]

    def predict(self, obs, deterministic=False):
        h1 = np.tanh(obs @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        logits = h2 @ self.w3 + self.b3
        l_turn, l_thr, l_fire = logits[0:3], logits[3:5], logits[5:7]

        if deterministic:
            return [int(np.argmax(l_turn)), int(np.argmax(l_thr)), int(np.argmax(l_fire))]
        else:
            def sample(l):
                e = np.exp(l - np.max(l))
                p = e / np.sum(e)
                return int(np.random.choice(len(l), p=p))
            return [sample(l_turn), sample(l_thr), sample(l_fire)]

# =====================================================================
# CORE ENTITIES
# =====================================================================
class Bullet:
    def __init__(self, x, y, vx, vy, color, owner):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.owner = owner
        self.life = BULLET_LIFE

    def update(self):
        self.x = (self.x + self.vx) % WIDTH
        self.y += self.vy
        self.life -= 1
        return self.life > 0 and self.y > GROUND_Y and self.y < HEIGHT

class Jet:
    def __init__(self, x, y, angle, color, name):
        self.x = x
        self.y = y
        self.angle = angle
        self.vx = 6.0 * math.cos(angle)
        self.vy = 6.0 * math.sin(angle)
        self.color = color
        self.name = name

        self.hp = 100
        self.alive = True
        self.crashed = False
        self.throttle = False
        self.turn_dir = 0

        self.fire_timer = 0
        self.heat = 0.0
        self.overheated = False

    def speed(self):
        return math.hypot(self.vx, self.vy)

    def update(self, turn_input, throttle_input, fire_input, bullets, safe_floor=False, fx_events=None):
        if not self.alive:
            return False

        self.turn_dir = turn_input
        self.throttle = bool(throttle_input)
        fired_now = False

        speed = self.speed()

        if speed >= V_STALL:
            authority = 1.0
            self.angle = wrap_angle(self.angle + turn_input * TURN_RATE * authority)
        else:
            authority = 0.15
            self.angle = wrap_angle(self.angle + turn_input * TURN_RATE * authority)
            down_err = wrap_angle(-math.pi * 0.5 - self.angle)
            self.angle = wrap_angle(self.angle + down_err * 0.04)

        vel_angle = math.atan2(self.vy, self.vx) if speed > 0.05 else self.angle
        if speed > 0.05:
            alpha = wrap_angle(self.angle - vel_angle)
            grip = 0.18 * min(1.0, speed / 5.0) if speed >= V_STALL else 0.02
            new_vel_angle = wrap_angle(vel_angle + alpha * grip)

            drag_i = INDUCED_DRAG * (abs(math.sin(alpha)) ** 1.5) * speed
            drag_p = PARASITIC_DRAG * (speed ** 2)
            new_speed = max(0.0, speed - drag_i - drag_p)

            self.vx = new_speed * math.cos(new_vel_angle)
            self.vy = new_speed * math.sin(new_vel_angle)

            if fx_events is not None and abs(alpha) > 0.25 and speed > 5.0 and random.random() < 0.6:
                fx_events.append(("vapor", self.x, self.y, 0.0, 0.0))

        # Air density fades sharply near ceiling (no ceiling-camping!)
        air_density = max(0.05, min(1.0, (CEILING_Y - self.y + 30.0) / 90.0))

        if self.throttle:
            self.vx += THRUST * air_density * math.cos(self.angle)
            self.vy += THRUST * air_density * math.sin(self.angle)

            if fx_events is not None:
                bx = self.x - 16 * math.cos(self.angle)
                by = self.y - 16 * math.sin(self.angle)
                p_vx = self.vx - 3.5 * math.cos(self.angle) + random.uniform(-0.6, 0.6)
                p_vy = self.vy - 3.5 * math.sin(self.angle) + random.uniform(-0.6, 0.6)
                fx_events.append(("flame", bx, by, p_vx, p_vy))

        # Gravity vs. Wing Lift (both fade near ceiling so you fall back down)
        wing_lift = GRAVITY * air_density * min(1.0, (speed / V_CRUISE_LIFT) ** 2) * abs(math.cos(vel_angle))
        self.vy -= (GRAVITY - wing_lift)

        final_speed = self.speed()
        if final_speed > V_MAX:
            scale = V_MAX / final_speed
            self.vx *= scale
            self.vy *= scale

        self.x = (self.x + self.vx) % WIDTH
        self.y += self.vy

        # Thin-air ceiling pushdown (no invisible glass shelf!)
        if self.y > CEILING_Y:
            self.vy -= GRAVITY * 1.4
            if self.y > HEIGHT - 5.0:
                self.y = HEIGHT - 5.0
                self.vy = min(self.vy, -0.5)

        if self.y <= GROUND_Y:
            if safe_floor:
                self.y = GROUND_Y + 5.0
                self.vy = max(3.5, abs(self.vy) * 0.6)
                if math.sin(self.angle) < 0:
                    self.angle = math.pi * 0.22 if math.cos(self.angle) >= 0 else math.pi * 0.78
                if fx_events is not None:
                    fx_events.append(("bounce", self.x, self.y, 0.0, 0.0))
            else:
                self.y = GROUND_Y
                self.hp = 0
                self.alive = False
                self.crashed = True
                if fx_events is not None:
                    fx_events.append(("crash", self.x, self.y, 0.0, 0.0))
                return False

        if self.fire_timer > 0:
            self.fire_timer -= 1

        self.heat = max(0.0, self.heat - COOL_RATE)
        if self.overheated and self.heat < 25.0:
            self.overheated = False

        if fire_input and self.fire_timer == 0 and not self.overheated:
            self.fire_timer = FIRE_COOLDOWN
            self.heat += HEAT_PER_SHOT
            if self.heat >= MAX_HEAT:
                self.heat = MAX_HEAT
                self.overheated = True

            mx = (self.x + 20 * math.cos(self.angle)) % WIDTH
            my = self.y + 20 * math.sin(self.angle)
            bvx = self.vx + BULLET_SPEED * math.cos(self.angle)
            bvy = self.vy + BULLET_SPEED * math.sin(self.angle)
            bullets.append(Bullet(mx, my, bvx, bvy, self.color, self))
            fired_now = True

        if fx_events is not None and self.hp <= 40 and random.random() < 0.4:
            fx_events.append(("smoke", self.x, self.y, random.uniform(-0.4, 0.4), random.uniform(0.2, 0.8)))

        return fired_now

# =====================================================================
# 28D EGO-CENTRIC COCKPIT TELEMETRY
# =====================================================================
def extract_rl_obs(jet: Jet, opp: Jet, bullets: list):
    alt_norm = 2.0 * (jet.y - GROUND_Y) / (CEILING_Y - GROUND_Y) - 1.0
    ca, sa = math.cos(jet.angle), math.sin(jet.angle)
    vx_norm = jet.vx / V_MAX
    vy_norm = jet.vy / V_MAX
    spd_norm = jet.speed() / V_MAX
    heat_norm = jet.heat / MAX_HEAT
    overheat_flag = 1.0 if jet.overheated else 0.0
    hp_norm = jet.hp / 100.0

    pred_y_30 = jet.y + jet.vy * 30.0 - 0.5 * GRAVITY * 900.0
    ground_danger = max(0.0, min(1.0, (GROUND_Y + 135.0 - pred_y_30) / 135.0))
    pred_y_ceil = jet.y + jet.vy * 15.0
    ceiling_danger = max(0.0, min(1.0, (pred_y_ceil - (CEILING_Y - 50.0)) / 80.0))

    dx = modular_dx(jet.x, opp.x)
    dy = opp.y - jet.y
    dist = math.hypot(dx, dy)
    rel_fwd = (dx * ca + dy * sa) / (WIDTH * 0.5)
    rel_left = (-dx * sa + dy * ca) / (WIDTH * 0.5)

    t_hit = min(35.0, dist / BULLET_SPEED)
    lead_dx = modular_dx(jet.x, opp.x + opp.vx * t_hit - jet.vx * t_hit * 0.35)
    lead_dy = (opp.y + opp.vy * t_hit - jet.vy * t_hit * 0.35) - jet.y
    lead_dist = max(1.0, math.hypot(lead_dx, lead_dy))
    lead_dir_fwd = (lead_dx * ca + lead_dy * sa) / lead_dist
    lead_dir_left = (-lead_dx * sa + lead_dy * ca) / lead_dist

    opp_v_fwd = (opp.vx * ca + opp.vy * sa) / V_MAX
    opp_v_left = (-opp.vx * sa + opp.vy * ca) / V_MAX

    rel_ang = wrap_angle(opp.angle - jet.angle)
    cos_rel = math.cos(rel_ang)
    sin_rel = math.sin(rel_ang)

    dist_norm = min(1.0, dist / 800.0)
    opp_hp_norm = opp.hp / 100.0

    opp_to_me_ang = math.atan2(-dy, -dx)
    opp_aim_cos = math.cos(wrap_angle(opp_to_me_ang - opp.angle))
    opp_overheat = 1.0 if opp.overheated else 0.0

    b_fwd, b_left, b_vfwd, b_vleft, b_threat = 0.0, 0.0, 0.0, 0.0, 0.0
    min_b_dist = 750.0
    for b in bullets:
        if b.owner is not jet:
            bdx = modular_dx(jet.x, b.x)
            bdy = b.y - jet.y
            bdist = math.hypot(bdx, bdy)
            if bdist < min_b_dist:
                min_b_dist = bdist
                b_fwd = (bdx * ca + bdy * sa) / 750.0
                b_left = (-bdx * sa + bdy * ca) / 750.0
                b_vfwd = ((b.vx - jet.vx) * ca + (b.vy - jet.vy) * sa) / (BULLET_SPEED + V_MAX)
                b_vleft = (-(b.vx - jet.vx) * sa + (b.vy - jet.vy) * ca) / (BULLET_SPEED + V_MAX)
                b_threat = 1.0 - (bdist / 750.0)

    return [
        alt_norm, ca, sa, vx_norm, vy_norm, spd_norm,
        heat_norm, overheat_flag, hp_norm,
        ground_danger, ceiling_danger,
        rel_fwd, rel_left, lead_dir_fwd, lead_dir_left,
        opp_v_fwd, opp_v_left, cos_rel, sin_rel,
        dist_norm, opp_hp_norm, opp_aim_cos, opp_overheat,
        b_fwd, b_left, b_vfwd, b_vleft, b_threat
    ]

# =====================================================================
# RULE-BASED BOTS
# =====================================================================
def easy_naive_ai_control(ai: Jet, opp: Jet):
    pred_y = ai.y + ai.vy * 24.0 - 0.5 * GRAVITY * 576.0
    if pred_y < GROUND_Y + 90.0:
        err = wrap_angle(math.pi * 0.5 - ai.angle)
        return (1 if err > 0.05 else -1), 1, 0

    if ai.y > CEILING_Y - 70.0 and ai.vy > 0.5:
        err = wrap_angle(-math.pi * 0.5 - ai.angle)
        return (1 if err > 0.05 else -1), 0, 0

    dx = modular_dx(ai.x, opp.x)
    dy = opp.y - ai.y
    dist = math.hypot(dx, dy)
    angle_err = wrap_angle(math.atan2(dy, dx) - ai.angle)
    turn = 1 if angle_err > 0.12 else (-1 if angle_err < -0.12 else 0)
    fire = 1 if (abs(angle_err) < 0.16 and dist < 420.0 and random.random() < 0.3) else 0
    return turn, 1, fire

def boom_zoom_bot_control(ai: Jet, opp: Jet):
    pred_y = ai.y + ai.vy * 28.0 - 0.5 * GRAVITY * 784.0
    if pred_y < GROUND_Y + 105.0:
        err = wrap_angle(math.pi * 0.5 - ai.angle)
        return (1 if err > 0.04 else -1), 1, 0

    if ai.y > CEILING_Y - 65.0 and ai.vy > 0.5:
        target_angle = -0.35 if ai.vx >= 0 else (-math.pi + 0.35)
        err = wrap_angle(target_angle - ai.angle)
        return (1 if err > 0.04 else -1), 1, 0

    dx = modular_dx(ai.x, opp.x)
    dy = opp.y - ai.y
    dist = math.hypot(dx, dy)
    t_hit = min(35.0, dist / BULLET_SPEED)
    lead_dx = modular_dx(ai.x, opp.x + opp.vx * t_hit - ai.vx * t_hit * 0.35)
    lead_dy = (opp.y + opp.vy * t_hit - ai.vy * t_hit * 0.35) - ai.y
    angle_err = wrap_angle(math.atan2(lead_dy, lead_dx) - ai.angle)

    turn = 1 if angle_err > 0.03 else (-1 if angle_err < -0.03 else 0)
    fire = 1 if (abs(angle_err) < 0.15 and dist < 640.0 and not ai.overheated and ai.heat < 90.0) else 0
    return turn, 1, fire
def kiter_bot_control(ai: Jet, opp: Jet):
    """Replicates the Human 'Kite & Escape' Exploiter Strategy."""
    dx = modular_dx(ai.x, opp.x)
    dy = opp.y - ai.y
    dist = math.hypot(dx, dy)

    pred_y = ai.y + ai.vy * 28.0 - 0.5 * GRAVITY * 784.0
    if pred_y < GROUND_Y + 105.0:
        err = wrap_angle(math.pi * 0.5 - ai.angle)
        return (1 if err > 0.04 else -1), 1, 0
    if ai.y > CEILING_Y - 60.0 and ai.vy > 0.5:
        target_angle = -0.35 if ai.vx >= 0 else (-math.pi + 0.35)
        err = wrap_angle(target_angle - ai.angle)
        return (1 if err > 0.04 else -1), 1, 0

    # Too close (< 380 px): Stop shooting, turn away & accelerate to escape!
    if dist < 380.0:
        away_angle = math.atan2(-dy, -dx)
        if away_angle < -math.pi * 0.35 and ai.y < GROUND_Y + 220.0:
            away_angle = 0.0 if math.cos(away_angle) >= 0 else math.pi
        err = wrap_angle(away_angle - ai.angle)
        turn = 1 if err > 0.04 else (-1 if err < -0.04 else 0)
        return turn, 1, 0

    # Long-Range Kiting (380 - 800 px): Lead aim & shoot relentlessly
    t_hit = min(40.0, dist / BULLET_SPEED)
    lead_dx = modular_dx(ai.x, opp.x + opp.vx * t_hit - ai.vx * t_hit * 0.35)
    lead_dy = (opp.y + opp.vy * t_hit - ai.vy * t_hit * 0.35) - ai.y
    angle_err = wrap_angle(math.atan2(lead_dy, lead_dx) - ai.angle)

    turn = 1 if angle_err > 0.03 else (-1 if angle_err < -0.03 else 0)
    throttle = 1 if ai.speed() < 4.5 else 0
    fire = 1 if (abs(angle_err) < 0.16 and dist < 800.0 and not ai.overheated and ai.heat < 94.0) else 0
    return turn, throttle, fire
def floor_hugger_bot_control(ai: Jet, opp: Jet):
    """
    Replicates the 'Low-Altitude Floor-Bait & Upward Snipe' human trap:
    Skims low at 30m-90m without crashing and shoots upward at diving opponents.
    """
    dx = modular_dx(ai.x, opp.x)
    dy = opp.y - ai.y
    dist = math.hypot(dx, dy)

    pred_y = ai.y + ai.vy * 20.0 - 0.5 * GRAVITY * 400.0
    if pred_y < GROUND_Y + 35.0 or (ai.y < GROUND_Y + 55.0 and ai.vy < -0.5):
        safe_ang = 0.42 if math.cos(ai.angle) >= 0 else (math.pi - 0.42)
        err = wrap_angle(safe_ang - ai.angle)
        turn = 1 if err > 0.04 else (-1 if err < -0.04 else 0)
        return turn, 1, 0

    if ai.y > GROUND_Y + 135.0:
        desc_ang = -0.35 if dx >= 0 else (-math.pi + 0.35)
        err = wrap_angle(desc_ang - ai.angle)
        turn = 1 if err > 0.04 else (-1 if err < -0.04 else 0)
        return turn, 1, 0

    t_hit = min(35.0, dist / BULLET_SPEED)
    lead_dx = modular_dx(ai.x, opp.x + opp.vx * t_hit - ai.vx * t_hit * 0.35)
    lead_dy = (opp.y + opp.vy * t_hit - ai.vy * t_hit * 0.35) - ai.y
    desired_angle = math.atan2(lead_dy, lead_dx)

    if ai.y < GROUND_Y + 75.0 and math.sin(desired_angle) < -0.12:
        desired_angle = -0.08 if math.cos(desired_angle) >= 0 else (-math.pi + 0.08)

    angle_err = wrap_angle(desired_angle - ai.angle)
    turn = 1 if angle_err > 0.03 else (-1 if angle_err < -0.03 else 0)
    throttle = 1 if ai.speed() < 6.0 else 0
    fire = 1 if (abs(angle_err) < 0.16 and dist < 720.0 and not ai.overheated and ai.heat < 92.0) else 0
    return turn, throttle, fire
def hard_naive_ai_control(ai: Jet, opp: Jet):
    speed = ai.speed()
    dx = modular_dx(ai.x, opp.x)
    dy = opp.y - ai.y
    dist = math.hypot(dx, dy)

    pred_y = ai.y + ai.vy * 28.0 - 0.5 * GRAVITY * (28.0 ** 2)
    if pred_y < GROUND_Y + 95.0 or (ai.y < GROUND_Y + 145.0 and ai.vy < -1.5):
        err = wrap_angle(math.pi * 0.5 - ai.angle)
        turn = 1 if err > 0.04 else (-1 if err < -0.04 else 0)
        return turn, 1, 0

    if ai.y > CEILING_Y - 65.0 and ai.vy > 0.5:
        target_angle = -0.35 if ai.vx >= 0 else (-math.pi + 0.35)
        err = wrap_angle(target_angle - ai.angle)
        turn = 1 if err > 0.04 else (-1 if err < -0.04 else 0)
        return turn, 0, 0

    if speed < V_STALL + 0.9 and ai.y > GROUND_Y + 190.0:
        target_angle = -0.35 if math.cos(ai.angle) >= 0 else (-math.pi + 0.35)
        err = wrap_angle(target_angle - ai.angle)
        turn = 1 if err > 0.04 else (-1 if err < -0.04 else 0)
        return turn, 1, 0

    opp_to_ai_angle = math.atan2(-dy, -dx)
    opp_aim_err = abs(wrap_angle(opp.angle - opp_to_ai_angle))
    if dist < 350.0 and opp_aim_err < 0.32 and abs(wrap_angle(ai.angle - opp.angle)) < 0.85:
        break_dir = 1 if (ai.y < HEIGHT * 0.65) else -1
        return break_dir, 1, 0

    t_hit = min(35.0, dist / BULLET_SPEED)
    lead_dx = modular_dx(ai.x, opp.x + opp.vx * t_hit - ai.vx * t_hit * 0.35)
    lead_dy = (opp.y + opp.vy * t_hit - ai.vy * t_hit * 0.35) - ai.y

    desired_angle = math.atan2(lead_dy, lead_dx)
    angle_err = wrap_angle(desired_angle - ai.angle)

    turn = 1 if angle_err > 0.035 else (-1 if angle_err < -0.035 else 0)

    throttle = 1
    if abs(angle_err) > 1.2 and speed > 8.5:
        throttle = 0

    fire = 0
    if abs(angle_err) < 0.14 and dist < 560.0 and not ai.overheated and ai.heat < 88.0:
        fire = 1

    return turn, throttle, fire
