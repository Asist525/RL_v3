# agent.py
# AvoidBlurp-Normal-v0 커리큘럼형 탭형 RL (obs 파서/축 정합 + YourAgent 포함)
# - 관측: x-bin(0..14), t-bin(0..35), 위에서 내려오는 적 3기 (dx,dy)
# - 액션: 0:stay, 1:left, 2:right
# - 마스크: 경계 + 관측으로 100% 확정되는 즉사만
# - 보상: 각 레벨마다 success=1, death=0
# - 커리큘럼: 400 → 800 → 1200 → 1800 → 2400 → 3000 → 3600
# - ε: 기본 0.2, 최종레벨 성공 후 0.01까지 감쇠
# - 무한학습: python agent.py
# - 체크포인트:
#     2000ep마다 → ckpt_auto/
#     레벨 업할 때 → ckpt_best/
#     최종 ε감쇠 끝 → ckpt_final/ 후 종료
#
# 주의(갱신):
# - 이 코드는 AvoidBlurp가 y축을 "위에서 아래로 증가"한다고 가정해서 dy = player_y - enemy_y 로 인코딩한다.
#   만약 env가 반대로(y가 아래→위)라면 encode_obs 안의 dy 한 줄만 바꿔라.

from __future__ import annotations
import os
import sys
import argparse
import pickle
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import gymnasium as gym
import kymnasium as kym  # noqa: F401

# =========================
# 전역 설정
# =========================
ENV_ID = "kymnasium/AvoidBlurp-Normal-v0"
ACTION_STAY = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
FORWARD = ACTION_RIGHT  # 오른쪽/전진이 죽음 후보라고 가정

LEVEL_STEPS = [400, 800, 1200, 1800, 2400, 3000, 3600]

EPS_INIT = 0.2
EPS_FINAL = 0.01
EPS_DECAY_STEPS = 20000

AUTO_CKPT_EVERY = 2000
SUCCESS_WINDOW = 50
SUCCESS_THRESHOLD = 45  # 50 중 45 = 90%

LOG_DIR = "logs"
CKPT_AUTO_DIR = "ckpt_auto"
CKPT_BEST_DIR = "ckpt_best"
CKPT_FINAL_DIR = "ckpt_final"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_AUTO_DIR, exist_ok=True)
os.makedirs(CKPT_BEST_DIR, exist_ok=True)
os.makedirs(CKPT_FINAL_DIR, exist_ok=True)


# =========================
# 로그
# =========================
def open_log_file():
    return open(os.path.join(LOG_DIR, "avoid_train.log"), "a", buffering=1, encoding="utf-8")


LOG_FH = open_log_file()


def log_line(msg: str):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    LOG_FH.write(f"[{now}] {msg}\n")


# =========================
# 관측 인코딩
# =========================
def encode_time(t: int) -> int:
    return min(t // 100, 35)


def bin_dx(dx: int) -> int:
    if dx <= -2:
        return 0
    if dx == -1:
        return 1
    if dx == 0:
        return 2
    if dx == 1:
        return 3
    return 4


def bin_dy(dy: int) -> int:
    if dy <= 1:
        return 0
    if dy <= 3:
        return 1
    if dy <= 6:
        return 2
    return 3


def _parse_raw_obs(raw_obs: Any) -> Tuple[int, int, List[Tuple[int, int]]]:
    """
    가능한 케이스 전부 커버:
    1) dict형
       {
         "player": {"x": int, "y": int}  또는  [x,y]  또는  np.array([x,y]),
         "enemies": [
            {"x": int, "y": int}  또는  [ex,ey]  또는  np.array([ex,ey]),
            ...
         ]
       }
    2) ndarray형 / list형
       [player_x, player_y, e1_x, e1_y, e2_x, e2_y, ...]
    """
    # 1) dict
    if isinstance(raw_obs, dict):
        p = raw_obs.get("player", {})
        if isinstance(p, (list, tuple, np.ndarray)):
            px = int(p[0]) if len(p) > 0 else 0
            py = int(p[1]) if len(p) > 1 else 0
        elif isinstance(p, dict):
            px = int(p.get("x", 0))
            py = int(p.get("y", 0))
        else:
            px, py = 0, 0

        enemies_list: List[Tuple[int, int]] = []
        for e in raw_obs.get("enemies", []):
            if isinstance(e, (list, tuple, np.ndarray)):
                ex = int(e[0]) if len(e) > 0 else 0
                ey = int(e[1]) if len(e) > 1 else 0
            elif isinstance(e, dict):
                ex = int(e.get("x", 0))
                ey = int(e.get("y", 0))
            else:
                ex, ey = 0, 0
            enemies_list.append((ex, ey))
        return px, py, enemies_list

    # 2) ndarray/list
    arr = np.array(raw_obs).astype(int).tolist()
    if len(arr) >= 2:
        px, py = arr[0], arr[1]
        enemies_list = []
        rest = arr[2:]
        for i in range(0, len(rest), 2):
            ex = rest[i]
            ey = rest[i + 1] if i + 1 < len(rest) else 0
            enemies_list.append((ex, ey))
        return int(px), int(py), enemies_list

    return 0, 0, []


def encode_obs(raw_obs: Any, t: int) -> List[int]:
    player_x, player_y, enemies = _parse_raw_obs(raw_obs)

    # 1) x-bin: 0..14 고정
    agent_x_bin = max(0, min(player_x, 14))

    # 2) t-bin
    t_bin = encode_time(t)

    # 3) 적 정렬 (위에서 내려온다고 보고 dy = player_y - enemy_y)
    cand = []
    for (ex, ey) in enemies:
        dx = ex - player_x
        dy = player_y - ey
        cand.append((dy, abs(dx), ex, ey))
    cand.sort(key=lambda x: (x[0], x[1]))

    top3: List[Tuple[int, int]] = []
    for i in range(min(3, len(cand))):
        dy, _, ex, ey = cand[i]
        dx = ex - player_x
        real_dy = player_y - ey
        top3.append((dx, real_dy))

    while len(top3) < 3:
        top3.append((0, 999))

    enc = [agent_x_bin, t_bin]
    for (dx, dy) in top3:
        enc.append(bin_dx(dx))
        enc.append(bin_dy(dy))
    return enc


# =========================
# 마스킹
# =========================
def enemy_exactly_above(enc: List[int]) -> bool:
    for i in range(3):
        dx = enc[2 + 2 * i]
        dy = enc[2 + 2 * i + 1]
        if dx == 2 and dy == 0:
            return True
    return False


def build_mask(enc: List[int]) -> List[bool]:
    agent_x = enc[0]
    mask = [False, False, False]

    if agent_x == 0:
        mask[ACTION_LEFT] = True
    if agent_x == 14:
        mask[ACTION_RIGHT] = True

    if enemy_exactly_above(enc):
        mask[FORWARD] = True

    return mask


# =========================
# Q-Table 에이전트
# =========================
class QAgent:
    def __init__(self, num_actions: int = 3, alpha: float = 0.1, gamma: float = 0.99):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.q = defaultdict(lambda: np.zeros(self.num_actions, dtype=np.float32))

    def act(self, enc: List[int], epsilon: float) -> int:
        s = tuple(enc)
        q_s = self.q[s]
        mask = build_mask(enc)

        q_masked = q_s.copy()
        for a, m in enumerate(mask):
            if m:
                q_masked[a] = -1e9

        if np.random.rand() < epsilon:
            valid_actions = [a for a, m in enumerate(mask) if not m]
            if not valid_actions:
                return ACTION_STAY
            return int(np.random.choice(valid_actions))
        else:
            return int(np.argmax(q_masked))

    def update(self, enc: List[int], a: int, r: float, enc_next: List[int], done: bool):
        s = tuple(enc)
        s_next = tuple(enc_next)
        q_s = self.q[s]
        q_next = self.q[s_next]

        if done:
            target = r
        else:
            target = r + self.gamma * np.max(q_next)

        q_s[a] += self.alpha * (target - q_s[a])

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "q": dict(self.q),
                    "num_actions": self.num_actions,
                    "alpha": self.alpha,
                    "gamma": self.gamma,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "QAgent":
        with open(path, "rb") as f:
            data = pickle.load(f)
        ag = cls(num_actions=data["num_actions"], alpha=data["alpha"], gamma=data["gamma"])
        ag.q = defaultdict(lambda: np.zeros(ag.num_actions, dtype=np.float32))
        for k, v in data["q"].items():
            ag.q[tuple(k)] = np.array(v, dtype=np.float32)
        return ag


# =========================
# YourAgent (evaluate.py 연동)
# =========================
class YourAgent:
    def __init__(self, inner: QAgent):
        self.inner = inner
        self.t = 0  # 평가 때 step 카운터

    @staticmethod
    def load(path: str) -> "YourAgent":
        qa = QAgent.load(path)
        return YourAgent(qa)

    def act(self, obs: Any, info: Any = None) -> int:
        enc = encode_obs(obs, self.t)
        a = self.inner.act(enc, epsilon=0.0)
        self.t += 1
        return a

    # 평가기에서 reset을 호출해주면 좋지만, 보통은 안 하니까 여기서는 생략


# =========================
# 학습 루프 요소
# =========================
def make_env() -> gym.Env:
    return gym.make(ENV_ID, obs_type="custom")


def run_episode(env: gym.Env, agent: QAgent, max_steps: int, epsilon: float) -> Tuple[int, int]:
    raw_obs, info = env.reset()
    t = 0
    enc = encode_obs(raw_obs, t)
    done = False
    success = 0
    ep_len = 0

    while not done and t < max_steps:
        a = agent.act(enc, epsilon)
        raw_obs_next, _r_env, terminated, truncated, info = env.step(a)
        t += 1
        ep_len += 1

        enc_next = encode_obs(raw_obs_next, t)

        if t >= max_steps:
            r = 1.0
            done = True
            success = 1
        elif terminated or truncated:
            r = 0.0
            done = True
            success = 0
        else:
            r = 0.0
            done = False

        agent.update(enc, a, r, enc_next, done)
        enc = enc_next

    return success, ep_len


def save_ckpt(folder: str, ep: int, level_idx: int, agent: QAgent):
    fname = f"ep{ep:06d}_L{level_idx}.pkl"
    path = os.path.join(folder, fname)
    agent.save(path)
    log_line(f"[CKPT] saved {path}")


# =========================
# 무한학습
# =========================
def train_forever():
    env = make_env()
    agent = QAgent()

    level_idx = 0
    success_buf: List[int] = []
    epsilon = EPS_INIT
    reached_final_level = False
    epsilon_decay_counter = 0

    ep = 0
    while True:
        max_steps = LEVEL_STEPS[level_idx]
        success, ep_len = run_episode(env, agent, max_steps, epsilon)

        log_line(
            f"[EP {ep:06d}] level={LEVEL_STEPS[level_idx]} len={ep_len} success={success} eps={epsilon:.4f}"
        )

        if ep > 0 and ep % AUTO_CKPT_EVERY == 0:
            save_ckpt(CKPT_AUTO_DIR, ep, level_idx, agent)

        success_buf.append(success)
        if len(success_buf) > SUCCESS_WINDOW:
            success_buf.pop(0)

        # 레벨업
        if len(success_buf) == SUCCESS_WINDOW and sum(success_buf) >= SUCCESS_THRESHOLD:
            save_ckpt(CKPT_BEST_DIR, ep, level_idx, agent)
            log_line(f"[LEVEL UP] from={LEVEL_STEPS[level_idx]}")
            if level_idx < len(LEVEL_STEPS) - 1:
                level_idx += 1
                success_buf = []
            else:
                reached_final_level = True

        # 마지막 레벨 → ε 감쇠
        if reached_final_level:
            if epsilon > EPS_FINAL:
                epsilon_decay_counter += 1
                ratio = min(1.0, epsilon_decay_counter / EPS_DECAY_STEPS)
                epsilon = EPS_INIT + (EPS_FINAL - EPS_INIT) * ratio
            else:
                save_ckpt(CKPT_FINAL_DIR, ep, level_idx, agent)
                log_line("[TRAIN] final epsilon reached, exit.")
                break

        ep += 1

    env.close()


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="무한학습 실행 (기본)")
    parser.add_argument("--eval", metavar="PATH", help="저장된 pkl로 1회 시연 (kym.evaluate용 YourAgent와 동일)")
    args = parser.parse_args()

    if args.eval:
        # 시연 모드: 이건 사람 눈으로 볼 때만
        agent = YourAgent.load(args.eval)
        env = make_env()
        raw_obs, info = env.reset()
        done = False
        t = 0
        while not done and t < 3600:
            a = agent.act(raw_obs, info)
            raw_obs, _, terminated, truncated, info = env.step(a)
            env.render()
            t += 1
            if terminated or truncated:
                done = True
        env.close()
    else:
        train_forever()


if __name__ == "__main__":
    main()
