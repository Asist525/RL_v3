# agent.py
# AvoidBlurp-Normal-v0 Expected SARSA
# - 관측: player + 위에서 가까운 적 N기까지 (기본 3)
# - 이어학습: --load 로 pkl 불러와서 그대로 train 이어가기
# - 탐색 옵션:
#     --epsilon-base 0.2     ← 시작/재시작 시 ε를 이 값으로 강제
#     --low-eps              ← 제출 직전 안정화 모드
#     --epsilon-low 0.06     ← low-eps일 때 ε값
# - 마스크 옵션:
#     --strict-mask          ← 진짜로 막힌 상태는 열어주지 않게
# - 2000ep마다 ckpt

from __future__ import annotations
import argparse
import pickle
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np
import gymnasium as gym
import kymnasium as kym  # noqa: F401  # env 등록용


# =========================
# 설정 데이터클래스
# =========================

@dataclass
class EnvConfig:
    env_id: str = "kymnasium/AvoidBlurp-Normal-v0"
    obs_type: str = "custom"


@dataclass
class RewardConfig:
    alive_reward: float = 0.1
    success_reward: float = 1000.0     # 2분 버티면 +1000
    crash_penalty: float = -1000.0     # 충돌하면 -1000
    danger_lambda: float = 1.0
    danger_y: float = 180.0
    switch_penalty: float = 0.05
    use_switch: bool = True
    use_center_shaping: bool = False
    center_x: float = 240.0
    center_alpha: float = 0.001


@dataclass
class MaskConfig:
    left_margin: float = 0.0
    right_margin: float = 480.0  # reset 때 실제 폭으로 덮어씀
    predict_collision: bool = True
    danger_y: float = 180.0
    allow_empty_recover: bool = True   # 막혀도 stay 하나는 열어주는지
    keep_at_least_two: bool = True     # 최소 2액션은 남기기


@dataclass
class TrainConfig:
    episodes: int = 30000
    gamma: float = 0.99
    alpha_init: float = 0.15
    alpha_c: float = 3
    alpha_min: float = 0.05
    epsilon_start: float = 0.2     # ← 기본을 0.2로
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.999   # --low-eps 모드에서 사용
    ckpt_every: int = 2000
    save_dir: str = "ckpt"
    infinite: bool = False


# =========================
# 관찰 → 상태 인코더
# =========================

class StateEncoder:
    """
    플레이어 x를 15구간
    위에서 가까운 적 max_enemies기까지 본다 (기본 3)
    각 적: relx 5구간 × dy 3구간
    상태 수 = px_bins * (e_space ** max_enemies) * 2
    ※ max_enemies 늘리면 상태수 기하급수적으로 늘어남
    """

    def __init__(
        self,
        num_px_bins: int = 15,
        relx_bins: int = 5,
        dy_bins: int = 3,
        danger_y: float = 180.0,
        width: float = 480.0,
        max_enemies: int = 3,   # ← 기본 3으로
    ):
        self.num_px_bins = num_px_bins
        self.relx_bins = relx_bins
        self.dy_bins = dy_bins
        self.danger_y = danger_y
        self.width = width
        self.max_enemies = max_enemies

    def _bin_player_x(self, cx: float) -> int:
        b = int(cx / (self.width / self.num_px_bins))
        return max(0, min(self.num_px_bins - 1, b))

    def _bin_relx(self, dx: float) -> int:
        if dx < -60:
            return 0
        elif dx < -20:
            return 1
        elif dx < 20:
            return 2
        elif dx < 60:
            return 3
        else:
            return 4

    def _bin_dy(self, dy: float) -> int:
        if dy < 60:
            return 0
        elif dy < 140:
            return 1
        else:
            return 2

    def encode(self, obs: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        player = obs["player"]
        enemies = obs["enemies"]

        px, py, pw, ph, pspeed = player
        cx = px + pw / 2.0

        # 위에 있는 적만 모으고 가까운 순으로
        active = []
        for row in enemies:
            if not np.any(row):
                continue
            ex, ey, ew, eh, espeed, eaccel = row
            if ey >= py:
                continue
            dy = py - ey
            active.append((dy, ex, ey, ew, eh))
        active.sort(key=lambda x: x[0])

        # max_enemies 만큼 채움 (없으면 더미)
        enemy_bins: List[Tuple[int, int]] = []
        for i in range(self.max_enemies):
            if i < len(active):
                dy, ex, ey, ew, eh = active[i]
                ecx = ex + ew / 2.0
                relx = ecx - cx
                relx_bin = self._bin_relx(relx)
                dy_bin = self._bin_dy(dy)
                enemy_bins.append((relx_bin, dy_bin))
            else:
                # 더미: 정면, 멀다
                enemy_bins.append((2, 2))

        # 위험 플래그 (가장 가까운 적 기준)
        if active:
            dy0, ex0, ey0, ew0, eh0 = active[0]
            ecx0 = ex0 + ew0 / 2.0
            relx0 = ecx0 - cx
            danger_flag = 1 if (0 < dy0 < self.danger_y and abs(relx0) < (pw / 2.0 + ew0 / 2.0)) else 0
        else:
            danger_flag = 0

        px_bin = self._bin_player_x(cx)
        e_space = self.relx_bins * self.dy_bins

        # 다중 적 인덱싱
        idx = px_bin
        for (relx_bin, dy_bin) in enemy_bins:
            e_idx = relx_bin * self.dy_bins + dy_bin
            idx = idx * e_space + e_idx

        state_idx = idx * 2 + danger_flag

        extra = {
            "player": player,
            "px": px,
            "py": py,
            "pw": pw,
            "ph": ph,
            "pspeed": pspeed,
            "cx": cx,
            "enemy_info": active,
            "danger_flag": danger_flag,
        }
        return state_idx, extra

    @property
    def num_states(self) -> int:
        e_space = self.relx_bins * self.dy_bins
        return self.num_px_bins * (e_space ** self.max_enemies) * 2


# =========================
# 마스크
# =========================

def make_hard_mask(
    extra: Dict[str, Any],
    mask_cfg: MaskConfig,
    return_blocked: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, bool]]:
    """
    기본 3액션: 0=stay, 1=left, 2=right
    """
    mask = np.ones(3, dtype=np.int8)
    px = extra["px"]
    pw = extra["pw"]
    pspeed = extra["pspeed"]
    was_blocked = False

    # 경계
    if px <= mask_cfg.left_margin:
        mask[1] = 0
    if px + pw >= mask_cfg.right_margin:
        mask[2] = 0

    # 충돌 예측
    if mask_cfg.predict_collision:
        enemy_info = extra.get("enemy_info", [])
        if enemy_info:
            dy, ex, ey, ew, eh = enemy_info[0]
            player_left = px
            player_right = px + pw
            enemy_left = ex
            enemy_right = ex + ew

            # 지금 위치에서 겹치면 제자리 금지
            x_overlap_now = not (player_right < enemy_left or enemy_right < player_left)
            if 0 < dy < mask_cfg.danger_y and x_overlap_now:
                mask[0] = 0

            # 왼쪽 이동 시
            new_left = px - pspeed
            new_right = new_left + pw
            x_overlap_left = not (new_right < enemy_left or enemy_right < new_left)
            if 0 < dy < mask_cfg.danger_y and x_overlap_left:
                mask[1] = 0

            # 오른쪽 이동 시
            new_left = px + pspeed
            new_right = new_left + pw
            x_overlap_right = not (new_right < enemy_left or enemy_right < new_left)
            if 0 < dy < mask_cfg.danger_y and x_overlap_right:
                mask[2] = 0

    # 전부 막혔으면 복구
    if mask.sum() == 0:
        was_blocked = True
        if mask_cfg.allow_empty_recover:
            mask[0] = 1  # 일단 정지

    # 1개만 남았으면 최소 2개로
    if mask_cfg.keep_at_least_two and mask.sum() < 2:
        for i in range(3):
            if mask.sum() >= 2:
                break
            if mask[i] == 0:
                mask[i] = 1

    if return_blocked:
        return mask, was_blocked
    return mask


# =========================
# 보상
# =========================

def compute_reward(
    prev_extra: Dict[str, Any],
    action: int,
    next_extra: Dict[str, Any],
    terminated: bool,
    truncated: bool,
    rew_cfg: RewardConfig,
) -> float:
    """
    문서 기준:
    - 충돌 → terminated=False, truncated=True → 죽음 → -1000
    - 2분 버팀 → terminated=True, truncated=False → 성공 → +1000
    """
    r = 0.0
    # 기본 생존 보상
    r += rew_cfg.alive_reward

    # 위험지대 패널티
    enemy_info = next_extra.get("enemy_info", [])
    if enemy_info:
        dy, ex, ey, ew, eh = enemy_info[0]
        relx = (ex + ew / 2.0) - next_extra["cx"]

        half_span = next_extra["pw"] / 2.0      # 플레이어 반폭
        enemy_half = ew / 2.0                   # 적 반폭

        if 0 < dy < rew_cfg.danger_y and abs(relx) < (half_span + enemy_half):
            r -= rew_cfg.danger_lambda * (1.0 / (dy + 1.0))


    # 액션 전환 패널티
    prev_action = prev_extra.get("prev_action", None)
    if rew_cfg.use_switch and prev_action is not None and prev_action != action:
        r -= rew_cfg.switch_penalty

    # 중앙 정렬 shaping (필요할 때만 사용)
    if rew_cfg.use_center_shaping:
        prev_cx = prev_extra["cx"]
        next_cx = next_extra["cx"]
        phi_prev = -rew_cfg.center_alpha * abs(prev_cx - rew_cfg.center_x)
        phi_next = -rew_cfg.center_alpha * abs(next_cx - rew_cfg.center_x)
        r += (0.99 * phi_next - phi_prev)

    # 종료 처리 (문서 기준)
    if truncated:
        # 충돌
        r += rew_cfg.crash_penalty
    elif terminated:
        # 2분 버팀
        r += rew_cfg.success_reward

    return r


# =========================
# SARSA 에이전트
# =========================

class SarsaAgent:
    def __init__(
        self,
        encoder: StateEncoder,
        reward_cfg: RewardConfig,
        mask_cfg: MaskConfig,
        train_cfg: TrainConfig,
        env_cfg: EnvConfig,
    ):
        self.encoder = encoder
        self.reward_cfg = reward_cfg
        self.mask_cfg = mask_cfg
        self.train_cfg = train_cfg
        self.env_cfg = env_cfg

        # 상태 수가 매우 크다 (max_enemies=5면 250MB급)
        self.q = np.zeros((encoder.num_states, 3), dtype=np.float32)
        self.visit_counts = np.zeros((encoder.num_states, 3), dtype=np.int32)

        self.gamma = train_cfg.gamma
        self.epsilon = train_cfg.epsilon_start
        self.episodes_trained = 0

    def select_action(self, state: int, mask: np.ndarray) -> int:
        allowed = np.where(mask == 1)[0]
        if len(allowed) == 1:
            return int(allowed[0])
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(allowed))
        qvals = self.q[state]
        best = allowed[np.argmax(qvals[allowed])]
        return int(best)

    def update(
        self,
        s: int,
        a: int,
        r: float,
        s_next: int,
        next_mask: np.ndarray,
        terminated: bool,
        truncated: bool,
    ):
        # 방문수 증가 → α 계산
        self.visit_counts[s, a] += 1
        visits = self.visit_counts[s, a]
        alpha = 1.0 / (self.train_cfg.alpha_c + float(visits))
        if alpha < self.train_cfg.alpha_min:
            alpha = self.train_cfg.alpha_min

        q_sa = self.q[s, a]

        # 터미널이면 부트스트랩 안 함
        if terminated or truncated:
            target = r
        else:
            allowed_next = np.where(next_mask == 1)[0]
            if len(allowed_next) == 0:
                target = r
            elif len(allowed_next) == 1:
                only_a = allowed_next[0]
                target = r + self.gamma * self.q[s_next, only_a]
            else:
                q_next = self.q[s_next]
                best_a = allowed_next[np.argmax(q_next[allowed_next])]

                # 기대값: ε-greedy (허용 액션에만 분배)
                probs = np.zeros(3, dtype=np.float32)
                for act in allowed_next:
                    probs[act] = self.epsilon / float(len(allowed_next) - 1)
                probs[best_a] = 1.0 - self.epsilon

                expected_q = float((q_next * probs).sum())
                target = r + self.gamma * expected_q

        self.q[s, a] = q_sa + alpha * (target - q_sa)

    def save(self, path: str):
        payload = {
            "q": self.q,
            "visit_counts": self.visit_counts,
            "epsilon": self.epsilon,
            "episodes_trained": self.episodes_trained,
            "encoder_cfg": {
                "num_px_bins": self.encoder.num_px_bins,
                "relx_bins": self.encoder.relx_bins,
                "dy_bins": self.encoder.dy_bins,
                "danger_y": self.encoder.danger_y,
                "width": self.encoder.width,
                "max_enemies": self.encoder.max_enemies,
            },
            "reward_cfg": asdict(self.reward_cfg),
            "mask_cfg": asdict(self.mask_cfg),
            "train_cfg": asdict(self.train_cfg),
            "env_cfg": asdict(self.env_cfg),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load(path: str) -> "SarsaAgent":
        with open(path, "rb") as f:
            data = pickle.load(f)

        if "q" in data and "encoder_cfg" in data:
            enc = StateEncoder(**data["encoder_cfg"])
            rew = RewardConfig(**data["reward_cfg"])
            mask = MaskConfig(**data["mask_cfg"])
            train = TrainConfig(**data["train_cfg"])
            env_cfg = EnvConfig(**data["env_cfg"])
            agent = SarsaAgent(enc, rew, mask, train, env_cfg)
            agent.q = data["q"]
            agent.visit_counts = data.get("visit_counts", np.zeros_like(agent.q, dtype=np.int32))
            agent.epsilon = data.get("epsilon", train.epsilon_start)
            agent.episodes_trained = data.get("episodes_trained", 0)
            return agent

        raise ValueError("Unsupported pkl format")


# =========================
# 평가 래퍼
# =========================

class YourAgent:
    def __init__(self, agent: SarsaAgent):
        self.agent = agent
        self.prev_action: Optional[int] = None
        self.prev_extra: Optional[Dict[str, Any]] = None

    @staticmethod
    def load(path: str) -> "YourAgent":
        agent = SarsaAgent.load(path)
        return YourAgent(agent)

    def act(self, observation: Dict[str, Any], info: Any = None) -> int:
        state, extra = self.agent.encoder.encode(observation)
        extra["prev_action"] = self.prev_action
        mask = make_hard_mask(extra, self.agent.mask_cfg)
        action = self.agent.select_action(state, mask)
        self.prev_action = action
        self.prev_extra = extra
        return action


# =========================
# 학습 루프
# =========================

def infer_width_from_obs(obs: Dict[str, Any]) -> float:
    """
    현재 obs에서 화면 실제 폭을 추정
    """
    player = obs["player"]
    enemies = obs["enemies"]
    px, py, pw, ph, pspeed = player
    max_x = px + pw
    for row in enemies:
        if not np.any(row):
            continue
        ex, ey, ew, eh, espeed, eaccel = row
        if ex + ew > max_x:
            max_x = ex + ew
    return max_x + 10.0


def train(args: argparse.Namespace):
    # 1) env 먼저 띄움
    env = gym.make("kymnasium/AvoidBlurp-Normal-v0", obs_type="custom", render_mode=None)
    obs, _ = env.reset()
    width_guess = infer_width_from_obs(obs)

    # 2) pkl 로드 여부
    if args.load:
        agent = SarsaAgent.load(args.load)

        # 실행 환경에서 다시 측정한 폭으로 덮어쓰기
        agent.encoder.width = width_guess
        agent.mask_cfg.right_margin = width_guess

        # 탐색 강제
        if args.epsilon_base is not None:
            agent.epsilon = args.epsilon_base

        # 마스크 엄격 모드
        if args.strict_mask:
            agent.mask_cfg.allow_empty_recover = False
            agent.mask_cfg.keep_at_least_two = False

        # train_cfg 덮어쓰기
        agent.train_cfg.episodes = args.episodes
        agent.train_cfg.ckpt_every = args.ckpt_every
        agent.train_cfg.save_dir = args.save_dir
        agent.train_cfg.infinite = args.infinite
        os.makedirs(agent.train_cfg.save_dir, exist_ok=True)
    else:
        env_cfg = EnvConfig()
        rew_cfg = RewardConfig()
        mask_cfg = MaskConfig()
        train_cfg = TrainConfig(
            episodes=args.episodes,
            ckpt_every=args.ckpt_every,
            save_dir=args.save_dir,
            infinite=args.infinite,
        )
        os.makedirs(train_cfg.save_dir, exist_ok=True)

        encoder = StateEncoder(width=width_guess, danger_y=rew_cfg.danger_y, max_enemies=args.max_enemies)
        mask_cfg.right_margin = width_guess
        # 새 학습 때도 엄격 모드 적용 가능
        if args.strict_mask:
            mask_cfg.allow_empty_recover = False
            mask_cfg.keep_at_least_two = False

        agent = SarsaAgent(encoder, rew_cfg, mask_cfg, train_cfg, env_cfg)

        # CLI로 ε 강제
        if args.epsilon_base is not None:
            agent.epsilon = args.epsilon_base

    # 로그 파일
    log_f = None
    if getattr(args, "log_file", None):
        log_dir = os.path.dirname(args.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_f = open(args.log_file, "a", encoding="utf-8")

    def log_print(msg: str):
        print(msg)
        if log_f is not None:
            print(msg, file=log_f, flush=True)

    episode_counter = 0
    len_buffer: List[int] = []
    death_buffer: List[int] = []

    try:
        while True:
            obs, _ = env.reset()
            state, extra = agent.encoder.encode(obs)
            extra["prev_action"] = None

            ep_reward = 0.0
            ep_len = 0
            ep_danger = 0
            ep_blocked = 0
            died = 0
            done = False

            while not done:
                mask, was_blocked = make_hard_mask(extra, agent.mask_cfg, return_blocked=True)
                if was_blocked:
                    ep_blocked += 1

                action = agent.select_action(state, mask)
                next_obs, _, terminated, truncated, info = env.step(action)
                next_state, next_extra = agent.encoder.encode(next_obs)
                next_extra["prev_action"] = action

                r = compute_reward(extra, action, next_extra, terminated, truncated, agent.reward_cfg)

                if next_extra.get("danger_flag", 0) == 1:
                    ep_danger += 1

                next_mask = make_hard_mask(next_extra, agent.mask_cfg)
                agent.update(state, action, r, next_state, next_mask, terminated, truncated)

                state = next_state
                extra = next_extra
                ep_reward += r
                ep_len += 1
                done = terminated or truncated

                # 문서 기준: truncated=True → 충돌
                if truncated:
                    died = 1
                elif terminated:
                    died = 0

            agent.episodes_trained += 1
            episode_counter += 1
            len_buffer.append(ep_len)
            death_buffer.append(died)

            # ε 안정화 모드
            if args.low_eps:
                # 제출 직전 안정화 모드: 조금씩 내려서 epsilon_low까지
                if agent.epsilon > args.epsilon_low:
                    agent.epsilon = max(
                        args.epsilon_low,
                        agent.epsilon * agent.train_cfg.epsilon_decay
                    )
            else:
                # low_eps 안 켰으면 그냥 고정 (ex. 0.2)
                pass



            # 에피소드 로그
            if episode_counter % 20 == 0 or episode_counter == 1:
                log_print(
                    f"[EP {agent.episodes_trained}] len={ep_len} R={ep_reward:.2f} "
                    f"danger={ep_danger} blocked={ep_blocked} died={died} eps={agent.epsilon:.3f}"
                )

            # CKPT
            if episode_counter % agent.train_cfg.ckpt_every == 0:
                ckpt_path = os.path.join(
                    agent.train_cfg.save_dir,
                    f"avoid_ep{agent.episodes_trained:06d}_eps{agent.epsilon:.3f}.pkl",
                )
                agent.save(ckpt_path)
                avg_len = sum(len_buffer) / len(len_buffer)
                max_len = max(len_buffer)
                death_rate = sum(death_buffer) / len(death_buffer)
                log_print(
                    f"[CKPT] ep={agent.episodes_trained} avg_len={avg_len:.1f} "
                    f"max_len={max_len} death_rate={death_rate:.3f} "
                    f"epsilon={agent.epsilon:.3f} saved={ckpt_path}"
                )
                len_buffer.clear()
                death_buffer.clear()

            # finite 모드면 종료
            if not agent.train_cfg.infinite and episode_counter >= agent.train_cfg.episodes:
                break

    finally:
        env.close()
        if log_f is not None:
            log_f.close()


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train")
    p_train.add_argument("--episodes", type=int, default=30000)
    p_train.add_argument("--ckpt-every", type=int, default=2000)
    p_train.add_argument("--save-dir", type=str, default="ckpt")
    p_train.add_argument("--infinite", action="store_true")
    p_train.add_argument("--log-file", type=str, default=None)

    # 이어학습/탐색/마스크 옵션
    p_train.add_argument("--load", type=str, default=None, help="기존 pkl 불러와 이어학습")
    p_train.add_argument("--epsilon-base", type=float, default=None, help="시작/재시작 ε 강제값 (예: 0.2)")
    p_train.add_argument("--low-eps", action="store_true", help="제출 직전 안정화 모드: ε를 낮은 값으로 고정")
    p_train.add_argument("--epsilon-low", type=float, default=0.06, help="--low-eps 모드에서 쓸 ε")
    p_train.add_argument("--max-enemies", type=int, default=3, help="관측에서 고려할 위협 개수 (새 학습 때만 적용)")
    p_train.add_argument("--strict-mask", action="store_true", help="막혔으면 진짜로 막기(allow_empty_recover 끔)")

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--pkl", type=str, required=True)

    args = parser.parse_args()

    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        agent = YourAgent.load(args.pkl)
        env = gym.make("kymnasium/AvoidBlurp-Normal-v0", obs_type="custom", render_mode="human")
        obs, _ = env.reset()
        done = False
        while not done:
            a = agent.act(obs)
            obs, _, terminated, truncated, info = env.step(a)
            done = terminated or truncated
        env.close()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
