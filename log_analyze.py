# log_analyze.py
# 현재 agent.py가 남기는 로그 형식:
# [2025-10-31 19:12:34] [EP 000123] level=800 len=798 success=1 eps=0.1730
# [2025-10-31 19:12:35] [CKPT] saved ckpt_auto/ep000200_L0.pkl
# [2025-10-31 19:12:36] [LEVEL UP] from=400
#
# 위 형식만 파싱해서 에피소드 길이, 성공여부, eps, 레벨 변화를 본다.

import re
import os
import pandas as pd
import matplotlib.pyplot as plt

LOG_PATH = os.path.join("logs", "avoid_train.log")

# 날짜, 시간은 무시하고 뒷부분만 본다.
EP_PATTERN = re.compile(
    r"\[EP\s+(?P<ep>\d+)\]\s+level=(?P<level>\d+)\s+len=(?P<len>\d+)\s+success=(?P<success>[01])\s+eps=(?P<eps>[0-9.]+)"
)
CKPT_PATTERN = re.compile(
    r"\[CKPT\]\s+saved\s+.+ep(?P<ep>\d+)_L(?P<level>\d+)\.pkl"
)
LEVELUP_PATTERN = re.compile(
    r"\[LEVEL UP\]\s+from=(?P<from>\d+)"
)


def main():
    if not os.path.exists(LOG_PATH):
        print(f"로그 파일이 없습니다: {LOG_PATH}")
        return

    ep_rows = []
    ckpt_rows = []
    levelups = []

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            m = EP_PATTERN.search(line)
            if m:
                ep_rows.append(
                    {
                        "ep": int(m.group("ep")),
                        "level": int(m.group("level")),
                        "len": int(m.group("len")),
                        "success": int(m.group("success")),
                        "eps": float(m.group("eps")),
                    }
                )
                continue

            m = CKPT_PATTERN.search(line)
            if m:
                ckpt_rows.append(
                    {
                        "ep": int(m.group("ep")),
                        "level": int(m.group("level")),
                    }
                )
                continue

            m = LEVELUP_PATTERN.search(line)
            if m:
                levelups.append({"from": int(m.group("from"))})
                continue

    ep_df = pd.DataFrame(ep_rows)
    ckpt_df = pd.DataFrame(ckpt_rows)

    if ep_df.empty:
        print("에피소드 로그가 없습니다.")
        return

    # 파생
    ep_df = ep_df.sort_values("ep").reset_index(drop=True)
    ep_df["len_ma200"] = ep_df["len"].rolling(200, min_periods=1).mean()
    ep_df["success_ma200"] = ep_df["success"].rolling(200, min_periods=1).mean()

    # 그림
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    # 1) len
    ax = axes[0]
    ax.plot(ep_df["ep"], ep_df["len"], label="len")
    ax.plot(ep_df["ep"], ep_df["len_ma200"], label="MA200")
    ax.set_title("Episode length")
    ax.set_xlabel("episode")
    ax.set_ylabel("len")
    ax.legend()

    # 2) success rate (MA)
    ax = axes[1]
    ax.plot(ep_df["ep"], ep_df["success_ma200"], label="success_MA200")
    ax.set_ylim(0, 1)
    ax.set_title("Success moving avg (200)")
    ax.set_xlabel("episode")
    ax.set_ylabel("success rate")
    ax.legend()

    # 3) epsilon
    ax = axes[2]
    ax.plot(ep_df["ep"], ep_df["eps"], label="eps")
    ax.set_title("Epsilon")
    ax.set_xlabel("episode")
    ax.set_ylabel("eps")
    ax.legend()

    # 4) level
    ax = axes[3]
    ax.plot(ep_df["ep"], ep_df["level"], label="level")
    if not ckpt_df.empty:
        ax.scatter(ckpt_df["ep"], ckpt_df["level"], marker="x", label="ckpt")
    ax.set_title("Level progress")
    ax.set_xlabel("episode")
    ax.set_ylabel("level steps")
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
