# log_analyze.py
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt

EP_PATTERN = re.compile(
    r"\[EP (?P<ep>\d+)\]\s+len=(?P<len>\d+)\s+R=(?P<R>[-+]?\d+\.?\d*)\s+danger=(?P<danger>\d+)\s+blocked=(?P<blocked>\d+)"
)
CKPT_PATTERN = re.compile(
    r"\[CKPT\]\s+ep=(?P<ep>\d+)\s+avg_len=(?P<avg_len>[-+]?\d+\.?\d*)\s+max_len=(?P<max_len>\d+)\s+epsilon=(?P<epsilon>[-+]?\d+\.?\d*)"
)


def main():
    if len(sys.argv) < 2:
        print("usage: python log_analyze.py <logfile>")
        sys.exit(1)

    logfile = sys.argv[1]
    ep_rows, ckpt_rows = [], []

    with open(logfile, "r", encoding="utf-8") as f:
        for line in f:
            m = EP_PATTERN.search(line)
            if m:
                ep_rows.append({
                    "ep": int(m.group("ep")),
                    "len": int(m.group("len")),
                    "R": float(m.group("R")),
                    "danger": int(m.group("danger")),
                    "blocked": int(m.group("blocked")),
                })
                continue
            m = CKPT_PATTERN.search(line)
            if m:
                ckpt_rows.append({
                    "ep": int(m.group("ep")),
                    "avg_len": float(m.group("avg_len")),
                    "max_len": int(m.group("max_len")),
                    "epsilon": float(m.group("epsilon")),
                })

    ep_df = pd.DataFrame(ep_rows)
    ckpt_df = pd.DataFrame(ckpt_rows)

    # -------------------------
    # 파생 컬럼
    # -------------------------
    if not ep_df.empty:
        ep_df["danger_rate"] = ep_df.apply(
            lambda r: r["danger"] / r["len"] if r["len"] > 0 else 0.0, axis=1
        )
        ep_df["blocked_rate"] = ep_df.apply(
            lambda r: r["blocked"] / r["len"] if r["len"] > 0 else 0.0, axis=1
        )
        ep_df["len_ma200"] = ep_df["len"].rolling(200, min_periods=1).mean()
        ep_df["R_ma200"] = ep_df["R"].rolling(200, min_periods=1).mean()
        # 롤링 표준편차도 하나 만들어두자
        ep_df["len_std200"] = ep_df["len"].rolling(200, min_periods=1).std().fillna(0.0)
        # 위험도*길이 같은 composite도 하나
        ep_df["danger_load"] = ep_df["danger_rate"] * ep_df["len"]

    if not ckpt_df.empty:
        ckpt_df["avg_len_diff"] = ckpt_df["avg_len"].diff()

    # -------------------------
    # 플롯 9개
    # -------------------------
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    # 1) EP len + MA
    ax = axes[0]
    if not ep_df.empty:
        ax.plot(ep_df["ep"], ep_df["len"], label="len")
        ax.plot(ep_df["ep"], ep_df["len_ma200"], label="MA200")
    ax.set_title("1. EP len")
    ax.set_xlabel("episode")
    ax.set_ylabel("len")
    ax.legend(loc="best")

    # 2) EP reward + MA
    ax = axes[1]
    if not ep_df.empty:
        ax.plot(ep_df["ep"], ep_df["R"], label="R")
        ax.plot(ep_df["ep"], ep_df["R_ma200"], label="MA200")
    ax.set_title("2. EP reward")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.legend(loc="best")

    # 3) Danger / Blocked count
    ax = axes[2]
    if not ep_df.empty:
        ax.plot(ep_df["ep"], ep_df["danger"], label="danger")
        ax.plot(ep_df["ep"], ep_df["blocked"], label="blocked")
    ax.set_title("3. Danger / Blocked (count)")
    ax.set_xlabel("episode")
    ax.set_ylabel("count")
    ax.legend(loc="best")

    # 4) Rates
    ax = axes[3]
    if not ep_df.empty:
        ax.plot(ep_df["ep"], ep_df["danger_rate"], label="danger_rate")
        ax.plot(ep_df["ep"], ep_df["blocked_rate"], label="blocked_rate")
    ax.set_title("4. Danger / Blocked (rate)")
    ax.set_xlabel("episode")
    ax.set_ylabel("rate")
    ax.legend(loc="best")

    # 5) Len hist
    ax = axes[4]
    if not ep_df.empty:
        ax.hist(ep_df["len"], bins=50)
    ax.set_title("5. Len hist")
    ax.set_xlabel("len")
    ax.set_ylabel("freq")

    # 6) Reward hist
    ax = axes[5]
    if not ep_df.empty:
        ax.hist(ep_df["R"], bins=50)
    ax.set_title("6. Reward hist")
    ax.set_xlabel("R")
    ax.set_ylabel("freq")

    # 7) len vs R
    ax = axes[6]
    if not ep_df.empty:
        ax.scatter(ep_df["len"], ep_df["R"], s=5)
    ax.set_title("7. len vs R")
    ax.set_xlabel("len")
    ax.set_ylabel("R")

    # 8) CKPT가 있으면 CKPT, 없으면 len_std200
    ax = axes[7]
    if not ckpt_df.empty:
        ax.plot(ckpt_df["ep"], ckpt_df["avg_len"], label="avg_len")
        ax.plot(ckpt_df["ep"], ckpt_df["max_len"], label="max_len")
        ax.set_title("8. CKPT len")
        ax.set_xlabel("episode")
        ax.set_ylabel("len")
        ax.legend(loc="best")
    else:
        # CKPT 없을 때는 롤링 표준편차로 변동성 보기
        if not ep_df.empty:
            ax.plot(ep_df["ep"], ep_df["len_std200"], label="len_std200")
        ax.set_title("8. Len rolling std (200)")
        ax.set_xlabel("episode")
        ax.set_ylabel("std")
        ax.legend(loc="best")

    # 9) CKPT Δlen & ε / or danger_load
    ax = axes[8]
    if not ckpt_df.empty:
        ax2 = ax.twinx()
        ax.plot(ckpt_df["ep"], ckpt_df["avg_len_diff"], label="Δavg_len")
        ax2.plot(ckpt_df["ep"], ckpt_df["epsilon"], "--", label="eps")
        ax.set_ylabel("Δavg_len")
        ax2.set_ylabel("epsilon")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best")
        ax.set_title("9. CKPT Δlen & ε")
        ax.set_xlabel("episode")
    else:
        # CKPT 없으면 위험도*길이 트렌드
        if not ep_df.empty:
            ax.plot(ep_df["ep"], ep_df["danger_load"], label="danger_load")
        ax.set_title("9. danger_load = danger_rate * len")
        ax.set_xlabel("episode")
        ax.set_ylabel("danger_load")
        ax.legend(loc="best")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
