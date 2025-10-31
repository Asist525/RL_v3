# evaluate.py
# python evaluate.py 만 해도 돌아가게 하드코딩한 버전
# 순서:
# 1) ckpt_final/ 안에서 가장 최근 .pkl 찾기
# 2) 없으면 ckpt_best/
# 3) 그것도 없으면 ckpt_auto/
# 4) 그래도 없으면 에러

import os
import sys
import kymnasium as kym
from agent import YourAgent   # agent.py랑 같은 폴더라고 가정

SEARCH_DIRS = [
    os.path.join(".", "ckpt_final"),
    os.path.join(".", "ckpt_best"),
    os.path.join(".", "ckpt_auto"),
]

def find_latest_pkl() -> str | None:
    candidates = []
    for d in SEARCH_DIRS:
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.lower().endswith(".pkl"):
                full = os.path.join(d, name)
                mtime = os.path.getmtime(full)
                candidates.append((mtime, full))
    if not candidates:
        return None
    # 가장 최근 파일
    candidates.sort(reverse=True)
    return candidates[0][1]

def main():
    # 인자가 있으면 그걸로, 없으면 하드코딩 탐색
    if len(sys.argv) >= 2:
        pkl_path = sys.argv[1]
    else:
        pkl_path = find_latest_pkl()
        if pkl_path is None:
            print("pkl을 찾을 수 없습니다. ckpt_final/ ckpt_best/ ckpt_auto/ 중 하나에 .pkl을 두세요.")
            sys.exit(1)

    print(f"[INFO] using checkpoint: {pkl_path}")
    agent = YourAgent.load(pkl_path)

    # 평가 시에는 탐색 0으로 고정 (우리 YourAgent는 epsilon 안 받도록 짜여 있으므로 생략)
    kym.evaluate(
        env_id="kymnasium/AvoidBlurp-Normal-v0",
        agent=agent,
        render_mode="human",
        bgm=True,
        obs_type="custom",
    )

if __name__ == "__main__":
    main()
