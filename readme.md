
---

# Avoid Blurp – Expected SARSA Agent (최종본)

Python 3.12 / `kymnasium` 1.0.9 기준
관찰: `obs_type="custom"`
액션: `{0: stay, 1: left, 2: right}`
**agent.py 단일 파일**로 학습 / 이어학습 / 중간시연 / 제출 시연까지 전부 처리하는 구조.

---

## 폴더 구조

```text
.
├─ agent.py           # 학습 + 시연 일체형 (Expected SARSA, 탭형)
├─ evaluate.py        # kym.evaluate 호출용 래퍼
├─ log_analyze.py     # (옵션) 로그 파서
├─ ckpt/              # 2000ep마다 떨어지는 .pkl
└─ logs/              # 학습 로그 저장
```

※ 예전 이름이 `evalute.py`면 `evaluate.py`로 바꿀 것.

---

## 설치

```powershell
pip install -U kymnasium gymnasium numpy pygame
```

제출용 의존성:

```powershell
pip list --format=freeze > requirements.txt
```

---

## 1. 기본 학습 (3만 에피소드 + 2000ep마다 저장)

```powershell
python agent.py train `
  --episodes 30000 `
  --save-dir .\ckpt `
  --ckpt-every 2000 `
  --log-file .\logs\avoid_train.log `
  --epsilon-base 0.2
```

이렇게 돌면:

* 20ep마다

  ```text
  [EP 220] len=152 R=-987.35 danger=0 blocked=0 died=1 eps=0.241
  ```
* 2000ep마다

  ```text
  [CKPT] ep=2000 avg_len=133.2 max_len=270 death_rate=1.000 epsilon=0.246 saved=ckpt\avoid_ep002000_eps0.246.pkl
  ```

이 두 줄 포맷만 유지되면 나중에 `log_analyze.py`로 전처리하기 편하다.

### 옵션 설명

* `--episodes N` : 이번에 **추가로** 돌릴 에피소드 수
* `--save-dir` : pkl 떨어질 폴더
* `--ckpt-every 2000` : 2000ep마다 `.pkl` 저장
* `--log-file path` : 콘솔에 찍는 걸 파일에도 같이 찍기
* `--epsilon-base 0.2` : 시작 ε 강제 세팅 (pkl에서 불러온 ε 무시하고 이걸로 시작)

---

## 2. 이어학습 (중간 pkl에서 계속)

이미 `ckpt\avoid_ep030000_eps0.080.pkl` 이런 게 있을 때 **거기서부터** 더 돌리고 싶으면:

```powershell
python agent.py train `
  --load .\ckpt\avoid_ep030000_eps0.080.pkl `
  --episodes 10000 `
  --save-dir .\ckpt `
  --ckpt-every 2000 `
  --log-file .\logs\avoid_continue.log `
  --epsilon-base 0.2
```

포인트:

1. `--load` 주면 pkl 안에 있던

   * Q 테이블
   * 방문수(visit_counts)
   * ε
   * 지금까지 학습된 ep 수
     전부 가져온다.
2. 근데 그 ε가 너무 작으면 탐색이 안 되니까 **`--epsilon-base 0.2`로 다시 키워버리는 패턴**을 쓴다.
3. 이후 동작은 처음 학습이랑 똑같이 2000ep마다 저장한다.

---

## 3. “안정화 모드”로 조금만 더

제출 직전에 “이제 탐색 그만, 거의 결정만 해”로 5,000ep 정도만 더 밀고 싶을 때:

```powershell
python agent.py train `
  --load .\ckpt\avoid_ep040000_eps0.200.pkl `
  --episodes 5000 `
  --save-dir .\ckpt `
  --ckpt-every 2000 `
  --log-file .\logs\avoid_stable.log `
  --low-eps `
  --epsilon-low 0.06
```

* `--low-eps` : 학습 루프에서 **매 에피소드마다 ε를 강제로 이 값으로 덮어씀**
* `--epsilon-low 0.06` : 기본 0.06인데, 0.03 같은 값으로 더 닫을 수도 있음

이렇게 하면 Q는 거의 안 흔들리고, 자주 나오는 상태만 조금씩 다듬는 형태가 된다.

---

## 4. 무한 학습 모드 (6일 플랜용)

과제 기한까지 계속 돌릴 때:

```powershell
python agent.py train `
  --infinite `
  --save-dir .\ckpt `
  --ckpt-every 2000 `
  --log-file .\logs\avoid_infinite.log `
  --epsilon-base 0.2
```

* 끄기 전까지 무한 루프
* 2000ep마다 계속 `.pkl` 생성 → 중간중간 `evaluate.py`로 바로 테스트 가능
* 6일 굴리는 플랜이면 대략:

  * 1~2일차: ε=0.2 고정(또는 0.3)로 “많이 보기”
  * 3~4일차: 여전히 0.2, 적 개수/패턴 늘려서 노출
  * 5~6일차: `--low-eps`로 닫아서 제출용 안정화

---

## 5. 시연 (과제에서 실제로 돌리는 형식)

```python
# evaluate.py
import kymnasium as kym
from agent import YourAgent
import sys

def main():
    if len(sys.argv) < 2:
        print("usage: python evaluate.py <pkl_path>")
        raise SystemExit(1)
    pkl_path = sys.argv[1]
    agent = YourAgent.load(pkl_path)
    kym.evaluate(
        env_id='kymnasium/AvoidBlurp-Normal-v0',
        agent=agent,
        render_mode='human',
        bgm=True,
        obs_type='custom'
    )

if __name__ == "__main__":
    main()
```

실행:

```powershell
python .\evaluate.py ".\ckpt\avoid_ep002000_eps0.246.pkl"
```

이게 과제 문서에 있는 평가 루틴이랑 1:1로 맞는다.

---

## 6. 내부 구조 설명 (짧게)

### 6.1 상태 인코더

* 플레이어 x → 15등분
* 위에서 떨어지는 적 중 **가까운 순서로 2개만** 본다

  * 각 적은 `(좌/좌가까/정면/우가까/우)` 5단계 × `(가까움/중간/멀다)` 3단계
  * 없으면 더미값 넣음
* 가장 가까운 적이 내 위쪽에 있고 x도 겹치면 `danger_flag=1`
* 최종 상태 수는 “수천~1만대”로 고정되게 설계 → 탭 SARSA 가능

(※ 나중에 3~5기까지 열고 싶으면 `max_enemies`만 늘리면 되지만, 그때는 pkl 재학습 필요하다고 리드미에 명시해두면 안전)

### 6.2 액션 마스크

1. 화면 왼/오른 경계 밖으로 나가는 이동 막음
2. “가장 가까운 적 1개”에 대해

   * 지금 위치
   * 한 칸 왼쪽 이동
   * 한 칸 오른쪽 이동
     을 1프레임 예측해서 겹치면 그 액션 막음
3. 그런데 이렇게 하면 3개 다 막혀서 학습이 멈출 수 있어서

   * 0개면 stay 열어주고
   * 1개면 하나 더 살려서 **최소 2개** 보장

이게 네 로그에 있는 `blocked=3` 같은 숫자를 만드는 부분.

### 6.3 보상

* 기본 생존: `+0.1`
* 위험존(가까운 적이 위에 있고 x 겹침): `- λ * 1/(dy+1)`
* 방향 전환: `-0.05` (옵션)
* 종료(충돌/타임아웃): `-1000`
* 그래서 에피소드 리워드는 대체로 `-995 ~ -979` 근처에 모인다
  → “길게 살았는지”는 **len=...** 으로 본다.

### 6.4 학습 (Expected SARSA)

* 업데이트식은 표준 Expected SARSA

* α는 “많이 본 상태/액션일수록 작게”:

  ```text
  α = max(α_min, 1 / (alpha_c + visit_count))
  ```

  여기서 `alpha_c=10`, `alpha_min=0.02`

* ε는 기본은 이렇게 설계돼 있었음:

  ```text
  ε₀ = 0.3
  ε ← max(0.02, ε * 0.999)


## 7. 로그 포맷

에피소드 로그:

```text
[EP 680] len=157 R=-987.01 danger=4 blocked=3 died=1 eps=0.152
```

필드 의미:

* `len` : 이번 에피소드가 몇 step 버텼나
* `R` : shaping까지 포함한 총 보상
* `danger` : 인코더가 “위험”이라고 표시한 프레임 수
* `blocked` : 마스크가 전부 막아서 복구까지 간 횟수
* `eps` : 그 시점 ε

체크포인트 로그:

```text
[CKPT] ep=2000 avg_len=133.2 max_len=270 death_rate=1.000 epsilon=0.246 saved=...
```

→ 이게 2000, 4000, 6000, … 단위로 늘어나야 “느릿느릿하지만 배우고 있다”라고 판단.

---

## 8. 자주 쓰는 명령

```powershell
# 3만 에피소드
python agent.py train --episodes 30000 --save-dir .\ckpt --ckpt-every 2000 --log-file .\logs\train_30000.log --epsilon-base 0.2

# 이어학습
python agent.py train --load .\ckpt\avoid_ep030000_eps0.200.pkl --episodes 10000 --save-dir .\ckpt --ckpt-every 2000 --log-file .\logs\train_more.log --epsilon-base 0.2

# 무한
python agent.py train --infinite --save-dir .\ckpt --ckpt-every 2000 --log-file .\logs\train_inf.log --epsilon-base 0.2

# 시연
python evaluate.py ".\ckpt\avoid_ep002000_eps0.246.pkl"
```

python agent.py train --load .\ckpt\avoid_ep030000_eps0.200.pkl --episodes 10000 --save-dir .\ckpt --ckpt-every 2000 --log-file .\logs\train_more.log --epsilon-base 0.2


---
