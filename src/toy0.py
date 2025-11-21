import numpy as np
import matplotlib.pyplot as plt

# --- 設定 ---
STEPS = 100
CHANGE_POINT = 40  # 40ステップ目で「人生の激変」が起きる
IMPACT_SIZE = 5.0  # 衝撃の大きさ

# --- 環境（世界）の生成 ---
# 最初は平穏(0)だが、途中で世界がガラッと変わる(5.0)
world_state = np.zeros(STEPS)
world_state[CHANGE_POINT:] = IMPACT_SIZE


# --- クラス定義 ---

class Agent:
    def __init__(self, name, mode):
        self.name = name
        self.mode = mode
        self.self_state = 0.0  # 自我の状態（初期値）
        self.history = []  # 自分の履歴
        self.plasticity_history = []  # 可塑性（変わりやすさ）の履歴

    def update(self, external_reality):
        # 1. 意味の不整合（Discrepancy）を感知する
        # 「世界」と「自分」のズレ
        discrepancy = abs(external_reality - self.self_state)

        # 2. 自我モードによる「可塑性（Plasticity）」の決定
        if self.mode == "Static":
            # 固定的自我：
            # 変化を恐れる。ズレが大きくても、頑なに変わろうとしない（防衛）
            # 常に低い学習率で固定
            plasticity = 0.05

        elif self.mode == "Generative":
            # 生成的自我：
            # ズレ（意味ショック）が大きいほど、「これは自分を変えるべき時だ」と判断。
            # 不整合の大きさに応じて、一時的に可塑性を上げる（痕跡を受け入れる）
            # 基本値 0.05 + 衝撃反応係数
            plasticity = 0.05 + (discrepancy * 0.15)
            # ただし、崩壊しないよう上限はある
            plasticity = min(plasticity, 0.8)

        # 3. 自己の再構成（Update）
        # S(t+1) = S(t) + Plasticity * (Target - S(t))
        direction = external_reality - self.self_state
        self.self_state += plasticity * direction

        # 記録
        self.history.append(self.self_state)
        self.plasticity_history.append(plasticity)


# --- シミュレーション実行 ---
agent_static = Agent("Static Self (固定的)", "Static")
agent_generative = Agent("Generative Self (生成的)", "Generative")

for t in range(STEPS):
    reality = world_state[t]
    agent_static.update(reality)
    agent_generative.update(reality)

# --- 可視化 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 上段：世界と自我の状態
ax1.plot(world_state, 'k--', label='World Reality (環境)', alpha=0.6, linewidth=1)
ax1.plot(agent_static.history, 'r-', label='Static Self (防衛的)', linewidth=2)
ax1.plot(agent_generative.history, 'b-', label='Generative Self (編集可能)', linewidth=2)
ax1.set_title('自我の再構成プロセス (Self-Reconstruction Process)', fontsize=14)
ax1.set_ylabel('意味の状態 (Internal State)')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# 下段：可塑性の変化（どれだけ心を開いたか）
ax2.plot(agent_static.plasticity_history, 'r-', label='Static Plasticity', linewidth=2)
ax2.plot(agent_generative.plasticity_history, 'b-', label='Generative Plasticity (Trace Acceptance)', linewidth=2)
ax2.set_title('可塑性の変化：痕跡の受容 (Acceptance of Imprint)', fontsize=14)
ax2.set_ylabel('可塑性 (Plasticity)')
ax2.set_xlabel('時間 (Time Step)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 注釈
ax1.annotate('Imprint Event\n(Loss/Trauma)', xy=(CHANGE_POINT, 0), xytext=(CHANGE_POINT + 5, 2),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
plt.show()