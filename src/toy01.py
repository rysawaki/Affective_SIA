import numpy as np
import matplotlib.pyplot as plt

# --- 設定 ---
STEPS = 120
CHANGE_POINT = 50  # ここで「人生の激変」
DIM = 3            # 自我・世界とも3次元の意味ベクトルで表現

# 「世界」の3次元の意味軸イメージ：
# 0: 安定 vs 不安定
# 1: 関係性 (つながり vs 孤立)
# 2: 自己価値 (肯定 vs 否定)

IMPACT_VEC = np.array([+5.0, -3.0, +4.0])  # 激変後の世界の「意味ベクトル」

# --- 環境（世界）の生成 ---
world_state = np.zeros((STEPS, DIM))
world_state += np.random.normal(0, 0.1, size=(STEPS, DIM))  # 微小なゆらぎ

# CHANGE_POINT 以降で、世界の意味構造が変わる
world_state[CHANGE_POINT:] += IMPACT_VEC


def softmax(x):
    x = np.array(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-8)


# --- 固定的自我（従来モデルに近い） ---
class StaticAgent:
    """
    外界をそのまま受け取り、低い一定学習率で追従しようとするだけの自我。
    痕跡も解釈層もなく、単なる「調整」。
    """
    def __init__(self, name="Static"):
        self.name = name
        self.self_state = np.zeros(DIM)
        self.history = []
        self.plasticity_history = []

    def update(self, external_vec):
        # 世界をそのまま「事実」として内面化（解釈なし）
        meaning = external_vec.copy()

        discrepancy = meaning - self.self_state
        shock = np.linalg.norm(discrepancy)

        # 固定的自我：いつも同じ低い学習率
        plasticity = 0.05

        self.self_state += plasticity * discrepancy

        self.history.append(self.self_state.copy())
        self.plasticity_history.append(plasticity)


# --- 生成的自我 (Self-Imprint Attribution 風) ---
class GenerativeSIAAgent:
    """
    - 世界を一度「意味解釈」してから自己に取り込む
    - 可塑性は
        ・ズレの大きさ（ショック）
        ・どの軸にどれだけ痕跡が刻まれているか
      に応じて次元ごとに変化
    - 痕跡（trace）は履歴依存・不可逆な変化として蓄積される
    """
    def __init__(self, name="Generative-SIA"):
        self.name = name
        self.self_state = np.zeros(DIM)

        # 「世界をどう解釈しがちか」というバイアス
        # 例: いつも少し「自己否定寄り」に解釈してしまう など
        self.interpret_bias = np.array([-0.5, -0.2, -0.7])

        # 痕跡ベクトル：どの軸にどれだけ強い刻印が残っているか
        self.trace = np.zeros(DIM)

        # 記録
        self.self_history = []
        self.trace_history = []
        self.plasticity_history = []   # 各次元ごとの可塑性
        self.shock_history = []

    def interpret(self, external_vec):
        """
        世界の「生」の変化を、その人特有のバイアスを通して意味解釈する層。
        ここで既に「世界＝事実」ではなく「世界＝意味」になる。
        """
        # シンプルにはバイアスを足すだけだが、ここに非線形を入れてもよい
        meaning = external_vec + self.interpret_bias
        return meaning

    def update(self, external_vec):
        # 1. 世界を「意味」として解釈
        meaning = self.interpret(external_vec)

        # 2. 自己とのズレ（意味の不整合）
        discrepancy = meaning - self.self_state
        shock = np.linalg.norm(discrepancy)  # ショックの総強度（スカラー）

        # 3. どの次元をどれだけ変えるかの「選択」をする
        #
        #   attention ~ |ズレ| + α * |痕跡|
        #   → すでに強く刻まれている軸 + 今回ズレが大きい軸 に注意が向かう
        #
        attention_logits = np.abs(discrepancy) + 0.5 * np.abs(self.trace)
        attention = softmax(attention_logits)  # 各次元の「注目度」（足して1）

        # 4. 可塑性ベクトルを決定
        #
        #   base_plasticity: 常にある程度は変わりうる余地
        #   shock_gain    : ショックが大きいほど変化しやすくなる
        #
        base_plasticity = 0.03
        shock_gain = 0.6 * (shock / (1.0 + shock))  # 0〜0.6くらいに収まる

        plasticity_vec = base_plasticity + shock_gain * attention
        plasticity_vec = np.clip(plasticity_vec, 0.0, 0.8)

        # 5. 自己更新（次元ごとに異なる学習率）
        self.self_state += plasticity_vec * discrepancy

        # 6. 痕跡の更新（履歴依存・ほぼ不可逆）
        #
        #   ・ズレが大きく、ショックが強いほど痕跡が刻まれる
        #   ・符号付きで残る（どちら向きの体験だったか）
        #
        imprint_strength = np.tanh(shock)   # ショックが大きいほど1に近づく
        self.trace += 0.1 * discrepancy * imprint_strength

        # ある程度の自然な「風化」は入れておくが、完全には消えない
        self.trace *= 0.995

        # 記録
        self.self_history.append(self.self_state.copy())
        self.trace_history.append(self.trace.copy())
        self.plasticity_history.append(plasticity_vec.copy())
        self.shock_history.append(shock)


# --- シミュレーション実行 ---
agent_static = StaticAgent("Static")
agent_sia = GenerativeSIAAgent("Generative-SIA")

for t in range(STEPS):
    reality_vec = world_state[t]
    agent_static.update(reality_vec)
    agent_sia.update(reality_vec)

# numpy 配列に変換（可視化しやすく）
static_self_hist = np.array(agent_static.history)          # (T, DIM)
sia_self_hist    = np.array(agent_sia.self_history)        # (T, DIM)
sia_trace_hist   = np.array(agent_sia.trace_history)       # (T, DIM)
sia_plast_hist   = np.array(agent_sia.plasticity_history)  # (T, DIM)
sia_shock_hist   = np.array(agent_sia.shock_history)       # (T,)


# --- 可視化 ---
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# ① 自己状態ノルムの時間変化（世界とどれくらい整合的かの大まかな指標）
axes[0].plot(
    np.linalg.norm(world_state, axis=1),
    'k--', label='World (norm)', alpha=0.6
)
axes[0].plot(
    np.linalg.norm(static_self_hist, axis=1),
    'r-', label='Static Self (norm)'
)
axes[0].plot(
    np.linalg.norm(sia_self_hist, axis=1),
    'b-', label='Generative-SIA Self (norm)'
)
axes[0].axvline(CHANGE_POINT, color='gray', linestyle=':', alpha=0.7)
axes[0].set_ylabel('||State||')
axes[0].set_title('自己状態の大きさの推移')
axes[0].legend()
axes[0].grid(alpha=0.3)

# ② 生成的自我の各次元の自己状態（どの軸がどう変容したか）
axes[1].plot(sia_self_hist[:, 0], label='Self dim 0 (安定感)')
axes[1].plot(sia_self_hist[:, 1], label='Self dim 1 (関係性)')
axes[1].plot(sia_self_hist[:, 2], label='Self dim 2 (自己価値)')
axes[1].axvline(CHANGE_POINT, color='gray', linestyle=':', alpha=0.7)
axes[1].set_ylabel('Self components')
axes[1].set_title('生成的自我の内部構造（各意味軸の推移）')
axes[1].legend()
axes[1].grid(alpha=0.3)

# ③ 痕跡とショック
axes[2].plot(
    np.linalg.norm(sia_trace_hist, axis=1),
    label='Trace norm (痕跡の強度)'
)
axes[2].plot(
    sia_shock_hist,
    label='Shock (意味不整合の強度)', alpha=0.7
)
axes[2].axvline(CHANGE_POINT, color='gray', linestyle=':', alpha=0.7)
axes[2].set_ylabel('Value')
axes[2].set_xlabel('Time step')
axes[2].set_title('痕跡の蓄積とショックの推移')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()