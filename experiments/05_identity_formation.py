import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# --- 研究用パラメータ設定 (Identity Phase) ---
STEPS = 300
DIM_MEANING = 3  # 知覚・痕跡の次元 (S, T)
DIM_AFFECT = 5  # 情動の次元 (A) [悲哀, 希望, 孤立, 連帯, 創造]

DT = 0.1
ALPHA_TRACE = 1.0
BETA_ACTION = 1.5
GAMMA_CREATION = 0.8

# Identity形成率
ETA_IDENTITY = 0.05

np.random.seed(45)  # 意図的なすれ違いを作るためのシード調整


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum() + 1e-8)


def cosine_similarity(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return np.dot(v1, v2) / (n1 * n2)


class Identity_SIA_Agent:
    def __init__(self, name, affect_matrix_seed=0):
        self.name = name

        # 1. 状態変数
        self.S = np.zeros(DIM_MEANING)  # Self State
        self.T = np.zeros(DIM_MEANING)  # Trace (Physical/Memory)
        self.A = np.zeros(DIM_AFFECT)  # Affect Vector (Meaning) ★Vectorized
        self.I = np.zeros(DIM_AFFECT)  # Narrative Identity (Structure) ★New

        self.last_action = np.zeros(DIM_MEANING)

        # 2. 構造パラメータ (Trace -> Affect 変換行列)
        # 痕跡(3次元)をどういう感情(5次元)に解釈するか？という「性格」
        np.random.seed(affect_matrix_seed)
        self.W_T2A = np.random.uniform(-0.5, 0.8, (DIM_AFFECT, DIM_MEANING))

        # 履歴
        self.history = {
            'P_self': [], 'Action_Norm': [],
            'Affect_Vec': [], 'Identity_Norm': [], 'Discrepancy': []
        }

    def compute_attribution(self, E_meaning):
        diff = E_meaning - self.S
        dist = np.linalg.norm(diff)
        trace_mag = np.linalg.norm(self.T)
        action_mag = np.linalg.norm(self.last_action)

        logit = -dist + (ALPHA_TRACE * trace_mag) + (BETA_ACTION * action_mag)
        return sigmoid(logit), dist, diff

    def step(self, world_vec, interact_force=None, shared_resonance=0.0):
        # --- 1. 解釈と帰属 ---
        E_meaning = world_vec.copy()
        if interact_force is not None:
            E_meaning += interact_force

        P_self, dist, diff = self.compute_attribution(E_meaning)

        # --- 2. 痕跡形成 (Trace) ---
        # 物理的な痛みの記録
        attention = softmax(np.abs(diff) + np.abs(self.T))
        self.S += 0.5 * P_self * attention * diff * DT

        shock = np.tanh(dist)
        d_T = (shock * diff * P_self * DT) - (0.01 * self.T * DT)
        self.T += d_T

        # --- 3. 情動ベクトルの生成 (Vectorized Affect) ---
        # A(t) = decay * A(t-1) + W * T * P_self
        # つまり、痕跡(T)が変換行列(W)を通って、特定の「感情の質」になる

        affect_input = np.dot(self.W_T2A, self.T) * P_self

        # 自己回帰（物語的持続性）+ 入力
        self.A = 0.9 * self.A + 0.1 * affect_input

        # --- 4. アイデンティティの形成 (Identity Update) ★Core Theory ---
        # Identityは「一人」では育たない。「共有された共鳴」があった時だけ、
        # その瞬間のAffectがIdentityとして結晶化する。
        # I(t+1) = I(t) + η * Shared * A(t)

        if shared_resonance > 0.01:
            d_I = ETA_IDENTITY * shared_resonance * self.A * DT
            self.I += d_I

        # --- 5. 行動生成 ---
        # 行動は Affect の強さと、Identity の安定感から生まれる
        # Identityが確立してくると、行動はブレなくなる（今回は簡易実装）

        drive = np.linalg.norm(self.A) * GAMMA_CREATION
        action_vec = (self.S - world_vec) * drive

        self.last_action = action_vec

        # 履歴
        self.history['P_self'].append(P_self)
        self.history['Action_Norm'].append(np.linalg.norm(action_vec))
        self.history['Affect_Vec'].append(self.A.copy())
        self.history['Identity_Norm'].append(np.linalg.norm(self.I))
        self.history['Discrepancy'].append(dist)

        return action_vec


# --- 実験セットアップ ---

# 異なる「解釈特性(W)」を持つ二人
agent_A = Identity_SIA_Agent("Protagonist", affect_matrix_seed=42)
agent_B = Identity_SIA_Agent("Partner", affect_matrix_seed=99)

world_state = np.zeros(DIM_MEANING)
shared_engram_history = []
resonance_components = []  # [Action_Sync, Affect_Sync]

# シナリオ
# 0-50: 平穏
# 50-60: 共通のトラウマ体験 (Impact)
# 60-150: すれ違い (Actionは合うがAffectが合わない)
# 150-: 真の共鳴 (Identity形成)

print("Running SIA Identity Simulation...")

for t in range(STEPS):
    # --- 世界 ---
    if 50 <= t < 60:
        target = np.array([5.0, 5.0, 5.0])  # 強い衝撃
        world_state = 0.8 * world_state + 0.2 * target
    else:
        world_state *= 0.95  # 減衰

    # --- Interaction (t >= 100) ---
    interact_force_A = None
    current_shared = 0.0
    cos_act = 0.0
    cos_aff = 0.0

    if t >= 100:
        # 互いに影響し合う
        action_B = agent_B.step(world_state, interact_force=agent_A.last_action * 0.3, shared_resonance=0)
        interact_force_A = action_B * 0.3

        # --- Shared Engram Formula (Final Ver) ---
        # Shared = P1 * P2 * cos(Action) * cos(Affect) * min(|A1|, |A2|)

        p1 = agent_A.history['P_self'][-1]
        p2 = agent_B.history['P_self'][-1]

        # 1. 行動の同期
        cos_act = cosine_similarity(agent_A.last_action, agent_B.last_action)

        # 2. 感情(意味)の同期 ★Critical
        # 互いの感情ベクトルの向きが合っているか？
        cos_aff = cosine_similarity(agent_A.A, agent_B.A)

        # 3. 情動強度
        min_depth = min(np.linalg.norm(agent_A.A), np.linalg.norm(agent_B.A))

        # 統合共鳴値 (負の共鳴はゼロとする)
        if cos_act > 0 and cos_aff > 0:
            current_shared = p1 * p2 * cos_act * cos_aff * min_depth
        else:
            current_shared = 0.0

    # ResonanceをフィードバックしてIdentity更新に使う
    shared_engram_history.append(current_shared)
    resonance_components.append([cos_act, cos_aff])

    # Agent A Update (Shared値を渡す -> Identity形成へ)
    action_A = agent_A.step(world_state, interact_force_A, shared_resonance=current_shared)

    # 世界更新
    world_state += action_A * 0.1

# --- 可視化 ---
fig = plt.figure(figsize=(12, 14))
gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1, 1])

# 1. Vectorized Affect Stream (感情の質の変化)
ax1 = fig.add_subplot(gs[0])
ax1.set_title("Vectorized Affect Structure: The Quality of Meaning", fontsize=14)
# 5次元の感情を積み上げグラフ等で表現したいが、ここでは主要な3次元をラインで
aff_data = np.array(agent_A.history['Affect_Vec'])  # (Steps, 5)
colors = ['#FF5733', '#33FF57', '#3357FF', '#F333FF', '#FF33A8']
labels = ['Sorrow', 'Hope', 'Isolation', 'Solidarity', 'Creation']

for i in range(5):
    ax1.plot(aff_data[:, i], label=labels[i], color=colors[i], alpha=0.8, linewidth=1.5)

ax1.axvspan(50, 60, color='gray', alpha=0.2, label='Trauma')
ax1.axvline(100, color='blue', linestyle='--', label='Encounter')
ax1.set_ylabel("Affect Intensity")
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax1.grid(True, alpha=0.3)

# 2. The Anatomy of Resonance (Action vs Affect Sync)
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.set_title("The Anatomy of Resonance: Action vs. Affect", fontsize=14)
res_data = np.array(resonance_components)  # (Steps-100, 2)
# 時間軸合わせ
padding = np.zeros((100, 2))
res_data = np.vstack([padding, res_data])

ax2.plot(res_data[:, 0], color='purple', linestyle=':', label='Action Sync (Behavior)')
ax2.plot(res_data[:, 1], color='orange', linewidth=2, label='Affect Sync (Meaning)')
ax2.fill_between(range(STEPS), np.array(shared_engram_history), color='green', alpha=0.3, label='Total Shared Engram')

ax2.set_ylabel("Correlation")
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.text(110, 0.8, "Misalignment:\nDoing same thing,\nfeeling differently", color='red', fontsize=9)

# 3. Identity Formation (The Final Goal)
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax3.set_title("Narrative Identity Formation (I accumulated from Shared Affect)", fontsize=14)
id_norm = np.array(agent_A.history['Identity_Norm'])
ax3.plot(id_norm, color='black', linewidth=3, label='Identity Strength')
ax3.fill_between(range(STEPS), id_norm, color='black', alpha=0.1)

ax3.set_ylabel("Identity Magnitude")
ax3.set_xlabel("Time Step")
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.text(200, id_norm[-1] * 0.5, "I become who we were", color='black', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()