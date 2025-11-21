import numpy as np
import matplotlib.pyplot as plt
from affective_sia.agents import Identity_SIA_Agent
from affective_sia.core import cosine_similarity

# --- 実験設定 ---
STEPS = 300
DIM_MEANING = 3
DIM_AFFECT = 5
np.random.seed(45)

# --- エージェントの初期化 (ライブラリから呼び出し) ---
agent_A = Identity_SIA_Agent("Protagonist", DIM_MEANING, DIM_AFFECT, affect_matrix_seed=42)
agent_B = Identity_SIA_Agent("Partner", DIM_MEANING, DIM_AFFECT, affect_matrix_seed=99)

world_state = np.zeros(DIM_MEANING)
shared_engram_history = []
resonance_components = []

print("Running SIA Identity Simulation (Refactored)...")

for t in range(STEPS):
    # 1. 世界の変動シナリオ
    if 50 <= t < 60:
        target = np.array([5.0, 5.0, 5.0])
        world_state = 0.8 * world_state + 0.2 * target
    else:
        world_state *= 0.95

    # 2. 相互作用と共鳴の計算
    interact_force_A = None
    current_shared = 0.0
    cos_act, cos_aff = 0.0, 0.0

    if t >= 100:
        # Bの行動とAへの干渉
        action_B = agent_B.step(world_state, interact_force=agent_A.last_action * 0.3, shared_resonance=0)
        interact_force_A = action_B * 0.3

        # 共鳴計算 (Shared Engram)
        p1 = agent_A.history['P_self'][-1]
        p2 = agent_B.history['P_self'][-1]

        # coreライブラリの関数を活用
        cos_act = cosine_similarity(agent_A.last_action, agent_B.last_action)
        cos_aff = cosine_similarity(agent_A.A, agent_B.A)
        min_depth = min(np.linalg.norm(agent_A.A), np.linalg.norm(agent_B.A))

        if cos_act > 0 and cos_aff > 0:
            current_shared = p1 * p2 * cos_act * cos_aff * min_depth

    shared_engram_history.append(current_shared)
    resonance_components.append([cos_act, cos_aff])

    # 3. 主人公の更新 (共鳴値をIdentity形成に利用)
    action_A = agent_A.step(world_state, interact_force=interact_force_A, shared_resonance=current_shared)

    # 世界へのフィードバック
    world_state += action_A * 0.1

# --- 以下、可視化コード（元のファイルからそのまま利用可能） ---
# ... (plt.figure以降は同じなので省略可能ですが、必要なら提示します)

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