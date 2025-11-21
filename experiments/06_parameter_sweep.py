import numpy as np
import matplotlib.pyplot as plt
from affective_sia.agents import Identity_SIA_Agent
from affective_sia.core import cosine_similarity
from affective_sia.config import SimulationConfig

# --- 実験設定: 感度分析 ---
alphas_to_test = [0.1, 0.5, 1.0, 1.5, 2.0]  # 試したいtrace感度のリスト
final_identities = []

print("Starting Parameter Sweep for Alpha (Trace Sensitivity)...")

# 設定のベースを作成
base_config = SimulationConfig(steps=300)

for alpha in alphas_to_test:
    print(f"Testing alpha = {alpha}...")

    # 1. Configを上書きしてエージェント生成
    # (ここがリファクタリングの成果です！設定を注入するだけで挙動を変えられます)
    agent_A = Identity_SIA_Agent("Protagonist", dim_meaning=3, dim_affect=5, affect_matrix_seed=42)
    agent_A.alpha_trace = alpha  # パラメータ注入

    agent_B = Identity_SIA_Agent("Partner", dim_meaning=3, dim_affect=5, affect_matrix_seed=99)
    # Partnerは標準値のままにする、あるいは合わせる等の設定が可能

    # 2. シミュレーション実行 (簡易版ループ)
    world_state = np.zeros(3)
    current_identity = 0.0

    for t in range(base_config.steps):
        # トラウマ
        if 50 <= t < 60:
            world_state = 0.8 * world_state + 0.2 * np.array([5., 5., 5.])
        else:
            world_state *= 0.95

        # 相互作用 (t>=100)
        interact = None
        shared = 0.0
        if t >= 100:
            act_b = agent_B.step(world_state, dt=0.1, interact_force=agent_A.last_action * 0.3)
            interact = act_b * 0.3

            # 共鳴判定
            p1, p2 = agent_A.history['P_self'][-1], agent_B.history['P_self'][-1]
            c_act = cosine_similarity(agent_A.last_action, agent_B.last_action)
            c_aff = cosine_similarity(agent_A.A, agent_B.A)
            if c_act > 0 and c_aff > 0:
                shared = p1 * p2 * c_act * c_aff * min(np.linalg.norm(agent_A.A), np.linalg.norm(agent_B.A))

        # 更新
        agent_A.step(world_state, dt=0.1, interact_force=interact, shared_resonance=shared)

    # 3. 結果の記録 (最終的なアイデンティティの強さ)
    final_identities.append(agent_A.history['Identity_Norm'][-1])

# --- 結果の可視化 ---
plt.figure(figsize=(8, 6))
plt.plot(alphas_to_test, final_identities, 'o-', linewidth=2, color='blue')
plt.title("Sensitivity Analysis: Trace Sensitivity (α) vs. Identity Strength")
plt.xlabel("Alpha (Trace Sensitivity)")
plt.ylabel("Final Identity Magnitude")
plt.grid(True, alpha=0.3)
plt.show()