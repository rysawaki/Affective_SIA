import numpy as np
import matplotlib.pyplot as plt
import os
from affective_sia.agents import Identity_SIA_Agent
from affective_sia.core import cosine_similarity
from affective_sia.config import SimulationConfig

# --- Experimental Setup: Sensitivity Analysis ---
alphas_to_test = [0.1, 0.5, 1.0, 1.5, 2.0]  # Trace sensitivities to test
NUM_TRIALS = 10  # ★重要: 各条件で10回試行して統計をとる
final_identities_mean = []
final_identities_std = []

print(f"Starting Parameter Sweep (Trials per alpha: {NUM_TRIALS})...")

base_config = SimulationConfig(steps=300)

for alpha in alphas_to_test:
    print(f"Testing alpha = {alpha}...")

    results = []

    # 統計的信頼性を得るために複数回実行 (Monte Carlo)
    for seed in range(NUM_TRIALS):
        # 1. Initialize Agents with different seeds
        # Protagonist: seed varies (42 + seed)
        agent_A = Identity_SIA_Agent("Protagonist", dim_meaning=3, dim_affect=5, affect_matrix_seed=42 + seed)
        agent_A.alpha_trace = alpha

        # Partner: seed varies (99 + seed)
        agent_B = Identity_SIA_Agent("Partner", dim_meaning=3, dim_affect=5, affect_matrix_seed=99 + seed)

        # 2. Run Simulation
        world_state = np.zeros(3)

        # ランダムな揺らぎも試行ごとに変える
        np.random.seed(seed)

        for t in range(base_config.steps):
            # Trauma Event
            if 50 <= t < 60:
                world_state = 0.8 * world_state + 0.2 * np.array([5., 5., 5.])
            else:
                world_state *= 0.95

            # Interaction Phase (t >= 100)
            interact = None
            shared = 0.0
            if t >= 100:
                act_b = agent_B.step(world_state, dt=0.1, interact_force=agent_A.last_action * 0.3)
                interact = act_b * 0.3

                # Resonance Check
                p1, p2 = agent_A.history['P_self'][-1], agent_B.history['P_self'][-1]
                c_act = cosine_similarity(agent_A.last_action, agent_B.last_action)
                c_aff = cosine_similarity(agent_A.A, agent_B.A)
                if c_act > 0 and c_aff > 0:
                    shared = p1 * p2 * c_act * c_aff * min(np.linalg.norm(agent_A.A), np.linalg.norm(agent_B.A))

            # Update Agent
            agent_A.step(world_state, dt=0.1, interact_force=interact, shared_resonance=shared)

        # Record Result
        results.append(agent_A.history['Identity_Norm'][-1])

    # 平均と標準偏差を計算
    final_identities_mean.append(np.mean(results))
    final_identities_std.append(np.std(results))

# --- Visualization ---
plt.figure(figsize=(8, 6))

# エラーバー付きプロット (これが科学的証拠になる)
plt.errorbar(alphas_to_test, final_identities_mean, yerr=final_identities_std,
             fmt='o-', linewidth=2, color='blue', capsize=5, label='Mean Identity Strength')

plt.title(
    f"Sensitivity Analysis: Trace Sensitivity (alpha) vs. Identity Strength\n(N={NUM_TRIALS} trials, Error bars=SD)")
plt.xlabel("Alpha (Trace Sensitivity)")
plt.ylabel("Final Identity Magnitude")
plt.grid(True, alpha=0.3)
plt.legend()

# Save Figure
os.makedirs('../results/figures', exist_ok=True)
plt.savefig('../results/figures/sensitivity_analysis.png', dpi=300)
plt.savefig('../results/figures_pdf/sensitivity_analysis.pdf')
print("Figure saved with error bars.")

plt.show()