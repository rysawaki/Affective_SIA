import numpy as np
import matplotlib.pyplot as plt
import os
from affective_sia.agents import Identity_SIA_Agent
from affective_sia.core import cosine_similarity
from affective_sia.config import SimulationConfig

# --- Experimental Setup: Sensitivity Analysis ---
alphas_to_test = [0.1, 0.5, 1.0, 1.5, 2.0]  # List of trace sensitivities to test
final_identities = []

print("Starting Parameter Sweep for Alpha (Trace Sensitivity)...")

# Create base configuration
base_config = SimulationConfig(steps=300)

for alpha in alphas_to_test:
    print(f"Testing alpha = {alpha}...")

    # 1. Initialize Agents (Inject Parameters)
    # (Refactoring benefit: behavior changes just by injecting settings)
    agent_A = Identity_SIA_Agent("Protagonist", dim_meaning=3, dim_affect=5, affect_matrix_seed=42)
    agent_A.alpha_trace = alpha  # Inject parameter

    agent_B = Identity_SIA_Agent("Partner", dim_meaning=3, dim_affect=5, affect_matrix_seed=99)
    # Partner uses standard configuration or can be matched

    # 2. Run Simulation (Simplified Loop)
    world_state = np.zeros(3)
    current_identity = 0.0

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

    # 3. Record Results (Final Identity Magnitude)
    final_identities.append(agent_A.history['Identity_Norm'][-1])

# --- Visualization ---
plt.figure(figsize=(8, 6))
plt.plot(alphas_to_test, final_identities, 'o-', linewidth=2, color='blue')
plt.title("Sensitivity Analysis: Trace Sensitivity (alpha) vs. Identity Strength")
plt.xlabel("Alpha (Trace Sensitivity)")
plt.ylabel("Final Identity Magnitude")
plt.grid(True, alpha=0.3)

# Save Figure
os.makedirs('../results/figures', exist_ok=True)
plt.savefig('../results/figures/sensitivity_analysis.png', dpi=300)
print("Figure saved to results/figures/sensitivity_analysis.png")

plt.show()