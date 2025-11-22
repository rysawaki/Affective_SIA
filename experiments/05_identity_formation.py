import numpy as np
import matplotlib.pyplot as plt
import os
from affective_sia.agents import Identity_SIA_Agent
from affective_sia.core import cosine_similarity
from affective_sia.config import SimulationConfig
from affective_sia.visualization import plot_identity_simulation

# --- 1. Load Configuration ---
config = SimulationConfig()
np.random.seed(config.random_seed)


# --- 2. Initialize Agents ---
def create_configured_agent(name, seed):
    agent = Identity_SIA_Agent(
        name,
        dim_meaning=config.dim_meaning,
        dim_affect=config.dim_affect,
        affect_matrix_seed=seed
    )
    # Inject Parameters
    agent.alpha_trace = config.alpha_trace
    agent.beta_action = config.beta_action
    agent.gamma_creation = config.gamma_creation
    agent.eta_identity = config.eta_identity
    return agent


agent_A = create_configured_agent("Protagonist", seed=42)
agent_B = create_configured_agent("Partner", seed=99)

# --- 3. Run Simulation ---
world_state = np.zeros(config.dim_meaning)
shared_engram_history = []
resonance_components = []

print("Running SIA Identity Simulation (Clean Architecture)...")

for t in range(config.steps):
    # Scenario: Trauma and Recovery
    if 50 <= t < 60:
        target = np.array([5.0, 5.0, 5.0])
        world_state = 0.8 * world_state + 0.2 * target
    else:
        world_state *= 0.95

    # Interaction Phase (t >= 100)
    interact_force_A = None
    current_shared = 0.0
    cos_act, cos_aff = 0.0, 0.0

    if t >= 100:
        # Partner's Action
        action_B = agent_B.step(world_state, dt=config.dt, interact_force=agent_A.last_action * 0.3)
        interact_force_A = action_B * 0.3

        # Calculate Resonance
        p1 = agent_A.history['P_self'][-1]
        p2 = agent_B.history['P_self'][-1]

        cos_act = cosine_similarity(agent_A.last_action, agent_B.last_action)
        cos_aff = cosine_similarity(agent_A.A, agent_B.A)
        min_depth = min(np.linalg.norm(agent_A.A), np.linalg.norm(agent_B.A))

        if cos_act > 0 and cos_aff > 0:
            current_shared = p1 * p2 * cos_act * cos_aff * min_depth

    shared_engram_history.append(current_shared)
    resonance_components.append([cos_act, cos_aff])

    # Update Protagonist
    action_A = agent_A.step(
        world_state,
        dt=config.dt,
        interact_force=interact_force_A,
        shared_resonance=current_shared
    )

    world_state += action_A * 0.1

# --- 4. Visualization & Saving ---
print("Simulation completed. Rendering plot...")

# Visualization is delegated to the plot_identity_simulation function.
plot_identity_simulation(agent_A, resonance_components, shared_engram_history, config)

# Save Figure explicitly for README
# Note: Ensure plt.show() is removed from visualization.py or called AFTER this.
os.makedirs('../results/figures', exist_ok=True)
plt.savefig('../results/figures/identity_formation.png', dpi=300)
print("Figure saved to results/figures/identity_formation.png")

plt.show()