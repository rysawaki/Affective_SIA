import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
STEPS = 150
CHANGE_POINT = 50
DIM = 3
IMPACT_VEC = np.array([+5.0, -3.0, +4.0])  # Impact from the world

# World "Stiffness" (Inertia)
# Closer to 1.0 means the world is harder for an individual to change
WORLD_INERTIA = 0.9


def softmax(x):
    x = np.array(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-8)


# --- Class Definition: Creative SIA Agent ---
class CreativeSIAAgent:
    def __init__(self, name="Creator"):
        self.name = name
        self.self_state = np.zeros(DIM)

        # Meaning Interpretation Bias
        self.interpret_bias = np.array([-0.5, -0.2, -0.7])

        # Trace (Internal Imprint)
        self.trace = np.zeros(DIM)

        # History recording
        self.self_history = []
        self.trace_history = []
        self.action_history = []  # History of actions (interventions on the world)

    def interpret(self, external_vec):
        return external_vec + self.interpret_bias

    def update_and_act(self, current_world_vec):
        # --- 1. Perception & Imprint ---

        # Interpret the world
        meaning = self.interpret(current_world_vec)

        # Discrepancy (Mismatch)
        discrepancy = meaning - self.self_state
        shock = np.linalg.norm(discrepancy)

        # Determine Plasticity (Trace-driven Attention Mechanism)
        attention_logits = np.abs(discrepancy) + 0.8 * np.abs(self.trace)  # Amplify effect of existing traces
        attention = softmax(attention_logits)

        base_plasticity = 0.02
        shock_gain = 0.5 * (shock / (1.0 + shock))
        plasticity_vec = base_plasticity + shock_gain * attention

        # Update Self (Adaptation)
        self.self_state += plasticity_vec * discrepancy

        # Update Trace (Irreversible Imprinting)
        imprint_strength = np.tanh(shock)
        self.trace += 0.1 * discrepancy * imprint_strength
        self.trace *= 0.998  # Decays very slowly (Persistent)

        # --- 2. Active Creation (Action) ---

        # Principle of Action:
        # Use internal "Trace" and "Current Self" to try and pull the world closer to oneself.

        # Creative Drive is proportional to the total magnitude of Trace (Pain/Meaning)
        creation_energy = np.linalg.norm(self.trace) * 0.3

        # Action Vector:
        # Direction = (Internal Self - World)
        # i.e., "World, become more like me (Understand my pain)"
        action_direction = self.self_state - current_world_vec

        # Calculate Force
        action_vec = action_direction * creation_energy * 0.1

        # Record history
        self.self_history.append(self.self_state.copy())
        self.trace_history.append(self.trace.copy())
        self.action_history.append(action_vec.copy())

        return action_vec


# --- Run Simulation ---

# Generate Initial World (Base Fate)
base_world_scenario = np.zeros((STEPS, DIM))
base_world_scenario += np.random.normal(0, 0.05, size=(STEPS, DIM))
base_world_scenario[CHANGE_POINT:] += IMPACT_VEC  # Fated Event

# The actual world the agent lives in (Dynamically changes)
actual_world_history = []
current_world = base_world_scenario[0].copy()

agent = CreativeSIAAgent("Artist")

for t in range(STEPS):
    # 1. Base environmental change (Force of Fate)
    external_force = base_world_scenario[t] - base_world_scenario[t - 1] if t > 0 else np.zeros(DIM)
    current_world += external_force

    # 2. Agent perceives, adapts, and acts
    action_from_agent = agent.update_and_act(current_world)

    # 3. World is modified by agent's action (Creative Intervention)
    # World Inertia vs Agent's Will
    current_world = WORLD_INERTIA * current_world + (1 - WORLD_INERTIA) * (current_world + action_from_agent)

    actual_world_history.append(current_world.copy())

# Data formatting
actual_world_hist = np.array(actual_world_history)
sia_self_hist = np.array(agent.self_history)
sia_trace_hist = np.array(agent.trace_history)
sia_action_hist = np.array(agent.action_history)

# --- Visualization ---
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# 1. Interaction between World and Self
# Dotted line: Fate (The world as it would have been without intervention)
# Solid line: Modified Reality (The world created by the agent)
# Blue line: Agent's Creative Self
axes[0].plot(np.linalg.norm(base_world_scenario, axis=1), 'k:', label='Fate (Unchanged World)', alpha=0.5)
axes[0].plot(np.linalg.norm(actual_world_hist, axis=1), 'k-', label='Modified Reality (Created World)', linewidth=2)
axes[0].plot(np.linalg.norm(sia_self_hist, axis=1), 'b-', label='Creative Self', linewidth=2)
axes[0].axvline(CHANGE_POINT, color='red', linestyle='--', alpha=0.5, label='Imprint Event')
axes[0].set_title('Creative Intervention: How Self Rewrites "Fate"')
axes[0].set_ylabel('Norm / State')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 2. Trace and Creative Drive (Action Magnitude)
trace_norm = np.linalg.norm(sia_trace_hist, axis=1)
action_norm = np.linalg.norm(sia_action_hist, axis=1)

axes[1].fill_between(range(STEPS), trace_norm, color='orange', alpha=0.3, label='Trace (Internal Pain/Meaning)')
axes[1].plot(action_norm, color='purple', linewidth=2, label='Creative Action Force (Expression)')
axes[1].set_title('Trace (Orange) Drives Creative Action (Purple)')
axes[1].set_ylabel('Magnitude')
axes[1].legend()
axes[1].grid(alpha=0.3)

# 3. Detail: Action Components (Vector of Will)
axes[2].plot(sia_action_hist[:, 0], label='Action dim 0')
axes[2].plot(sia_action_hist[:, 1], label='Action dim 1')
axes[2].plot(sia_action_hist[:, 2], label='Action dim 2')
axes[2].set_title('Vector of Will (Action Components)')
axes[2].set_xlabel('Time Step')
axes[2].set_ylabel('Action Vector')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()

# Save figure
os.makedirs('../results/figures', exist_ok=True)
plt.savefig('../results/figures/creative_action.png', dpi=300)

plt.show()