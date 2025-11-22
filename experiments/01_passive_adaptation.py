import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
STEPS = 120
CHANGE_POINT = 50  # "Life-changing event" occurs here
DIM = 3            # Represent self/world with 3D meaning vectors

# Image of the world's 3D meaning axes:
# 0: Stability vs Instability
# 1: Relatedness (Connection vs Isolation)
# 2: Self-Worth (Affirmation vs Denial)

IMPACT_VEC = np.array([+5.0, -3.0, +4.0])  # "Meaning vector" of the world after the upheaval

# --- Environment (World) Generation ---
world_state = np.zeros((STEPS, DIM))
world_state += np.random.normal(0, 0.1, size=(STEPS, DIM))  # Tiny fluctuations

# World's meaning structure changes after CHANGE_POINT
world_state[CHANGE_POINT:] += IMPACT_VEC


def softmax(x):
    x = np.array(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-8)


# --- Static Self (Traditional Model) ---
class StaticAgent:
    """
    Agent that accepts the external world as is and tries to track it
    with a low constant learning rate.
    No trace, no interpretation layer. Just 'adjustment'.
    """
    def __init__(self, name="Static"):
        self.name = name
        self.self_state = np.zeros(DIM)
        self.history = []
        self.plasticity_history = []

    def update(self, external_vec):
        # Internalize world as "fact" directly (no interpretation)
        meaning = external_vec.copy()

        discrepancy = meaning - self.self_state
        shock = np.linalg.norm(discrepancy)

        # Static Self: Always same low learning rate
        plasticity = 0.05

        self.self_state += plasticity * discrepancy

        self.history.append(self.self_state.copy())
        self.plasticity_history.append(plasticity)


# --- Generative Self (SIA-style) ---
class GenerativeSIAAgent:
    """
    - Interprets the world into 'meaning' before internalization.
    - Plasticity changes per dimension based on:
        - Magnitude of discrepancy (Shock)
        - How much trace is already imprinted on that axis
    - Trace accumulates as history-dependent, irreversible change.
    """
    def __init__(self, name="Generative-SIA"):
        self.name = name
        self.self_state = np.zeros(DIM)

        # Bias: "How one tends to interpret the world"
        # e.g., Always interpreting slightly towards self-denial
        self.interpret_bias = np.array([-0.5, -0.2, -0.7])

        # Trace vector: How strong the imprint is on each axis
        self.trace = np.zeros(DIM)

        # Recording
        self.self_history = []
        self.trace_history = []
        self.plasticity_history = []   # Plasticity for each dimension
        self.shock_history = []

    def interpret(self, external_vec):
        """
        Layer that interprets raw changes in the world through personal bias.
        Here, 'World' becomes 'Meaning', not just 'Fact'.
        """
        # Simply adding bias here, but non-linearity could be added
        meaning = external_vec + self.interpret_bias
        return meaning

    def update(self, external_vec):
        # 1. Interpret world as "Meaning"
        meaning = self.interpret(external_vec)

        # 2. Discrepancy with self (Meaning mismatch)
        discrepancy = meaning - self.self_state
        shock = np.linalg.norm(discrepancy)  # Total shock intensity (scalar)

        # 3. Select which dimensions to change and how much (Attention)
        #
        #   attention ~ |discrepancy| + alpha * |trace|
        #   -> Attention goes to axes already strongly imprinted + current large discrepancy
        #
        attention_logits = np.abs(discrepancy) + 0.5 * np.abs(self.trace)
        attention = softmax(attention_logits)  # Attention per dimension (sums to 1)

        # 4. Determine plasticity vector
        #
        #   base_plasticity: Room for change that always exists
        #   shock_gain     : Easier to change as shock increases
        #
        base_plasticity = 0.03
        shock_gain = 0.6 * (shock / (1.0 + shock))  # Falls within approx 0 to 0.6

        plasticity_vec = base_plasticity + shock_gain * attention
        plasticity_vec = np.clip(plasticity_vec, 0.0, 0.8)

        # 5. Self update (different learning rates per dimension)
        self.self_state += plasticity_vec * discrepancy

        # 6. Trace update (History-dependent, mostly irreversible)
        #
        #   - Larger discrepancy/shock imprints more trace
        #   - Remains with sign (direction of experience)
        #
        imprint_strength = np.tanh(shock)   # Approaches 1 as shock increases
        self.trace += 0.1 * discrepancy * imprint_strength

        # Include some natural "weathering", but never completely disappears
        self.trace *= 0.995

        # Recording
        self.self_history.append(self.self_state.copy())
        self.trace_history.append(self.trace.copy())
        self.plasticity_history.append(plasticity_vec.copy())
        self.shock_history.append(shock)


# --- Run Simulation ---
agent_static = StaticAgent("Static")
agent_sia = GenerativeSIAAgent("Generative-SIA")

for t in range(STEPS):
    reality_vec = world_state[t]
    agent_static.update(reality_vec)
    agent_sia.update(reality_vec)

# Convert to numpy arrays (for visualization)
static_self_hist = np.array(agent_static.history)          # (T, DIM)
sia_self_hist    = np.array(agent_sia.self_history)        # (T, DIM)
sia_trace_hist   = np.array(agent_sia.trace_history)       # (T, DIM)
sia_plast_hist   = np.array(agent_sia.plasticity_history)  # (T, DIM)
sia_shock_hist   = np.array(agent_sia.shock_history)       # (T,)


# --- Visualization ---
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# 1. Time evolution of self-state norm
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
axes[0].set_title('Evolution of Self-State Magnitude')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 2. Self-state of Generative Self per dimension
axes[1].plot(sia_self_hist[:, 0], label='Self dim 0 (Stability)')
axes[1].plot(sia_self_hist[:, 1], label='Self dim 1 (Relatedness)')
axes[1].plot(sia_self_hist[:, 2], label='Self dim 2 (Self-Worth)')
axes[1].axvline(CHANGE_POINT, color='gray', linestyle=':', alpha=0.7)
axes[1].set_ylabel('Self components')
axes[1].set_title('Internal Structure of Generative Self')
axes[1].legend()
axes[1].grid(alpha=0.3)

# 3. Trace and Shock
axes[2].plot(
    np.linalg.norm(sia_trace_hist, axis=1),
    label='Trace norm (Imprint Strength)'
)
axes[2].plot(
    sia_shock_hist,
    label='Shock (Discrepancy Magnitude)', alpha=0.7
)
axes[2].axvline(CHANGE_POINT, color='gray', linestyle=':', alpha=0.7)
axes[2].set_ylabel('Value')
axes[2].set_xlabel('Time step')
axes[2].set_title('Accumulation of Trace and Shock')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()

# Save Figure
os.makedirs('results/figures', exist_ok=True)
plt.savefig('results/figures/passive_adaptation.png', dpi=300)
print("Figure saved to results/figures/passive_adaptation.png")

plt.show()