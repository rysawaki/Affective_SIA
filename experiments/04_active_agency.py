import numpy as np
import matplotlib.pyplot as plt
import os

# --- Research Parameter Settings (Affective SIA) ---
STEPS = 250  # Extended duration to observe narrative gestation
DIM = 3      # Dimension of Meaning Space
DT = 0.1     # Time step

# Theoretical Parameters
ALPHA_TRACE = 1.0     # alpha: Trace sensitivity/dependence
BETA_ACTION = 1.5     # beta: Action dependence (Agency)
GAMMA_CREATION = 0.6  # gamma: Strength of creative action

# Affect Parameter (New)
PHI_NARRATIVE = 0.9   # phi: Narrative Inertia (temporal persistence). Higher = consistent affect.
# Low = fleeting emotion, High = sustained affect/mood

# Fix random seed
np.random.seed(42)


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


class Affective_SIA_Agent:
    def __init__(self, name):
        self.name = name
        self.S = np.zeros(DIM)  # Self State
        self.T = np.zeros(DIM)  # Trace (Engram)
        self.A = 0.0  # Affective Meaning (Affect) *New
        self.last_action = np.zeros(DIM)

        # History
        self.history = {
            'P_self': [], 'Trace_Norm': [], 'Action_Norm': [],
            'Affect': [], 'Discrepancy': []
        }

    def compute_attribution(self, E_meaning):
        """
        Active SIA Core:
        P(Self|E) is determined by "how much one was involved" and "depth of trace".
        """
        diff = E_meaning - self.S
        dist = np.linalg.norm(diff)
        trace_mag = np.linalg.norm(self.T)
        action_mag = np.linalg.norm(self.last_action)

        # Agency Boost: Easier to attribute if one acted recently
        logit = -dist + (ALPHA_TRACE * trace_mag) + (BETA_ACTION * action_mag)

        return sigmoid(logit), dist, diff

    def step(self, world_vec, interact_force=None):
        # --- 1. Interpretation and Attribution ---
        E_meaning = world_vec.copy()
        if interact_force is not None:
            E_meaning += interact_force

        P_self, dist, diff = self.compute_attribution(E_meaning)

        # --- 2. Learning and Trace Formation ---
        attention = softmax(np.abs(diff) + np.abs(self.T))

        # Self update
        learning_rate = 0.5 * P_self * attention
        self.S += learning_rate * diff * DT

        # Trace formation (Imprinting)
        shock = np.tanh(dist)
        d_T = (shock * diff * P_self * DT) - (0.01 * self.T * DT)
        self.T += d_T

        # --- 3. Genesis of Affective Meaning (New) ---
        # Affect is energy supplied by "Trace magnitude" x "Self-Attribution"
        # However, it is not instantaneous but integrated over time (Narratization)

        raw_affect_input = np.linalg.norm(self.T) * P_self

        # Narrative Inertia
        # A(t) = phi * A(t-1) + (1-phi) * Input
        self.A = (PHI_NARRATIVE * self.A) + ((1 - PHI_NARRATIVE) * raw_affect_input)

        # --- 4. Action Generation ---
        # Modification: Action flows not directly from Trace, but from Affect
        # "I express because I feel meaning."

        creation_force = self.A * GAMMA_CREATION

        action_vec = (self.S - world_vec) * creation_force

        # Update state
        self.last_action = action_vec

        # Record history
        self.history['P_self'].append(P_self)
        self.history['Trace_Norm'].append(np.linalg.norm(self.T))
        self.history['Affect'].append(self.A)  # Affect history
        self.history['Action_Norm'].append(np.linalg.norm(action_vec))
        self.history['Discrepancy'].append(dist)

        return action_vec


# --- Run Simulation ---

agent_A = Affective_SIA_Agent("Protagonist")
agent_B = Affective_SIA_Agent("Partner")

world_state = np.zeros(DIM)
world_history = []
shared_engram_history = []

# Scenario
# 0-40: Calm
# 40-50: Trauma (Impact)
# 50-140: Solitary Meaning Generation (Affective Genesis)
# 140-: Deep Resonance with Other (Affective Resonance)

print("Running Affective SIA Simulation...")

for t in range(STEPS):
    # --- World Dynamics ---
    if 40 <= t < 50:
        target = np.array([5.0, -4.0, 4.0])
        world_state = 0.8 * world_state + 0.2 * target
    elif t == 50:
        pass
    else:
        world_state += np.random.normal(0, 0.05, DIM)
        world_state *= 0.95

    # --- Interaction Phase (t >= 140) ---
    interact_force_A = None
    current_shared_score = 0

    if t >= 140:
        # Agent B also reacts to world
        action_B = agent_B.step(world_state, interact_force=agent_A.last_action * 0.3)
        interact_force_A = action_B * 0.3

        # --- Shared Engram Calculation (Deep Resonance) ---
        # 1. Attribution Sync (Cognitive Sync)
        p1 = agent_A.history['P_self'][-1]
        p2 = agent_B.history['P_self'][-1]

        # 2. Action Sync (Behavioral Sync)
        sync_action = max(0, cosine_similarity(agent_A.last_action, agent_B.last_action))

        # 3. Affective Sync (Affective Sync) *New
        # "Do both hold deep affect (meaning)?"
        # Shallow emotion does not resonate with deep affect. Limited by the lower level.
        a1 = agent_A.A
        a2 = agent_B.A
        sync_affect = min(a1, a2)

        # Integrated Score: Cognitive x Action x Affect
        current_shared_score = p1 * p2 * sync_action * sync_affect

    shared_engram_history.append(current_shared_score)

    # --- Agent A Step ---
    action_A = agent_A.step(world_state, interact_force_A)

    # --- World Update ---
    world_state += action_A * 0.15
    if t >= 140:
        world_state += agent_B.last_action * 0.15

    world_history.append(world_state.copy())

# --- Visualization ---
fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True, constrained_layout=True)

# 1. Affective Genesis (Meaning Making)
ax1 = axes[0]
ax1.set_title(r"Genesis of Affect: From Trace to Meaning ($Trace \rightarrow Affect \rightarrow Action$)", fontsize=14)
# Trace (Material)
ax1.fill_between(range(STEPS), agent_A.history['Trace_Norm'], color='gray', alpha=0.2, label='Raw Trace (Pain)')
# Affect (Meaning)
ax1.plot(agent_A.history['Affect'], color='orange', linewidth=3, label='Affective Meaning (Narrative)')
# Action (Expression)
ax1.plot(agent_A.history['Action_Norm'], color='purple', linestyle='--', linewidth=2, label='Action (Expression)')

ax1.axvspan(40, 50, color='red', alpha=0.1, label='Trauma')
ax1.axvline(140, color='blue', linestyle='--')
ax1.legend(loc='upper left')
ax1.set_ylabel("Magnitude")
ax1.grid(True, alpha=0.3)
ax1.text(80, 1.5, "Transformation:\nTrace becomes Affect", color='orange', fontsize=10, fontweight='bold')

# 2. The Self-Attribution Loop
ax2 = axes[1]
ax2.set_title(r"Self-Attribution Loop ($P(Self)$ restored by Affect-driven Action)", fontsize=14)
ax2.plot(agent_A.history['Discrepancy'], 'k:', alpha=0.5, label='Discrepancy')
ax2.plot(agent_A.history['P_self'], 'r-', linewidth=2, label=r'Self-Attribution $P(Self|E)$')
ax2.set_ylabel("Probability")
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Deep Shared Engram
ax3 = axes[2]
ax3.set_title(r"Deep Resonance: Sync of Action & Affect ($P \cdot \cos(A) \cdot \min(Aff)$)", fontsize=14)
ax3.plot(shared_engram_history, color='green', linewidth=2, label='Shared Meaning')
ax3.axvline(140, color='blue', linestyle=':', label='Partner Enters')
ax3.set_ylabel("Resonance Depth")
ax3.set_xlabel("Time Step")
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.text(160, 0.1, "True Resonance requires\nShared Affective Depth", color='green')

# Save figure
os.makedirs('results/figures', exist_ok=True)
plt.savefig('results/figures/active_agency.png', dpi=300)

plt.show()