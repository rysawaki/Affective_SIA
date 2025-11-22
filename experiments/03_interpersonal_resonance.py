import numpy as np
import matplotlib.pyplot as plt
import os

# --- Research Parameter Settings ---
STEPS = 200
DIM = 3  # Dimension of Meaning Space (e.g., [Stability, Intimacy, Self-Worth])
DT = 0.1  # Time step
ALPHA_TRACE = 1.2  # Alpha: Coefficient where Trace reinforces Self-Attribution (Re-enactment factor)
BETA_PLASTICITY = 0.5  # Base coefficient for Plasticity
GAMMA_ACTION = 0.4  # Coefficient for strength of Creative Action

# Fix random seed for reproducibility
np.random.seed(42)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum() + 1e-8)


class SIA_Agent:
    def __init__(self, name, sensitivity=1.0):
        self.name = name
        self.S = np.zeros(DIM)  # Self State
        self.T = np.zeros(DIM)  # Trace
        self.sensitivity = sensitivity  # Individual difference (Sensitivity)

        # For recording history
        self.history = {
            'S': [], 'T': [], 'Action': [],
            'P_self': [], 'Loss': [], 'Discrepancy': []
        }

    def compute_attribution(self, E_meaning):
        """
        Core Equation of SIA Theory: P(Self|E)
        Calculates the probability that an experience E belongs to the Self.

        Args:
            E_meaning (np.array): Interpreted experience vector
        Returns:
            P_self (float): Self-Attribution Probability (0.0 - 1.0)
            dist (float): Pure semantic discrepancy
        """
        # 1. Semantic Discrepancy
        diff = E_meaning - self.S
        dist = np.linalg.norm(diff)

        # 2. Trace Gravity
        # The deeper the trace, the easier it is to feel "This is my story" (even if painful)
        trace_magnitude = np.linalg.norm(self.T)

        # 3. Logit Calculation (SIA Core Equation)
        # Large discrepancy reduces attribution (-dist), Trace increases it (+alpha*trace)
        logit = -dist + (ALPHA_TRACE * trace_magnitude)

        P_self = sigmoid(logit)
        return P_self, dist, diff

    def step(self, world_vec, interact_force=None):
        """
        One-step update: Perception -> Attribution -> Trace Formation -> Action
        """
        # --- 1. Interpretation ---
        # Here we simply use the world vector as meaning,
        # but ideally an Interpretation Layer enters here (Meaning = f(World))
        E_meaning = world_vec.copy()
        if interact_force is not None:
            E_meaning += interact_force  # Influence from others

        # --- 2. Self-Attribution ---
        P_self, dist, diff = self.compute_attribution(E_meaning)

        # --- 3. Loss Calculation (For theory validation) ---
        # L = -log(P(Self|E)) + Regularization
        # The Self should move to minimize this Loss
        loss = -np.log(P_self + 1e-8)

        # --- 4. Update (Plasticity & Trace) ---

        # Attention is directed towards dimensions with stronger traces
        attention = softmax(np.abs(diff) + np.abs(self.T))

        # Self Plasticity
        # Deep learning occurs only when P_self is high (Recognized as "Self")
        learning_rate = BETA_PLASTICITY * P_self * attention
        d_S = learning_rate * diff * DT
        self.S += d_S

        # Trace Formation (Imprinting)
        # Imprinted only when shock is strong AND recognized as "Self"
        shock = np.tanh(dist)
        d_T = (shock * diff * P_self * DT) - (0.01 * self.T * DT)  # Decay term included
        self.T += d_T

        # --- 5. Creative Action ---
        # Force to modify the world to resolve the Trace
        # Movement akin to Action = - grad(Loss)
        # Stronger traces and higher self-attribution lead to stronger expression

        action_magnitude = np.linalg.norm(self.T) * GAMMA_ACTION * P_self
        # Direction is "Self - World" (Pulling the world towards the Self)
        action_vec = (self.S - world_vec) * action_magnitude

        # --- Record History ---
        self.history['S'].append(self.S.copy())
        self.history['T'].append(self.T.copy())
        self.history['Action'].append(action_vec.copy())
        self.history['P_self'].append(P_self)
        self.history['Loss'].append(loss)
        self.history['Discrepancy'].append(dist)

        return action_vec


# --- Run Simulation ---

# Scenario Settings:
# 0-50: Calm
# 50-60: Trauma Impact (Traumatic Event)
# 60-120: Recovery and Trace Consolidation
# 120-200: Encounter with Other (Shared Engram Phase)

agent = SIA_Agent("Protagonist")
world_history = []

# World state vector (Starts near zero)
world_state = np.zeros(DIM)

# Other agent (Appears in the second half)
other_agent = SIA_Agent("Partner")
shared_engram = []  # Strength of shared trace

print("Simulation Started...")

for t in range(STEPS):
    # --- World Fluctuations ---
    if 50 <= t < 60:
        # Trauma Impact: Forcibly give a huge meaning discrepancy
        world_target = np.array([5.0, -5.0, 5.0])
        world_state = 0.8 * world_state + 0.2 * world_target
    elif t == 60:
        # Impact passes, world tries to return but...
        world_target = np.zeros(DIM)
    else:
        # Normal tiny fluctuations
        world_state += np.random.normal(0, 0.05, DIM)
        # World slowly tries to return (Elasticity)
        world_state *= 0.95

    # --- Agent Update ---
    # First half: Solitary. Second half: Interaction with other.

    interact_force = None

    if t >= 120:
        # Interaction Phase
        # Other's action interferes with own world perception
        # (Simplified implementation of Shared Engram Model)

        # Partner also perceives world and acts
        act_other = other_agent.step(world_state)

        # Influence on Protagonist
        interact_force = act_other * 0.5

        # Shared Engram: "Shared" when P_self is high for both
        p1 = agent.history['P_self'][-1] if agent.history['P_self'] else 0
        p2 = other_agent.history['P_self'][-1] if other_agent.history['P_self'] else 0
        resonance = p1 * p2  # Resonance rate
        shared_engram.append(resonance)
    else:
        shared_engram.append(0)

    # Protagonist's action
    action = agent.step(world_state, interact_force)

    # Agent's action slightly rewrites the world (Creative Impact)
    world_state += action * 0.1

    world_history.append(world_state.copy())

print("Simulation Completed.")

# --- Data Visualization ---
S_hist = np.array(agent.history['S'])
T_hist = np.array(agent.history['T'])
A_hist = np.array(agent.history['Action'])
P_hist = np.array(agent.history['P_self'])
W_hist = np.array(world_history)
Shared_hist = np.array(shared_engram)

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# 1. Discrepancy & Attribution (Core of Theory)
# How does P(Self) move when shock occurs?
ax1 = axes[0]
ax1.set_title(r"Core Mechanism: Discrepancy vs. Self-Attribution $P(Self|E)$", fontsize=14)
ax1.plot(agent.history['Discrepancy'], color='gray', linestyle='--', label='Discrepancy (Shock)', alpha=0.7)
ax1.plot(P_hist, color='red', linewidth=2.5, label=r'Attribution $P(Self|E)$')
ax1.axvspan(50, 60, color='red', alpha=0.1, label='Trauma Event')
ax1.axvline(120, color='blue', linestyle=':', label='Partner Enters')
ax1.set_ylabel("Probability / Magnitude")
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Annotation: Why does P(Self) recover?
ax1.text(70, 0.5, "Trace Effect:\nPain becomes 'Self'", color='darkred', fontsize=9)

# 2. Trace Accumulation & Creative Action (Source of Action)
# Action (Intervention in the world) is born because there is a Trace
ax2 = axes[1]
ax2.set_title("From Suffering to Creation: Trace $\Rightarrow$ Action", fontsize=14)
trace_norm = np.linalg.norm(T_hist, axis=1)
action_norm = np.linalg.norm(A_hist, axis=1)

ax2.fill_between(range(STEPS), trace_norm, color='orange', alpha=0.3, label='Internal Trace (Engram)')
ax2.plot(action_norm, color='purple', linewidth=2, label='Creative Action (World Modification)')
ax2.set_ylabel("Magnitude")
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# 3. Shared Engram (Resonance with Other)
# Shared Trace is born when attribution probabilities of both sync
ax3 = axes[2]
ax3.set_title("Shared Engram: Interpersonal Resonance", fontsize=14)
ax3.plot(Shared_hist, color='green', linewidth=2, label='Resonance (Shared Attribution)')
ax3.set_ylabel("Interaction Strength")
ax3.set_xlabel("Time Step")
ax3.axvline(120, color='blue', linestyle=':')
ax3.text(130, 0.2, "Meaning becomes shared\nwhen both attribute it to Self", color='green')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
os.makedirs('../results/figures', exist_ok=True)
plt.savefig('../results/figures/interpersonal_resonance.png', dpi=300)

plt.show()