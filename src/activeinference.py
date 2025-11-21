import os
import numpy as np
import matplotlib.pyplot as plt

# --- Phase 2: Active & Affective Model ---
# Focus: Agency Boost (Action -> P_self) and Affect Genesis (Trace -> Affect)
# Key Concept: "Action restores Self" & "Meaning takes time to grow"

OUTPUT_DIR = "../../results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Parameters
STEPS = 250
DIM = 3
DT = 0.1
ALPHA_TRACE = 1.0
BETA_ACTION = 1.5  # Agency Boost Introduced
GAMMA_CREATION = 0.6
PHI_NARRATIVE = 0.9  # Narrative Inertia Introduced
np.random.seed(42)


def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def softmax(x): e = np.exp(x - np.max(x)); return e / (e.sum() + 1e-8)


def cosine_sim(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return np.dot(v1, v2) / (n1 * n2) if n1 > 1e-6 and n2 > 1e-6 else 0


class Affective_SIA_Agent_v2:
    def __init__(self, name):
        self.name = name
        self.S = np.zeros(DIM)
        self.T = np.zeros(DIM)
        self.A = 0.0  # Affect is Scalar here (Magnitude only)
        self.last_act = np.zeros(DIM)
        self.history = {'P_self': [], 'Trace': [], 'Affect': [], 'Action': [], 'Disc': []}

    def step(self, world_vec, interact=None):
        E = world_vec.copy()
        if interact is not None: E += interact

        # 1. Active Attribution (Includes Action Magnitude)
        diff = E - self.S
        dist = np.linalg.norm(diff)
        trace_mag = np.linalg.norm(self.T)
        act_mag = np.linalg.norm(self.last_act)

        # v2 Logic: Agency Boost
        logit = -dist + (ALPHA_TRACE * trace_mag) + (BETA_ACTION * act_mag)
        P_self = sigmoid(logit)

        # 2. Trace Imprint
        att = softmax(np.abs(diff) + np.abs(self.T))
        self.S += 0.5 * P_self * att * diff * DT
        shock = np.tanh(dist)
        self.T += (shock * diff * P_self * DT) - (0.01 * self.T * DT)

        # 3. Affect Genesis (Scalar)
        raw_input = np.linalg.norm(self.T) * P_self
        self.A = (PHI_NARRATIVE * self.A) + ((1 - PHI_NARRATIVE) * raw_input)

        # 4. Action (Driven by Affect)
        drive = self.A * GAMMA_CREATION
        action = (self.S - world_vec) * drive
        self.last_act = action

        self.history['P_self'].append(P_self)
        self.history['Trace'].append(np.linalg.norm(self.T))
        self.history['Affect'].append(self.A)
        self.history['Action'].append(np.linalg.norm(action))
        self.history['Disc'].append(dist)
        return action


def run_simulation_v2():
    print("Running Phase 2: Active/Affective Model...")
    agent = Affective_SIA_Agent_v2("Protagonist")
    partner = Affective_SIA_Agent_v2("Partner")
    world = np.zeros(DIM)
    shared_hist = []

    for t in range(STEPS):
        if 40 <= t < 50:
            world = 0.8 * world + 0.2 * np.array([5, -4, 4])
        else:
            world *= 0.95

        interact = None
        shared = 0
        if t >= 140:
            act_b = partner.step(world, agent.last_act * 0.3)
            interact = act_b * 0.3

            # Deep Resonance v2 (Scalar Affect Sync)
            p1, p2 = agent.history['P_self'][-1], partner.history['P_self'][-1]
            c_act = max(0, cosine_sim(agent.last_act, partner.last_act))
            min_aff = min(agent.A, partner.A)
            shared = p1 * p2 * c_act * min_aff

        shared_hist.append(shared)
        act = agent.step(world, interact)
        world += act * 0.15

    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 1. Affect Genesis
    axes[0].set_title("v2: Genesis of Affect (Trace -> Affect -> Action)")
    axes[0].fill_between(range(STEPS), agent.history['Trace'], color='gray', alpha=0.2, label='Trace')
    axes[0].plot(agent.history['Affect'], color='orange', linewidth=2, label='Affect (Scalar)')
    axes[0].plot(agent.history['Action'], color='purple', linestyle='--', label='Action')
    axes[0].legend()

    # 2. Active Loop
    axes[1].set_title("v2: Agency Boost")
    axes[1].plot(agent.history['Disc'], 'k:', label='Discrepancy')
    axes[1].plot(agent.history['P_self'], 'r-', label='P(Self)')
    axes[1].legend()

    # 3. Deep Resonance
    axes[2].set_title("v2: Resonance (Sync of Action & Affect Depth)")
    axes[2].plot(shared_hist, color='green', label='Shared Meaning')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "experimental_active.png"))
    print("Phase 2 completed.")


if __name__ == "__main__":
    run_simulation_v2()