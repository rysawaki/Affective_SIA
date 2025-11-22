import numpy as np
import matplotlib.pyplot as plt


def plot_identity_simulation(agent, resonance_history, shared_engram_history, config):
    """
    Visualizes the Identity Formation process (Corresponding to Figure 1 in the paper).

    Args:
        agent: The protagonist agent (containing history data).
        resonance_history: History of resonance components [[act_sync, aff_sync], ...].
        shared_engram_history: History of shared engram strength [float, ...].
        config: Simulation configuration object.
    """
    steps = config.steps

    # Data formatting
    aff_data = np.array(agent.history['Affect_Vec'])  # (Steps, 5)
    # Padding resonance history to match steps if necessary (due to interaction start delay)
    res_data = np.array(resonance_history)
    if len(res_data) < steps:
        padding_len = steps - len(res_data)
        padding = np.zeros((padding_len, 2))
        res_data = np.vstack([padding, res_data])

        padding_shared = np.zeros(padding_len)
        shared_engram_history = np.concatenate([padding_shared, shared_engram_history])

    # --- Plotting ---
    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1, 1])

    # 1. Vectorized Affect Stream (Quality of Meaning)
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title("Vectorized Affect Structure: The Quality of Meaning", fontsize=14)

    colors = ['#FF5733', '#33FF57', '#3357FF', '#F333FF', '#FF33A8']
    labels = ['Sorrow', 'Hope', 'Isolation', 'Solidarity', 'Creation']

    # Plot affective dimensions safely
    dim_to_plot = min(aff_data.shape[1], 5)
    for i in range(dim_to_plot):
        ax1.plot(aff_data[:, i], label=labels[i] if i < 5 else f"Dim {i}",
                 color=colors[i] if i < 5 else None, alpha=0.8, linewidth=1.5)

    # Event lines (Trauma and Encounter)
    # Note: Hardcoded for visualization consistency with the paper scenario
    ax1.axvspan(50, 60, color='gray', alpha=0.2, label='Trauma')
    ax1.axvline(100, color='blue', linestyle='--', label='Encounter')

    ax1.set_ylabel("Affect Intensity")
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax1.grid(True, alpha=0.3)

    # 2. The Anatomy of Resonance (Action vs Affect Sync)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_title("The Anatomy of Resonance: Action vs. Affect", fontsize=14)

    ax2.plot(res_data[:, 0], color='purple', linestyle=':', label='Action Sync (Behavior)')
    ax2.plot(res_data[:, 1], color='orange', linewidth=2, label='Affect Sync (Meaning)')
    ax2.fill_between(range(steps), shared_engram_history, color='green', alpha=0.3, label='Total Shared Engram')

    ax2.set_ylabel("Correlation")
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Annotation for misalignment
    ax2.text(110, 0.8, "Misalignment:\nDoing same thing,\nfeeling differently", color='red', fontsize=9)

    # 3. Identity Formation (The Final Goal)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_title("Narrative Identity Formation (I accumulated from Shared Affect)", fontsize=14)

    id_norm = np.array(agent.history['Identity_Norm'])
    ax3.plot(id_norm, color='black', linewidth=3, label='Identity Strength')
    ax3.fill_between(range(steps), id_norm, color='black', alpha=0.1)

    ax3.set_ylabel("Identity Magnitude")
    ax3.set_xlabel("Time Step")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Annotation for the key insight
    ax3.text(200, max(id_norm) * 0.5 if max(id_norm) > 0 else 0, "I become who we were", color='black', fontsize=12,
             fontweight='bold')

    plt.tight_layout()
    # plt.show() is removed to allow saving the figure in the calling script.