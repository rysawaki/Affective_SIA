from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """
    Simulation Configuration for SIA Model.
    Manages all parameters for the simulation centrally.
    """
    # --- Basic Experiment Settings ---
    steps: int = 300
    dt: float = 0.1
    random_seed: int = 45  # Seed for reproducibility

    # --- Dimensionality ---
    dim_meaning: int = 3  # Perception/Trace Space (S, T)
    dim_affect: int = 5  # Affect Space (A)

    # --- SIA Core Theoretical Parameters (No Magic Numbers) ---
    alpha_trace: float = 1.0  # alpha: Sensitivity to Trace (Trace Sensitivity)
    beta_action: float = 1.5  # beta: Sense of Agency (Agency Boost)
    gamma_creation: float = 0.8  # gamma: Strength of Creative Action (Creative Drive)

    # --- Identity Formation Parameters ---
    eta_identity: float = 0.05  # eta: Identity accumulation rate

    # --- Thresholds ---
    resonance_threshold: float = 0.01  # Threshold for resonance detection

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.alpha_trace < 0:
            raise ValueError("Alpha must be non-negative")