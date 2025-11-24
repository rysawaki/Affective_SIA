# Version Information
__version__ = '0.1.0'

# Expose main classes at the top level for user convenience
# This allows: from affective_sia import Identity_SIA_Agent
from .agents import Identity_SIA_Agent
from .config import SimulationConfig
from .core import sigmoid, compute_attribution_gate
