import numpy as np


def sigmoid(x):
    """
    Sigmoid function: Converts input to probability between 0.0 and 1.0.
    """
    # To prevent overflow, input clipping could be used, but
    # we maintain the pure form as a theoretical model.
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    """
    Softmax function: Converts a vector into a probability distribution.
    """
    # Subtract max for numerical stability (exp(x - max))
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum() + 1e-8)  # Prevent division by zero


def cosine_similarity(v1, v2):
    """
    Cosine similarity: Calculates the closeness of direction between two vectors (-1.0 ~ 1.0).
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return np.dot(v1, v2) / (n1 * n2)


def compute_attribution_gate(discrepancy, trace_mag, action_mag, alpha, beta):
    """
    Core Equation of SIA Theory: Calculates Self-Attribution Probability P(Self|E).

    Args:
        discrepancy (float): Semantic discrepancy (Prediction Error).
        trace_mag (float): Magnitude of Trace.
        action_mag (float): Intensity of Action (Agency Boost).
        alpha (float): Trace sensitivity parameter.
        beta (float): Agency boost parameter.

    Returns:
        float: Self-Attribution Probability (0.0 ~ 1.0).
    """
    logit = -discrepancy + (alpha * trace_mag) + (beta * action_mag)
    return sigmoid(logit)