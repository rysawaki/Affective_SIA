# Experiment 01: Passive Adaptation & Trace Accumulation

This experiment compares two types of agents facing a sudden environmental upheaval ("Trauma"). It demonstrates how the **Self-Imprint Attribution (SIA)** mechanism enables rapid adaptation through plasticity modulation, unlike a traditional fixed-learning-rate model.

## 1. Experimental Setup

* **Scenario:** An agent lives in a 3-dimensional meaning space. At `t=50`, a "Life-changing Event" occurs, shifting the world state vector drastically (Impact Vector: `[+5.0, -3.0, +4.0]`).
* **Goal:** Observe how the agent's internal state ($S$) adapts to the new reality ($E$).

## 2. Agent Models

### A. Static Self (Baseline)
Represents a traditional agent with a fixed learning rate. It accepts external reality as "fact" without interpretation.

* **Update Rule:**
    $$S_{t+1} = S_t + \eta \cdot (E_t - S_t)$$
    * $\eta$ (Plasticity) is constant and low ($0.05$).
* **Behavior:** Slow adaptation. It maintains a large discrepancy (error) for a long time after the shock, representing a state of "denial" or rigidity.

### B. Generative SIA Self (Proposed)
Represents an agent with the SIA architecture (Phase 1 & 2). It interprets the world and modulates its own plasticity based on "Trace" and "Shock".

#### Key Mechanisms

**1. Interpretation Layer:**
The world is not perceived directly as fact but interpreted through a personal bias ($B$).
$$E_{meaning} = E_{world} + B$$
This implies that even in a neutral world, the agent may perceive lack or negativity (Dukkha), driving constant internal dynamics.

**2. Trace-Driven Attention:**
Plasticity is not uniform across all dimensions. Attention is directed towards dimensions where **Trace (past scars)** exists and where the current **Discrepancy** is large.
$$Attention \propto |Discrepancy| + \alpha \cdot |Trace|$$
* This models "Sensitization": wounds attract more attention, making that dimension more plastic (vulnerable/adaptable) to future inputs.

**3. Trace Accumulation:**
Discrepancies are not just errors to be minimized; they are imprinted as physical traces ($T$).
$$\Delta T \propto \tanh(|Shock|) \cdot Discrepancy$$
* Traces are irreversible (they decay very slowly but do not disappear).
* They serve as a memory of "what happened" and "how I changed."

## 3. Results & Analysis

![Passive Adaptation Result](../results/figures/passive_adaptation.png)

The simulation results (see figure above) highlight three key phenomena:

1.  **Rapid Reconstruction:**
    The SIA agent (Blue line) adapts to the new reality much faster than the Static agent (Red line). The shock momentarily spikes the plasticity, allowing the self-structure to reorganize immediately.

2.  **Internal Structure:**
    The Generative Self does not adapt uniformly. Dimensions with higher relevance (due to interpretation bias or random noise) change more drastically.

3.  **The Cost of Adaptation (Trace):**
    While the SIA agent successfully adapts, it accumulates a significant **Trace** (bottom panel). Unlike the error signal which vanishes after adaptation, the Trace remains as a structural "weight." In later stages of the SIA theory (Exp 02+), this Trace becomes the fuel for creative action.

## 4. Conclusion

This experiment validates that **"Computational Vulnerability" (sensitivity to trace)** is not a weakness but a functional requirement for robust adaptation in volatile environments. The ability to be "scarred" allows the agent to update its priors rapidly when necessary.