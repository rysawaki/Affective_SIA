# Experiment 04: Active Agency & The Loop of Self-Recovery

This experiment introduces the concept of **"Sense of Agency"** into the SIA framework. Previous experiments modeled the self as a passive entity that adapts to or resists environmental changes. Here, we demonstrate that **action itself is a prerequisite for self-attribution**, creating a closed loop where "I act, therefore I am."

## 1. Theoretical Background

### The Passive Self Problem
In standard predictive coding, minimizing prediction error is the primary goal. However, this does not explain the **Sense of Agency**—the feeling that "I am the one causing this change."
Without agency, an agent cannot distinguish between a change caused by the environment (passive) and a change caused by itself (active).

### The Active SIA Solution
We propose that the **Attribution Gate** is modulated not only by static memory (Trace) but also by dynamic behavior (Action).

## 2. Key Mechanisms

### A. Agency Boost ($\beta$)
We modify the core attribution equation to include the magnitude of recent actions.

$$P(Self|E) = \sigma \left( -\|E - S\| + \alpha \|T\| + \mathbf{\beta \|Action_{prev}\|} \right)$$

* **$\beta$ (Agency Parameter):** Controls how much acting on the world reinforces the sense of self.
* **Mechanism:** When the agent acts strongly, $P(Self)$ increases regardless of the current discrepancy. This models the phenomenon: **"I recognize this experience as mine because I caused it."**

### B. Deep Resonance (Action Synchronization)
Resonance is no longer just about feeling the same way (Affect Sync); it requires moving in the same direction.

$$\text{Resonance} = P_1 P_2 \cdot \underbrace{\cos(\vec{A}_1, \vec{A}_2)}_{\text{Action Sync}} \cdot \min(\text{Affect}_1, \text{Affect}_2)$$

* True resonance requires **Joint Attention / Joint Action**. Even if two agents share similar traces, they cannot form a Shared Engram if their actions are misaligned ($\cos \theta < 0$).

## 3. Results & Analysis

![Active Agency Result](../results/figures/active_agency.png)

The simulation results (see figure above) reveal the emergence of the **Action-Perception Loop**:

1.  **Genesis of Affect (Top Panel):**
    * The **Orange line (Affect)** acts as a buffer, integrating the raw **Trace (Gray area)** over time.
    * Action (Purple line) does not spike immediately after trauma but follows the maturation of Affect. This represents the "gestation" period of meaning.

2.  **The Self-Attribution Loop (Middle Panel):**
    * **Crucial Observation:** Look at the recovery phase (post-trauma). The **Red line ($P_{self}$)** is boosted *after* the action intensity increases.
    * This confirms the **Agency Boost** hypothesis: The agent recovers its sense of self *through* the act of expression. Action is not just an output; it is a mechanism for healing.

3.  **The Fragility of Resonance (Bottom Panel):**
    * The **Green line (Shared Meaning)** is volatile and spiky compared to previous passive models.
    * This "instability" reflects reality: maintaining perfect synchronization in both *internal affect* and *external action* is computationally difficult.
    * The peaks represent "moments of meeting"—fleeting but powerful instances where intention and emotion perfectly align between two agents.

## 4. Conclusion

Experiment 04 validates that **Identity is not a static container but a dynamic loop.**
By acting on the world, the SIA agent bootstraps its own self-attribution, transforming from a passive victim of trauma into an active author of its narrative.