# Experiment 05: Narrative Identity Formation & The Shared Engram

This experiment represents the culmination of the SIA framework. It moves beyond scalar "pain" or "pleasure" to model **Affective Qualia (Vectorized Meaning)** and defines **Narrative Identity** not as a static attribute, but as a dynamic integral of shared histories.

## 1. Theoretical Framework

### A. Vectorized Affect (Qualia)
In previous models, "Trace" was a simple magnitude. Here, we introduce the **Affect Transformation Matrix ($W_{T \to A}$)**, which converts raw physical traces into qualitative emotional vectors.

$$\mathbf{A}(t) = \phi \mathbf{A}(t-1) + (1-\phi) W_{T \to A} \mathbf{T}(t)$$

* **$\mathbf{A} \in \mathbb{R}^5$:** Represents complex states such as [Sorrow, Hope, Isolation, Solidarity, Creation].
* **Implication:** Two agents may have similar magnitudes of pain (Trace) but experience completely different qualitative meanings (Affect) depending on their internal matrix structure.

### B. Multi-Level Resonance
Resonance is strictly defined. It requires synchronization on two distinct levels:
1.  **Behavioral Sync ($\cos \theta_{act}$):** Are we doing the same thing?
2.  **Affective Sync ($\cos \theta_{aff}$):** Are we feeling the same meaning?

$$\text{Shared}(t) = P_1 P_2 \cdot \cos(\theta_{act}) \cdot \cos(\theta_{aff}) \cdot \min(\|\mathbf{A}_1\|, \|\mathbf{A}_2\|)$$

* **Misalignment:** If agents act together but feel differently ($\cos \theta_{aff} < 0$), no Shared Engram is formed. This models "empty rituals" or "misunderstandings."

### C. Identity Integration
Identity ($I$) is defined as the accumulated history of these shared moments.

$$I(t) = \int_{0}^{t} \eta \cdot \text{Shared}(\tau) \cdot \mathbf{A}(\tau) \, d\tau$$

* **Hypothesis:** "I become who we were." My identity is the crystallized residue of resonant interactions.

## 2. Results & Analysis

![Identity Formation](../../results/figures/identity_formation.png)

The simulation results (see figure above) illustrate the genesis of a "We-Identity":

### Phase 1: The Internal Drama (Top Panel)
* **Vectorized Affect Structure:** Following the trauma ($t=50$), the agent's internal state explodes into various affective colors.
* Initially dominated by **Sorrow (Red)**, the affect gradually shifts towards **Creation (Pink)** and **Solidarity (Purple)**. This represents the "internal narrative" evolving even before true contact is made.

### Phase 2: The Anatomy of Resonance (Middle Panel)
* **The Gap between Action and Affect:**
    * At $t=100$ (Encounter), **Action Sync (Purple Dotted)** spikes immediately. The agents are cooperating behaviorally.
    * However, **Affect Sync (Orange Line)** lags behind. For a period, they are "doing the same thing but feeling differently."
* **The Green Zone:** Only when both lines are high ($t > 150$) does the **Shared Engram (Green Area)** emerge. This green zone represents the moments of true connection.

### Phase 3: Identity Accumulation (Bottom Panel)
* **The Black Line ($I$):** Notice that the Identity line remains flat (zero) until the Shared Engram appears.
* **Conclusion:** Private experiences (Trace/Affect) shape the *potential* for identity, but only *shared* experiences crystallize into **Narrative Identity**. The final slope represents the robust formation of a social self.

## 3. Conclusion

Experiment 05 provides a computational proof of concept for **Affective Active Inference**. It demonstrates that:
1.  **Qualia matter:** The vector direction of affect determines compatibility.
2.  **Resonance is rare:** It requires simultaneous alignment of agency, affect, and cognition.
3.  **Identity is historical:** It is the integral of successfully shared meanings over time.