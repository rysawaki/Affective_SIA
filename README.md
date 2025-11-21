# Affective_SIA


Generative Identity: Self-Imprint Attribution (SIA) Model

A Computational Framework for Identity Formation via Shared Affective Resonance

Abstract

This repository contains the reference implementation of the Self-Imprint Attribution (SIA) theory. Unlike traditional Reinforcement Learning or Active Inference models which treat prediction errors as costs to be minimized, SIA posits that identity is formed by attributing, imprinting, and sharing these errors as affective meanings.

Repository Structure

src/sia_model.py: The Final Model (v3). Implements Vectorized Affect and Identity Integration. This is the core artifact of the theory.

src/experimental/: Contains the evolutionary history of the theory.

01_passive_research.py: (v1) The Passive Model. Demonstrates "Trace $\to$ Action" mechanics (Pain drives Creation).

02_active_affective.py: (v2) The Active Model. Introduces "Agency Boost" ($P(Self) \leftarrow Action$) and scalar Affect Genesis.

Core Theory (v3 Final)

The model operates on a cycle of five phases:

Trace (T): Physical accumulation of discrepancy (Trauma).

Attribution (P_self): The gate of ownership. $P(Self|E) \propto Agency + Trace$.

Affect (A): Vectorized meaning (e.g., Sorrow, Creation) generated from Traces.

Action (Act): Creative intervention to resolve internal Affect.

Identity (I): The integral of shared affective history.

Key Equation: Identity Integration

Identity is defined not as a static attribute, but as the time-integral of shared affective resonance:

$$I(t) = \int_{0}^{t} \text{Shared}(\tau) \cdot A(\tau) \, d\tau$$

Installation & Usage

Clone the repository:

git clone [https://github.com/YOUR_USERNAME/SIA-Generative-Identity.git](https://github.com/YOUR_USERNAME/SIA-Generative-Identity.git)


Install dependencies:

pip install -r requirements.txt


Run the simulation:

Final Model: python src/sia_model.py

Evolution History: python src/experimental/01_passive_research.py

License

MIT License.
