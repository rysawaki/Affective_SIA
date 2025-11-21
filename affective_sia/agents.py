import numpy as np
from .core import sigmoid, softmax, compute_attribution_gate


class SIA_BaseAgent:
    """SIAエージェントの基底クラス（将来的な拡張用）"""

    def __init__(self, name, dim_meaning):
        self.name = name
        self.dim = dim_meaning
        self.S = np.zeros(dim_meaning)  # Self State
        self.T = np.zeros(dim_meaning)  # Trace
        self.last_action = np.zeros(dim_meaning)
        self.history = {}


class Identity_SIA_Agent(SIA_BaseAgent):
    """
    Identity Phase Model (v3 Final)
    痕跡(Trace) -> 情動(Vector Affect) -> 同一性(Identity) のプロセスを実装
    """

    def __init__(self, name, dim_meaning=3, dim_affect=5, affect_matrix_seed=0):
        super().__init__(name, dim_meaning)

        # 状態変数
        self.A = np.zeros(dim_affect)  # Affect Vector (Meaning)
        self.I = np.zeros(dim_affect)  # Narrative Identity (Structure)

        # パラメータ: 痕跡(Trace) -> 情動(Affect) 変換行列
        # 「痛みをどういう感情に翻訳するか」という個人の性格構造
        rng = np.random.RandomState(affect_matrix_seed)
        self.W_T2A = rng.uniform(-0.5, 0.8, (dim_affect, dim_meaning))

        # 理論パラメータ (デフォルト値)
        self.alpha_trace = 1.0
        self.beta_action = 1.5
        self.gamma_creation = 0.8
        self.eta_identity = 0.05

        # 履歴初期化
        self.history = {
            'P_self': [], 'Action_Norm': [],
            'Affect_Vec': [], 'Identity_Norm': [], 'Discrepancy': []
        }

    def step(self, world_vec, dt=0.1, interact_force=None, shared_resonance=0.0):
        """
        1ステップの更新：知覚 -> 帰属 -> 痕跡 -> 情動 -> Identity -> 行動
        """
        # 1. 解釈と帰属
        E_meaning = world_vec.copy()
        if interact_force is not None:
            E_meaning += interact_force

        diff = E_meaning - self.S
        dist = np.linalg.norm(diff)
        trace_mag = np.linalg.norm(self.T)
        action_mag = np.linalg.norm(self.last_action)

        # coreライブラリの関数を使用
        P_self = compute_attribution_gate(
            dist, trace_mag, action_mag, self.alpha_trace, self.beta_action
        )

        # 2. 痕跡形成 (Trace)
        attention = softmax(np.abs(diff) + np.abs(self.T))
        self.S += 0.5 * P_self * attention * diff * dt

        shock = np.tanh(dist)
        d_T = (shock * diff * P_self * dt) - (0.01 * self.T * dt)
        self.T += d_T

        # 3. 情動ベクトルの生成 (Vectorized Affect)
        # A(t) = decay * A(t-1) + W * T * P_self
        affect_input = np.dot(self.W_T2A, self.T) * P_self
        self.A = 0.9 * self.A + 0.1 * affect_input

        # 4. アイデンティティの形成 (Identity Update)
        # 共有された共鳴(Shared)がある時だけ、情動がアイデンティティになる
        if shared_resonance > 0.01:
            d_I = self.eta_identity * shared_resonance * self.A * dt
            self.I += d_I

        # 5. 行動生成
        drive = np.linalg.norm(self.A) * self.gamma_creation
        action_vec = (self.S - world_vec) * drive
        self.last_action = action_vec

        # 履歴記録
        self.history['P_self'].append(P_self)
        self.history['Action_Norm'].append(np.linalg.norm(action_vec))
        self.history['Affect_Vec'].append(self.A.copy())
        self.history['Identity_Norm'].append(np.linalg.norm(self.I))
        self.history['Discrepancy'].append(dist)

        return action_vec