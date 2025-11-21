from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """
    SIAシミュレーションの全パラメータを管理する設定クラス
    論文記載のパラメータ値はここを一元的に参照する
    """
    # --- 実験の基本設定 ---
    steps: int = 300
    dt: float = 0.1
    random_seed: int = 45  # 再現性確保のためのシード

    # --- 空間の次元 ---
    dim_meaning: int = 3  # 知覚・痕跡空間 (S, T)
    dim_affect: int = 5  # 情動空間 (A)

    # --- SIA理論コアパラメータ (Magic Numbersの排除) ---
    alpha_trace: float = 1.0  # α: 痕跡への感受性 (Trace Sensitivity)
    beta_action: float = 1.5  # β: 行為主体感 (Agency Boost)
    gamma_creation: float = 0.8  # γ: 創造的行動の強さ (Creative Drive)

    # --- Identity形成パラメータ ---
    eta_identity: float = 0.05  # η: アイデンティティ蓄積率

    # --- 閾値など ---
    resonance_threshold: float = 0.01  # 共鳴判定の閾値

    def __post_init__(self):
        """不正なパラメータのチェックなどをここで行える"""
        if self.alpha_trace < 0:
            raise ValueError("Alpha must be non-negative")