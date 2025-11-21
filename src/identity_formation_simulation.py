import numpy as np
import matplotlib.pyplot as plt

# --- 研究用パラメータ設定 ---
STEPS = 200
DIM = 3                   # 意味空間の次元 (例: [安定性, 親密性, 自己価値])
DT = 0.1                  # 時間刻み
ALPHA_TRACE = 1.2         # α: 痕跡が自己帰属を強化する係数 (Re-enactment factor)
BETA_PLASTICITY = 0.5     # 可塑性の基本係数
GAMMA_ACTION = 0.4        # 創造的行動の強さ係数

# 乱数シード固定（再現性のため）
np.random.seed(42)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum() + 1e-8)

class SIA_Agent:
    def __init__(self, name, sensitivity=1.0):
        self.name = name
        self.S = np.zeros(DIM)   # Self State (自我)
        self.T = np.zeros(DIM)   # Trace (痕跡)
        self.sensitivity = sensitivity # 個体差（感受性）
        
        # 記録用
        self.history = {
            'S': [], 'T': [], 'Action': [], 
            'P_self': [], 'Loss': [], 'Discrepancy': []
        }

    def compute_attribution(self, E_meaning):
        """
        SIA理論の核心式: P(Self|E)
        経験Eが「自己に属する」確率を計算する。
        
        Args:
            E_meaning (np.array): 解釈された経験ベクトル
        Returns:
            P_self (float): 自己帰属確率 (0.0 - 1.0)
            dist (float): 純粋な意味のズレ
        """
        # 1. 意味的乖離 (Discrepancy)
        diff = E_meaning - self.S
        dist = np.linalg.norm(diff)
        
        # 2. 痕跡による引力 (Trace Gravity)
        # 痕跡が深いほど、「これは私の物語だ」と感じやすくなる（たとえ苦痛でも）
        trace_magnitude = np.linalg.norm(self.T)
        
        # 3. ロジット計算 (SIA Core Equation)
        # 乖離が大きいと帰属低下(-dist)、痕跡があると帰属上昇(+alpha*trace)
        logit = -dist + (ALPHA_TRACE * trace_magnitude)
        
        P_self = sigmoid(logit)
        return P_self, dist, diff

    def step(self, world_vec, interact_force=None):
        """
        1ステップの更新：知覚 -> 帰属 -> 痕跡形成 -> 行動
        """
        # --- 1. 解釈 (Interpretation) ---
        # ここではシンプルに世界ベクトルをそのまま意味とするが、
        # 本来は Interpretation Layer が入る (Meaning = f(World))
        E_meaning = world_vec.copy()
        if interact_force is not None:
            E_meaning += interact_force # 他者からの影響
            
        # --- 2. 自己帰属 (Attribution) ---
        P_self, dist, diff = self.compute_attribution(E_meaning)
        
        # --- 3. 損失関数の計算 (理論検証用) ---
        # L = -log(P(Self|E)) + Regularization
        # 自我はこのLossを最小化するように動くはずである
        loss = -np.log(P_self + 1e-8)
        
        # --- 4. 更新 (Plasticity & Trace) ---
        
        # 痕跡が強い次元ほど、注意が向く (Attention)
        attention = softmax(np.abs(diff) + np.abs(self.T))
        
        # 自我の可塑性 (Plasticity)
        # P_selfが高い（自分のことだと思っている）時だけ、深く学習する
        learning_rate = BETA_PLASTICITY * P_self * attention
        d_S = learning_rate * diff * DT
        self.S += d_S
        
        # 痕跡の形成 (Imprinting)
        # ショックが強く、かつ「自分のこと」だと認めた時だけ刻まれる
        shock = np.tanh(dist)
        d_T = (shock * diff * P_self * DT) - (0.01 * self.T * DT) # 減衰項あり
        self.T += d_T
        
        # --- 5. 創造的行動 (Creative Action) ---
        # 痕跡を解消するために世界に働きかける力
        # Action = - grad(Loss) 的な動き
        # 痕跡が深く、かつ自己帰属が高いほど、強い表現が生まれる
        
        action_magnitude = np.linalg.norm(self.T) * GAMMA_ACTION * P_self
        # 方向は「自我 - 世界」（世界を自分に引き寄せる方向）
        action_vec = (self.S - world_vec) * action_magnitude
        
        # --- 履歴保存 ---
        self.history['S'].append(self.S.copy())
        self.history['T'].append(self.T.copy())
        self.history['Action'].append(action_vec.copy())
        self.history['P_self'].append(P_self)
        self.history['Loss'].append(loss)
        self.history['Discrepancy'].append(dist)
        
        return action_vec

# --- シミュレーション実行 ---

# シナリオ設定:
# 0-50: 平穏
# 50-60: トラウマ的イベント (Trauma Impact)
# 60-120: 回復と痕跡の定着
# 120-200: 他者との出会い (Shared Engram Phase)

agent = SIA_Agent("Protagonist")
world_history = []

# 世界の状態ベクトル (最初はゼロ付近)
world_state = np.zeros(DIM)

# 他者エージェント（後半で登場）
other_agent = SIA_Agent("Partner")
shared_engram = [] # 共有された痕跡の強度

print("Simulation Started...")

for t in range(STEPS):
    # --- 世界の変動 ---
    if 50 <= t < 60:
        # トラウマ衝撃: 巨大な意味の乖離を強制的に与える
        world_target = np.array([5.0, -5.0, 5.0]) 
        world_state = 0.8 * world_state + 0.2 * world_target
    elif t == 60:
        # 衝撃が去り、世界は戻るが...
        world_target = np.zeros(DIM)
    else:
        # 通常の微細なゆらぎ
        world_state += np.random.normal(0, 0.05, DIM)
        # ゆっくりと元に戻ろうとする世界（弾性）
        world_state *= 0.95

    # --- エージェントの更新 ---
    # 前半は単独、後半は他者との相互作用
    
    interact_force = None
    
    if t >= 120:
        # 他者との相互作用フェーズ
        # 他者の行動が、自分の世界認識に干渉する
        # (Shared Engram Modelの簡易実装)
        
        # Partnerも世界を知覚し行動する
        act_other = other_agent.step(world_state)
        
        # Protagonistへの影響
        interact_force = act_other * 0.5
        
        # Shared Engram: 互いのP_selfが高いほど「共有」される
        p1 = agent.history['P_self'][-1] if agent.history['P_self'] else 0
        p2 = other_agent.history['P_self'][-1] if other_agent.history['P_self'] else 0
        resonance = p1 * p2 # 共鳴率
        shared_engram.append(resonance)
    else:
        shared_engram.append(0)

    # 主人公の行動
    action = agent.step(world_state, interact_force)
    
    # エージェントの行動が世界をわずかに書き換える (Creative Impact)
    world_state += action * 0.1
    
    world_history.append(world_state.copy())

print("Simulation Completed.")

# --- データの可視化 ---
S_hist = np.array(agent.history['S'])
T_hist = np.array(agent.history['T'])
A_hist = np.array(agent.history['Action'])
P_hist = np.array(agent.history['P_self'])
W_hist = np.array(world_history)
Shared_hist = np.array(shared_engram)

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# 1. Discrepancy & Attribution (理論の核心)
# 衝撃が起きた時、P(Self)はどう動くか？
ax1 = axes[0]
ax1.set_title(r"Core Mechanism: Discrepancy vs. Self-Attribution $P(Self|E)$", fontsize=14)
ax1.plot(agent.history['Discrepancy'], color='gray', linestyle='--', label='Discrepancy (Shock)', alpha=0.7)
ax1.plot(P_hist, color='red', linewidth=2.5, label=r'Attribution $P(Self|E)$')
ax1.axvspan(50, 60, color='red', alpha=0.1, label='Trauma Event')
ax1.axvline(120, color='blue', linestyle=':', label='Partner Enters')
ax1.set_ylabel("Probability / Magnitude")
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# 注釈: なぜP(Self)が回復するのか？
ax1.text(70, 0.5, "Trace Effect:\nPain becomes 'Self'", color='darkred', fontsize=9)

# 2. Trace Accumulation & Creative Action (行動の源泉)
# 痕跡があるからこそ、行動（世界への介入）が生まれる様子
ax2 = axes[1]
ax2.set_title("From Suffering to Creation: Trace $\Rightarrow$ Action", fontsize=14)
trace_norm = np.linalg.norm(T_hist, axis=1)
action_norm = np.linalg.norm(A_hist, axis=1)

ax2.fill_between(range(STEPS), trace_norm, color='orange', alpha=0.3, label='Internal Trace (Engram)')
ax2.plot(action_norm, color='purple', linewidth=2, label='Creative Action (World Modification)')
ax2.set_ylabel("Magnitude")
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# 3. Shared Engram (他者との共鳴)
# 二人の帰属確率が同期した時、共有痕跡が生まれる
ax3 = axes[2]
ax3.set_title("Shared Engram: Interpersonal Resonance", fontsize=14)
ax3.plot(Shared_hist, color='green', linewidth=2, label='Resonance (Shared Attribution)')
ax3.set_ylabel("Interaction Strength")
ax3.set_xlabel("Time Step")
ax3.axvline(120, color='blue', linestyle=':')
ax3.text(130, 0.2, "Meaning becomes shared\nwhen both attribute it to Self", color='green')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
