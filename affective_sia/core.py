import numpy as np


def sigmoid(x):
    """シグモイド関数: 入力を 0.0 ~ 1.0 の確率に変換する"""
    # オーバーフロー対策のため、入力値をクリップしても良いが、
    # 理論モデルとしては純粋な形を維持する。
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    """ソフトマックス関数: ベクトルを確率分布に変換する"""
    # 数値安定性のために最大値を引く (exp(x - max))
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum() + 1e-8)  # ゼロ除算防止


def cosine_similarity(v1, v2):
    """コサイン類似度: 2つのベクトルの向きの近さを計算 (-1.0 ~ 1.0)"""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return np.dot(v1, v2) / (n1 * n2)


def compute_attribution_gate(discrepancy, trace_mag, action_mag, alpha, beta):
    """
    SIA理論の核心式: 自己帰属確率 P(Self|E) を計算する

    Args:
        discrepancy (float): 意味の不整合（予測誤差）
        trace_mag (float): 痕跡の強度
        action_mag (float): 行為の強度（Agency Boost）
        alpha (float): 痕跡感受性パラメータ
        beta (float): 行為主体感パラメータ

    Returns:
        float: 自己帰属確率 (0.0 ~ 1.0)
    """
    logit = -discrepancy + (alpha * trace_mag) + (beta * action_mag)
    return sigmoid(logit)