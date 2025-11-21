from setuptools import setup, find_packages

setup(
    name="affective_sia",
    version="0.1.0",
    description="A Computational Framework for Identity Formation via Shared Affective Resonance",
    author="Ryota Sawaki",  # あなたのお名前（任意）
    packages=find_packages(),  # ディレクトリ内の __init__.py を探してパッケージとして認識する
    install_requires=[
        "numpy",
        "matplotlib",
    ],
)