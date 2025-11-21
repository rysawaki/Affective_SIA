# バージョン情報
__version__ = '0.1.0'

# ユーザーが使いやすいように、主要クラスをトップレベルで公開する
# これを書くと from affective_sia import Identity_SIA_Agent と書けるようになります
from .agents import Identity_SIA_Agent
from .config import SimulationConfig
from .core import sigmoid, compute_attribution_gate