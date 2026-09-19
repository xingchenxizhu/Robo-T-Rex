"""机器霸王龙尾巴-身体协同强化学习实验包。"""

from .env import TASKS, MODES, TrexEnv, DIMENSION_FOR_TASK
from .model import build_xml, export_models, load_config, load_model, tail_profile

__all__ = [
    "TrexEnv",
    "TASKS",
    "MODES",
    "DIMENSION_FOR_TASK",
    "build_xml",
    "load_model",
    "load_config",
    "export_models",
    "tail_profile",
]

__version__ = "0.1.0"
