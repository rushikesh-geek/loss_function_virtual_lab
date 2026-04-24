# losses/__init__.py
from losses.focal_loss import FocalLoss
from losses.loss_registry import get_loss

__all__ = ["FocalLoss", "get_loss"]
