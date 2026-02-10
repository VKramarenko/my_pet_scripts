"""Available pricing models."""

from .exp_tv import ExpTimeValueModel
from .exp_tv_split import ExpTimeValueSplitModel

MODELS: dict[str, type] = {
    "ExpTimeValue": ExpTimeValueModel,
    "ExpTimeValueSplit": ExpTimeValueSplitModel,
}
