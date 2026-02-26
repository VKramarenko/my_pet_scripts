"""Available pricing models."""

from .exp_tv_shared_c import ExpTimeValueSharedCModel
from .exp_tv_split import ExpTimeValueSplitModel

MODELS: dict[str, type] = {
    "ExpTimeValueSharedC": ExpTimeValueSharedCModel,
    "ExpTimeValueSplit": ExpTimeValueSplitModel,
}
