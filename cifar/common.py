from enum import Enum


class LossType(Enum):
    ORIGINAL = 0
    L1_SPARSITY_REGULARIZATION = 1
    POLARIZATION = 4

    @staticmethod
    def from_string(desc: str):
        mapping = LossType.loss_name()
        return mapping[desc.lower()]

    @staticmethod
    def loss_name():
        return {"original": LossType.ORIGINAL,
                "sr": LossType.L1_SPARSITY_REGULARIZATION,
                "zol": LossType.POLARIZATION,
                }
