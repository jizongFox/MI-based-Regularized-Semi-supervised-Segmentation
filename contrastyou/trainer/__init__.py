from .contrast_trainer import ContrastTrainer, ContrastTrainerMT
from .iic_trainer import IICContrastTrainer

trainer_zoos = {"contrast": ContrastTrainer, "contrastMT": ContrastTrainerMT, "iiccontrast": IICContrastTrainer}
