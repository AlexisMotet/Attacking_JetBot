import new_patch
import constants.constants as c
from configs import config

patch_trainer = new_patch.PatchTrainer(config=config, n_epochs=2, patch_relative_size=0.08)
patch_trainer.train()
patch_trainer.save_patch("test.patch")
