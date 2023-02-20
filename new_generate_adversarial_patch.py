import new_patch
from configs import config_

patch_trainer = new_patch.PatchTrainer(config=config_, n_epochs=3, patch_relative_size=0.08)
patch_trainer.train()
patch_trainer.save_patch("test.patch")
