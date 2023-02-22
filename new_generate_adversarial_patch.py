import new_patch
from configs import config_dataset0, config_dataset1, config_dataset2

patch_trainer = new_patch.PatchTrainer(config=config_dataset0, n_epochs=40, patch_relative_size=0.08)
patch_trainer.train()
patch_trainer.save_patch("C_%d.patch" % 0)
