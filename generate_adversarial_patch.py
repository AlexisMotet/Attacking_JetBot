import new_patch
from configs import config

patch_trainer = new_patch.PatchTrainer(config=config, 
                                       n_epochs=10, 
                                       patch_relative_size=0.08, 
                                       target_class=0, flee_class=None)
patch_trainer.train()
patch_trainer.save_patch("test.patch")
