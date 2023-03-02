import patch_trainer
from configs import config

trainer = patch_trainer.PatchTrainer(config=config, 
                                       n_epochs=10, 
                                       patch_relative_size=0.08, 
                                       target_class=0, flee_class=None)
trainer.train()
trainer.save_patch("test.patch")
