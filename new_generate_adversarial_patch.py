import new_patch
from configs import config_colab

for i in range(3):
    patch_trainer = new_patch.PatchTrainer(config=config_colab, n_epochs=5, patch_relative_size=0.08)
    patch_trainer.train()
    patch_trainer.save_patch("test.patch")
