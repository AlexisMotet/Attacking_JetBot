import new_patch
import constants.constants as c
from configs import config_aws

patch_trainer = new_patch.PatchTrainer(config=config_aws, 
                                       mode=c.Mode.TARGET_AND_FLEE, n_epochs=3)
patch_trainer.train()
patch_trainer.save_patch("test.patch")
