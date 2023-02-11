import new_patch
from IPython.display import clear_output

import constants.constants as c
import math

config = {
    "ANGLES_RANGE" : math.radians(10)
}

patch_trainer = new_patch.PatchTrainer(config=config, 
                                       mode=c.Mode.TARGET_AND_FLEE, n_epochs=3)
patch_trainer.train()
patch_trainer.save_patch("test.patch")
clear_output()