# ------------------------------------------------------------ #
#
# file : utils/learning/callbacks.py
# author : CM
# Custom callbacks
#
# ------------------------------------------------------------ #

import numpy as np
from keras.callbacks import LearningRateScheduler

# reduce learning rate on each epoch
def learningRateSchedule(initialLr=1e-4, decayFactor=0.99, stepSize=1):
    def schedule(epoch):
        lr = initialLr * (decayFactor ** np.floor(epoch / stepSize))
        print("Learning rate : ", lr)
        return lr
    return LearningRateScheduler(schedule)