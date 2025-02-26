# -*- coding: utf-8 -*-
"""
@author: Syed Hasib Akhter Faruqui
@email: shafnehal@gmail.com
@web: www.shafnehal.com

Note: Set GPU use to Growth/As needed
"""
import tensorflow as tf

def set_memory_growth():
    ##################################### Setting up GPU  #########################################################
    # Consume Memory as Needed
    physical_devices = tf.config.list_physical_devices('GPU')

    if len(physical_devices)>0:
        try:
            if len(physical_devices) > 1:
                tf.config.experimental.set_memory_growth(physical_devices[0:1], True) # Selecting only the first GPU
            else:
                tf.config.experimental.set_memory_growth(physical_devices[0], True) # IF the device have one GPU then this will execute
        except:
          # Invalid device or cannot modify virtual devices once initialized.
          pass
    else:
        print('No GPU Detected! CPU Training will Run!')
    ##################################### Setting up GPU  #########################################################