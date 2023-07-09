# coding: utf-8

import numpy as np
import torch as t
from hareef.mashcool.dataset import load_validation_data
from hareef.mashcool.config import Config
from hareef.mashcool.model import  MashcoolModel


config = Config("config/mashcool/config.json")
model = MashcoolModel(config)


# viter = iter(load_validation_data(config))
# batch = next(viter)
# ret = model._process_batch(batch)


input_array = np.array([i for i in range(20)])
input_array = np.expand_dims(input_array, axis=0)
inputs = t.LongTensor(input_array)
ret = model(inputs)
print(ret)
