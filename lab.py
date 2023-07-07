# coding: utf-8

import numpy as np
import torch as t
from hareef.shakkala.dataset import load_validation_data
from hareef.shakkala.config import Config
from hareef.shakkala.model import ShakkalaModel


config = Config("config/shakkala/config.json")
model = ShakkalaModel(config)


# viter = iter(load_validation_data(config))
# batch = next(viter)
# ret = model._process_batch(batch)


input_array = np.array([i for i in range(20)])
input_array = np.expand_dims(input_array, axis=0)
inputs = t.LongTensor(input_array)
hints_array = np.array([i for i in range(5)] * 4)
hints_input = t.LongTensor(np.expand_dims(hints_array, axis=0))
ret = model(inputs, hints_input)
print(ret)
