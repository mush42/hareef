# coding: utf-8

import numpy as np
import torch as t
from hareef.mashkool.dataset import load_validation_data
from hareef.mashkool.config import Config
from hareef.mashkool.model import  MashkoolModel


config = Config("config/mashkool/config.json")
model = MashkoolModel(config)


# viter = iter(load_validation_data(config))
# batch = next(viter)
# ret = model._process_batch(batch)


input_array = np.array([[i for i in range(20)]  * 2]).reshape(2, 20)
inputs = t.LongTensor(input_array)
ret = model(inputs)
print(ret)
