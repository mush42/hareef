import torch
from lightning.pytorch.utilities.model_summary import summarize
from hareef.sarf.model import SarfModel
from hareef.sarf.config import Config

config = Config("config/sarf/dev.json")

model = SarfModel(config)
print(summarize(model))

inputs = torch.randint(0, 20, (1, 124,))
lengths = torch.LongTensor([124])
out = model(inputs, lengths)
