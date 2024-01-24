import torch
from lightning.pytorch.utilities.model_summary import summarize
from hareef.sarf.dataset import load_inference_data, load_test_data
from hareef.sarf.model import SarfModel
from hareef.sarf.config import Config

config = Config("config/sarf/dev.json")
loader = load_test_data(
    config,
)
batch = next(iter(loader))

model = SarfModel(config)
print(summarize(model))

loss = model._process_batch(batch)