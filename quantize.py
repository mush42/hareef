import more_itertools
import quanto 
import torch
from lightning import Trainer

from hareef.sarf.config import Config
from hareef.sarf.dataset import load_training_data
from hareef.sarf.model import SarfModel

config = Config("config/sarf/dev.json")
model = SarfModel.load_from_checkpoint(
    "./logs/sarf/lightning_logs/version_0/epoch=42-step=187265.ckpt",
    map_location="cpu",
    config=config
)

# 1. dynamic quant
quanto.quantize(model, weights=quanto.qint8, activations=quanto.qint8)

# 2. calibration
data_loader = load_training_data(config, batch_size=64)
batch = next(iter(data_loader))
with quanto.Calibration(momentum=0.9):
    model._process_batch(batch)

model.to("cuda")
model.automatic_optimization = False
model.train()
train_loader = more_itertools.take(
    8192,
    load_training_data(config)
)
device = model.device
opt = model.optimizers()
for batch_idx, batch in enumerate(train_loader):
    opt.zero_grad()

    predictions, pos_predictions = model(
        batch["chars"].to(device),
        batch["diacs"].to(device),
        batch["lengths"].to("cpu"),
    )
    predictions = predictions.dequantize()
    pos_predictions = pos_predictions.dequantize()
    target = batch["target"].contiguous()
    pos_target = batch["char2pos"].contiguous()

    predictions = predictions.view(-1, predictions.shape[-1])
    targets = batch["target"].view(-1)
    pos_predictions = pos_predictions.view(-1, pos_predictions.shape[-1])
    pos_targets = pos_target.view(-1)

    diac_loss = model.criterion(predictions.to(device), targets.to(device))
    pos_loss = model.pos_criterion(
        pos_predictions.to(device), pos_targets.to(device)
    )

    loss = (2.0 * pos_loss) + (8.0 * diac_loss)

    model.manual_backward(loss)
    opt.step()

 
quanto.freeze(model)