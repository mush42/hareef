# Hareef 

**Hareef** is an implementation of state-of-the-art models for diacritics restoration for Arabic language.

## Features

* Training using pytorch-lightning
* Standardized calculation of diacritization evaluation metrics
* Export trained models to onnx
* Easy to use scripts for preprocessing, cleaning, tokenizing, and post-processing text and outputs
* Support for extracting sentences from any diacritized corpus

## Currently implemented models

Implementation of the following models is considered complete:

- **Sarf**: our own model that uses deep GRU network and transformer encoder layers
-  **CBHG** model from the paper [Effective Deep Learning Models for Automatic Diacritization of Arabic Text](https://ieeexplore.ieee.org/document/9274427)


## Planned models

The following models will be implemented in the near future:

- **2SDiac** from the paper [Take the Hint: Improving Arabic Diacritization with Partially-Diacritized Text](https://arxiv.org/abs/2306.03557)
- **D2/D3** models  from the paper [Deep Diacritization: Efficient Hierarchical Recurrence for Improved Arabic Diacritization](https://arxiv.org/abs/2011.00538)


# Usage

Here's how to train the **CBHG** model. The process is very similar for the other models.

## Review model config

Every command requires passing a `--config` argument. The **config** contains model hyper parameters and data paths.

For **CBHG** model this is the file **config/cbhg/config.json**.

Please review the keys and change them based on your environment and needs. For instance, if you have abundant vram, you can increase `batch_size ` or `max_len`, both of which may improve the model's predictions.


## Install packages

Make sure you have **Python 3.10** or later.

Then clone this repo:

```bash
git clone https://github.com/mush42/hareef
```

After this  cd to the repo, create a `virtualenv`, and install required packages:

```bash
cd ./hareef
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools
python3 -m pip install -r requirements.txt
```

## Prepareing the dataset

Training the models included in this repo requires a large corpus of fully diacritized Arabic text. You can download such a corpus from this [drive link](https://drive.google.com/file/d/1shhIEKc2FITVorSX26cmN9dPkKkYxI48/view?usp=sharing) and unzip it to a location of your choice.

After downloading and unzipping the corpus, run the following command from the root of the repo:

```bash
python3 -m hareef.cbhg.process_corpus --config ./config/cbhg/config.json --validate [/path/to/extracted/arabic-diacritization-corpus.txt]
```

This will create `train.txt`, `val.txt`, and `test.txt` in the `./data/cbhg/CA_MSA/` directory (or the path you configured in `config.json`)

## Training

**Lightning** is used for training. Run the following command to start the training loop.

```
python3 -m hareef.cbhg.train --config ./config/cbhg/config.json
```

By default the model will train for 100 epochs. Early stop criteria  will stop training earlier if the `loss` metric does not improve for 5 consecutive epochs.


## Evaluation

To calculate **WER/DER** metrics with and without **case-endings**, use the following command:

```bash
python3 -m hareef.cbhg.error_rates --config ./config/cbhg/config.json
``` 

## Testing

To test the model using the **test** data split, use the following command:

```bash
python3 -m hareef.cbhg.train --test --config ./config/cbhg/config.json
```


## Inference

Use the following command to diacritize some passage of Arabic text using the last checkpoint:

```bash
python -m hareef.cbhg.infer --config ./config/cbhg/config.json --text "الجو جميل، والهواء عليل."
```

If you exported the model to ONNX, you can use the ONNX model instead of torch checkpoint by passing the `--onnx` argument to the script.


## Exporting to ONNX

To export the last checkpoint to **ONNX**, use the following command:

```bash
python3 -m hareef.cbhg.export_onnx --config ./config/cbhg/config.json --output ./model.onnx
```


# License

MIT License
