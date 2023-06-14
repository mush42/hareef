# Hareef 

This is the Pytorch implementation of the models as described in the paper 
[Effective Deep Learning Models for Automatic Diacritization of Arabic Text](https://ieeexplore.ieee.org/document/9274427).


# Creating the dataset

First download the corpus from this [drive link] (https://drive.google.com/file/d/1oHk7hxTTU4M_IWpDcbUsg7jR9kGGXrUn/view?usp=sharing) and unzip it using 7-z.

Then run the following command from the root of the repo:

```bash
python3 make_dataset.py --corpus /path/to/corpus.txt --config ./config/cbhg.yml --validate
```

This will create `train.txt`, `eval.txt`, and `test.txt` in the `./data/CA_MSA/` directory.


# Data Preprocessing

- The data can either be processed before training or when loading each batch.
- If you decide to process the corpus before training, then the corpus must have test.csv, train.csv, and valid.csv. Each file must contain three columns: text (the original text), text without diacritics, and diacritics. You have to define the column separator and diacritics separator in the config file.
- If the data is not preprocessed, you can specify that in the config.
  In that case,  each batch will be processed and the text and diacritics 
  will be extracted from the original text.
- You also have to specify the text encoder and the cleaner functions.
  This work includes two text encoders: BasicArabicEncoder, ArabicEncoderWithStartSymbol.
  Moreover, we have one cleaning function: valid_arabic_cleaners, which clean all characters except valid Arabic characters,
  which include Arabic letters, punctuations, and diacritics.

# Training

All models config are placed in the config directory.

```bash
python train.py --config config/cbhg.yml
```

The model will report the WER and DER while training using the
diacritization_evaluation package. The frequency of calculating WER and
DER can be specified in the config file.

# Testing

The testing is done in the same way as training:

```bash
python test.py --config config/cbhg.yml
```

The model will load the last saved model unless you specified it in the config:
test_data_path. If the test file name is different than test.csv, you
can add it to the config: test_file_name.

### Citation

Please cite the paper if you use this repository:

```text
M. A. H. Madhfar and A. M. Qamar, "Effective Deep Learning Models for Automatic Diacritization of Arabic Text," in IEEE Access, vol. 9, pp. 273-288, 2021, doi: 10.1109/ACCESS.2020.3041676.
```
