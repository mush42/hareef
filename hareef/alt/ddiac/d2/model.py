# coding: utf-8

import logging

import torch as T
import numpy as np

from hareef.utils import calculate_error_rates, format_error_rates_as_table, categorical_accuracy
from lightning.pytorch import LightningModule
from torch import nn, optim
from torch.nn import functional as F

from ..modules.k_lstm import K_LSTM
from ..modules.attention import Attention

_LOGGER = logging.getLogger(__package__)


class DiacritizerD2(LightningModule):

    def __init__(self, config):
        super(DiacritizerD2, self).__init__()
        self.config = config
        self.automatic_optimization = False
        self.training_step_outputs = {}
        self.val_step_outputs = {}
        self.test_step_outputs = {}

        self.max_word_len = config["train"]["max-word-len"]
        self.max_sent_len = config["train"]["max-sent-len"]
        self.char_embed_dim = config["train"]["char-embed-dim"]

        self.final_dropout_p = config["train"]["final-dropout"]
        self.sent_dropout_p = config["train"]["sent-dropout"]
        self.diac_dropout_p = config["train"]["diac-dropout"]
        self.vertical_dropout = config['train']['vertical-dropout']
        self.recurrent_dropout = config['train']['recurrent-dropout']
        self.recurrent_dropout_mode = config['train'].get('recurrent-dropout-mode', 'gal_tied')
        self.recurrent_activation = config['train'].get('recurrent-activation', 'sigmoid')

        self.sent_lstm_units = config["train"]["sent-lstm-units"]
        self.word_lstm_units = config["train"]["word-lstm-units"]
        self.decoder_units = config["train"]["decoder-units"]

        self.sent_lstm_layers = config["train"]["sent-lstm-layers"]
        self.word_lstm_layers = config["train"]["word-lstm-layers"]

        self.cell = config['train'].get('rnn-cell', 'lstm')
        self.num_layers = config["train"].get("num-layers", 2)
        self.RNN_Layer = K_LSTM

        self.batch_first = config['train'].get('batch-first', True)
        self.num_classes = 15

        word_embeddings = config.data_utils.embeddings
        vocab_size = len(config.data_utils.letter_list)
        self.build(word_embeddings, vocab_size)

    def build(self, wembs: T.Tensor, abjad_size: int):
        self.closs = F.cross_entropy
        self.bloss = F.binary_cross_entropy_with_logits

        rnn_kargs = dict(
            recurrent_dropout_mode=self.recurrent_dropout_mode,
            recurrent_activation=self.recurrent_activation,
        )

        self.sent_lstm = self.RNN_Layer(
            input_size=wembs.shape[-1],
            hidden_size=self.sent_lstm_units,
            num_layers=self.sent_lstm_layers,
            bidirectional=True,
            vertical_dropout=self.vertical_dropout,
            recurrent_dropout=self.recurrent_dropout,
            batch_first=self.batch_first,
            **rnn_kargs,
        )
        
        self.word_lstm = self.RNN_Layer(
            input_size=self.sent_lstm_units * 2 + self.char_embed_dim,
            hidden_size=self.word_lstm_units,
            num_layers=self.word_lstm_layers,
            bidirectional=True,
            vertical_dropout=self.vertical_dropout,
            recurrent_dropout=self.recurrent_dropout,
            batch_first=self.batch_first,
            return_states=True,
            **rnn_kargs,
        )

        self.char_embs = nn.Embedding(
            abjad_size,
            self.char_embed_dim,
            padding_idx=0,
        )

        self.attention = Attention(
            kind="dot",
            query_dim=self.word_lstm_units * 2,
            input_dim=self.sent_lstm_units * 2,
       )

        self.word_embs = T.tensor(wembs, dtype=T.float32)

        self.classifier = nn.Linear(self.attention.Dout + self.word_lstm_units * 2, self.num_classes)
        self.dropout = nn.Dropout(self.final_dropout_p) 

    def forward(self, sents, words, labels):
        sents = sents.to('cpu')
        words = words.to(self.device)
        labels = labels.to(self.device)

        #^ sents : [b ts]
        #^ words : [b ts tw]
        #^ labels: [b ts tw]

        word_mask = words.ne(0.).float()
        #^ word_mask: [b ts tw]

        if self.training:
            q = 1.0 - self.sent_dropout_p
            sdo = T.bernoulli(T.full(sents.shape, q))
            sents_do = sents * sdo.long()
            #^ sents_do : [b ts] ; DO(ts)
            wembs = self.word_embs[sents_do]
            #^ wembs : [b ts dw] ; DO(ts)
        else:
            wembs = self.word_embs[sents]
            #^ wembs : [b ts dw]

        sent_enc = self.sent_lstm(wembs.to(self.device))
        #^ sent_enc : [b ts dwe]

        sentword_do = sent_enc.unsqueeze(2)
        #^ sentword_do : [b ts _ dwe]

        sentword_do = self.dropout(sentword_do * word_mask.unsqueeze(-1))
        #^ sentword_do : [b ts tw dwe]

        word_index = words.view(-1, self.max_word_len)
        #^ word_index: [b*ts tw]?

        cembs = self.char_embs(word_index)
        #^ cembs : [b*ts tw dc]

        sentword_do = sentword_do.view(-1, self.max_word_len, self.sent_lstm_units * 2)
        #^ sentword_do : [b*ts tw dwe]

        char_embs = T.cat([cembs, sentword_do], dim=-1)
        #^ char_embs : [b*ts tw dcw] ; dcw = dc + dwe

        char_enc, _ = self.word_lstm(char_embs)
        #^ char_enc: [b*ts tw dce]

        char_enc_reshaped = char_enc.view(-1, self.max_sent_len, self.max_word_len, self.word_lstm_units * 2)
        # #^ char_enc: [b ts tw dce]

        omit_self_mask = (1.0 - T.eye(self.max_sent_len)).unsqueeze(0).to(self.device)
        attn_enc, attn_map = self.attention(char_enc_reshaped, sent_enc, word_mask.bool(), prejudice_mask=omit_self_mask)
        # # #^ attn_enc: [b ts tw dae]

        attn_enc = attn_enc.reshape(-1, self.max_word_len, self.attention.Dout)
        # #^ attn_enc: [b*ts tw dae]

        final_vec = T.cat([attn_enc, char_enc], dim=-1)
    
        diac_out = self.classifier(self.dropout(final_vec))
        #^ diac_out: [b*ts tw 7]

        diac_out = diac_out.view(-1, self.max_sent_len, self.max_word_len, self.num_classes)
        #^ diac_out: [b ts tw 7]

        if not self.batch_first:
            diac_out = diac_out.swapaxes(1, 0)
 
        return diac_out, attn_map

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        scheduler = self.lr_schedulers()

        opt.zero_grad()

        x, y = batch
        loss = self.predict_and_calculate_loss(x, y)
        self.training_step_outputs.setdefault("loss", []).append(loss)
        self.log("loss", loss)

        self.manual_backward(loss)

        gradient_clip_val = self.config["train"].get("gradient_clip_val")
        if gradient_clip_val:
            self.clip_gradients(
                opt, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm="norm"
            )

        opt.step()
        scheduler.step(loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        val_loss = self.predict_and_calculate_loss(x, y)
        self.val_step_outputs.setdefault("val_loss", []).append(val_loss)
        self.log("val_loss", val_loss)
        return {"val_loss": val_loss}

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        self.test_step_outputs.setdefault("test_loss", []).append(metrics["val_loss"])
        return metrics

    def configure_optimizers(self):
        optimizer_name = self.config['train'].get('optimizer', 'adam').lower()
        init_lr = self.config["train"]["lr-init"]
        weight_decay = self.config["train"]["weight-decay"]
        if optimizer_name == 'adamw':
            optimizer = T.optim.AdamW(self.parameters(), lr=init_lr, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            optimizer = T.optim.RMSprop(self.parameters(), lr=init_lr, weight_decay=weight_decay)
        else:
            optimizer = T.optim.Adam(self.parameters(), lr=init_lr, weight_decay=weight_decay)
        lr_factor = self.config["train"]["lr-factor"]
        lr_patience = self.config["train"]["lr-patience"]
        min_lr = self.config["train"]["lr-min"]
        self.scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_factor, patience=lr_patience, min_lr=min_lr)
        return [optimizer], [self.scheduler]

    def on_train_epoch_end(self):
        self._log_epoch_metrics(self.training_step_outputs)

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics(self.val_step_outputs)

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics(self.test_step_outputs)

    def _log_epoch_metrics(self, metrics):
        for name, values in metrics.items():
            epoch_metric_mean = T.stack(values).mean()
            self.log(name, epoch_metric_mean)
            values.clear()

    def predict_and_calculate_loss(self, xt, yt, mask=None):
        xt[1] = xt[1].to(self.device)
        xt[2] = xt[2].to(self.device)
        
        yt = yt.to(self.device)        
        #^ yt: [b ts tw]

        diac, _ = self(*xt)
        loss = self.closs(diac.view(-1, self.num_classes), yt.view(-1))

        return loss

    def predict(self, dataloader):
        preds = {'haraka': [], 'shadda': [], 'tanween': []}
        _LOGGER.info("> Predicting...")
        for inputs, _ in dataloader:
            inputs[0] = inputs[0].to(self.device)
            inputs[1] = inputs[1].to(self.device)
            diac, _ = self(*inputs)

            output = np.argmax(T.softmax(diac.detach(), dim=-1).cpu().numpy(), axis=-1)
            #^ [b ts tw]

            haraka, tanween, shadda = self.flat_2_3head(output)

            preds['haraka'].extend(haraka)
            preds['tanween'].extend(tanween)
            preds['shadda'].extend(shadda)
        
        return (
            np.array(preds['haraka']),
            np.array(preds["tanween"]),
            np.array(preds["shadda"]),
        )

    @staticmethod
    def flat_2_3head(output):
        haraka, tanween, shadda = [], [], []

        # 0, 1,  2, 3,  4, 5,  6, 7, 8,  9,     10,  11,   12,  13,   14
        # 0, F, FF, K, KK, D, DD, S, Sh, ShF, ShFF, ShK, ShKK, ShD, ShDD

        convert = [
            [0,0,0],
            [1,0,0],
            [1,1,0],
            [2,0,0],
            [2,1,0],
            [3,0,0],
            [3,1,0],
            [4,0,0],
            [0,0,1],
            [1,0,1],
            [1,1,1],
            [2,0,1],
            [2,1,1],
            [3,0,1],
            [3,1,1]
        ]

        b, ts, tw = output.shape

        for b_idx in range(b):
            h_s, t_s, s_s = [], [], []
            for w_idx in range(ts):
                h_w, t_w, s_w = [], [], []
                for c_idx in range(tw):
                    c = convert[int(output[b_idx, w_idx, c_idx])]
                    h_w  += [c[0]]
                    t_w += [c[1]]
                    s_w  += [c[2]]
                h_s += [h_w]
                t_s += [t_w]
                s_s += [s_w]
            
            haraka  += [h_s]
            tanween += [t_s]
            shadda  += [s_s]
                

        return haraka, tanween, shadda
