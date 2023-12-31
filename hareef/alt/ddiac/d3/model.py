# coding: utf-8

import logging

import numpy as np
import torch as T

from lightning.pytorch import LightningModule
from torch import nn, optim
from torch.nn import functional as F

from ..modules.k_lstm import K_LSTM
from ..modules.attention import Attention
from ..modules.linear_scheduler import LinearSchedule

_LOGGER = logging.getLogger("hareef.process_corpus")


class DiacritizerD3(LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.automatic_optimization = False
        self.training_step_outputs = {}
        self.val_step_outputs = {}
        self.test_step_outputs = {}

        self.max_word_len = config["train"]["max-word-len"]
        self.max_sent_len = config["train"]["max-sent-len"]
        self.char_embed_dim = config["train"]["char-embed-dim"]

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

        self.baseline = config["train"].get("baseline", False)

        self.anneal_ddo = config["train"]["anneal-ddo"]
        self.ddo_scheduler = LinearSchedule(config["train"]["anneal-ddo-range"], 0, 1) 

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

        self.lstm_decoder = self.RNN_Layer(
            input_size=self.word_lstm_units * 2 + self.attention.Dout + 8,
            hidden_size=self.word_lstm_units * 2,
            num_layers=1,
            bidirectional=False,
            vertical_dropout=self.vertical_dropout,
            recurrent_dropout=self.recurrent_dropout,
            batch_first=self.batch_first,
            return_states=True,
            **rnn_kargs,
        )
 
        self.word_embs = T.tensor(wembs, dtype=T.float32)

        self.classifier = nn.Linear(self.lstm_decoder.hidden_size, 15)
        self.dropout = nn.Dropout(0.2)

    def forward(self, sents, words, labels):
        sents = sents.to('cpu')
        words = words.to(self.device)
        labels = labels.to(self.device)

        #^ sents : [b ts]
        #^ words : [b ts tw]
        #^ labels: [b ts tw]

        word_mask = words.ne(0.).float()
        #^ word_mask: [b ts tw 1]

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
        #^ char_enc: [b ts tw dce]

        omit_self_mask = (1.0 - T.eye(self.max_sent_len)).unsqueeze(0).to(self.device)
        attn_enc, attn_map = self.attention(char_enc_reshaped, sent_enc, word_mask.bool(), prejudice_mask=omit_self_mask)
        #^ attn_enc: [b ts tw dae]

        attn_enc = attn_enc.view(-1, self.max_sent_len*self.max_word_len, self.attention.Dout)
        #^ attn_enc: [b*ts tw dae]

        if self.training and self.diac_dropout_p > 0:
            q = 1.0 - self.diac_dropout_p
            ddo = T.bernoulli(T.full(labels.shape[:-1], q))
            labels = labels * ddo.unsqueeze(-1).long().to(self.device)
            #^ labels : [b ts tw] ; DO(ts)
                         
        labels = labels.view(-1, self.max_sent_len*self.max_word_len, 8).float()
        #^ labels: [b*ts tw 8]
        
        char_enc = char_enc.view(-1, self.max_sent_len*self.max_word_len, self.word_lstm_units * 2)

        final_vec = T.cat([attn_enc, char_enc, labels], dim=-1)
        #^ final_vec: [b ts*tw dae+8]

        dec_out, _ = self.lstm_decoder(final_vec)
        #^ dec_out: [b*ts tw du]

        dec_out = dec_out.reshape(-1, self.max_word_len, self.lstm_decoder.hidden_size)
    
        diac_out = self.classifier(self.dropout(dec_out))
        #^ diac_out: [b*ts tw 7]

        diac_out = diac_out.view(-1, self.max_sent_len, self.max_word_len, 15)
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

    def on_fit_start(self) -> None:
        self._start_epoch = self.current_epoch
    
    def on_test_epoch_start(self) -> None:
        if self.anneal_ddo:
            self.diac_dropout_p = self.ddo_scheduler.value(self.current_epoch + 1 - self._start_epoch)

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
        #^ yt: [b ts tw]
        yt = yt.to(self.device)        

        if self.training:
            diac, _ = self(*xt)
        else:
            diac = self._predict_step(*xt)
        #^ diac[0] : [b ts tw 5]

        loss = self.closs(diac.view(-1,15), yt.view(-1))
        return loss

    def predict(self, dataloader):
        preds = {'haraka': [], 'shadda': [], 'tanween': []}
        _LOGGER.info("> Predicting...")
        for inputs, _ in dataloader:
            inputs[1] = inputs[1].to(self.device)
            inputs[2] = inputs[2].to(self.device)
            diac = self._predict_step(*inputs)
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

    def _predict_step(self, sents, words, labels):
        sents = sents.to('cpu')
        words = words.to(self.device)
        labels = labels.to(self.device)

        word_mask = words.ne(0.).float()
        #^ mask: [b ts tw 1]

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
        #^ word_states: ([b*ts dce], [b*ts dce])

        char_enc = char_enc.view(-1, self.max_sent_len, self.max_word_len, self.word_lstm_units*2)
        #^ char_enc: [b ts tw dce]

        omit_self_mask = (1.0 - T.eye(self.max_sent_len)).unsqueeze(0).to(self.device)
        attn_enc, _ = self.attention(char_enc, sent_enc, word_mask.bool(), prejudice_mask=omit_self_mask)
        #^ attn_enc: [b ts tw dae]

        all_out = T.zeros(*char_enc.size()[:-1], 15).to(self.device)
        #^ all_out: [b ts tw 7]

        batch_sz = char_enc.size()[0]
        #^ batch_sz: b

        zeros = T.zeros(1, batch_sz, self.lstm_decoder.hidden_size).to(self.device)
        #^ zeros: [1 b du]

        bos_tag = T.tensor([0,0,0,0,0,1,0,0]).unsqueeze(0)
        #^ bos_tag: [1 8]

        prev_label = T.cat([bos_tag]*batch_sz).to(self.device).float()
        # bos_vec = T.cat([bos_tag]*batch_sz).to(self.device).float()
        #^ prev_label: [b 8]

        for ts in range(self.max_sent_len):
            dec_hx = (zeros, zeros)
            #^ dec_hx: [1 b du]
            for tw in range(self.max_word_len):     
                final_vec = T.cat([attn_enc[:,ts,tw,:], char_enc[:,ts,tw,:], prev_label], dim=-1).unsqueeze(1)
                #^ final_vec: [b 1 dce+8]
                dec_out, dec_hx = self.lstm_decoder(final_vec, dec_hx)
                #^ dec_out: [b 1 du]
                dec_out = dec_out.squeeze(0)
                dec_out = dec_out.transpose(0,1)

                logits_raw = self.classifier(self.dropout(dec_out))
                #^ logits_raw: [b 1 15]

                out_idx = T.max(T.softmax(logits_raw.squeeze(), dim=-1), dim=-1)[1]

                haraka, tanween, shadda = self.flat2_3head(out_idx.detach().cpu().numpy())

                haraka_onehot = T.eye(6)[haraka].float().to(self.device)
                #^ haraka_onehot+bos_tag: [b 6]

                tanween = T.tensor(tanween).float().unsqueeze(-1).to(self.device)
                shadda = T.tensor(shadda).float().unsqueeze(-1).to(self.device)

                prev_label = T.cat([haraka_onehot, tanween, shadda], dim=-1)

                all_out[:,ts,tw,:] = logits_raw.squeeze()

        if not self.batch_first:
            all_out = all_out.swapaxes(1, 0)
 
        return all_out 

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

    @staticmethod
    def flat2_3head(diac_idx):
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


        for diac in diac_idx:
            c_out = convert[diac]
            haraka += [c_out[0]]
            tanween += [c_out[1]]
            shadda += [c_out[2]]

        return np.array(haraka), np.array(tanween), np.array(shadda) 
