import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.autograd import Variable
import torch.nn.functional as F
import math
from ScheduleOptim import ScheduledOptim
from data_functions import from_batch_get_model_input, from_test_batch_get_model_input
from Vocab import Vocab
import torch.optim as Optim

'''===============define embedding layers ======================'''

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size, pad_idx=None):
        super(Embeddings, self).__init__()
        self.word_embdding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.d_model = d_model

    def forward(self, x):
        return self.word_embdding(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=300):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:,:x.size(1)], requires_grad=False)
        return x

'''===============define Multi-head Attention =================='''

def ScaleDotAttention(query, key, value, mask=None, dropout=None):

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        Dropout = nn.Dropout(dropout)
        p_attn = Dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

        self.linear_q = nn.Linear(d_model, self.n_heads*self.d_k, bias=False)
        self.linear_k = nn.Linear(d_model, self.n_heads*self.d_k, bias=False)
        self.linear_v = nn.Linear(d_model, self.n_heads*self.d_v, bias=False)
        self.linear_fc = nn.Linear(self.n_heads*self.d_v, d_model, bias=False)

        self.attn = None
        self.dropout = dropout
        self.drpo = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size, len_q, len_k, len_v = query.size(0), query.size(1), key.size(1), value.size(1)
        d_k, d_v, n_heads = self.d_k, self.d_v, self.n_heads

        residual = query

        # step 1
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.linear_q(query).view(batch_size, len_q, n_heads, d_k)
        k = self.linear_k(key).view(batch_size, len_k, n_heads, d_k)
        v = self.linear_v(value).view(batch_size, len_v, n_heads, d_v)

        # step 2
        # Transpose for attention dot product: b x n x lq x dv
        q, k ,v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, attn = ScaleDotAttention(q, k, v, mask=mask, dropout=self.dropout)

        # step 3
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        q = self.drpo(self.linear_fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

'''===============define Positional FFN ========================'''

class PositionWiseFeedForward(nn.Module):
    '''
    Implements FFN equation
    '''
    def __init__(self, d_model, d_linear, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_linear)
        self.w_2 = nn.Linear(d_linear, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.layer_norm(self.w_2(self.dropout(F.relu(self.w_1(x)))))

'''===============define Positional Layers ====================='''

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_linear, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_linear, dropout)

    def forward(self, encoder_input, slf_attn_mask=None):
        encoder_output, enc_slf_attn = self.slf_attn(encoder_input, encoder_input, encoder_input, mask=slf_attn_mask)
        encoder_output = self.ffn(encoder_output)
        return encoder_output, enc_slf_attn

class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_linear, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout=dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_linear, dropout=dropout)

    def forward(self, decoder_input, encoder_output,
                slf_attn_mask=None, dec_enc_attn_mask=None):
        decoder_output, dec_slf_attn = self.self_attn(decoder_input, decoder_input, decoder_input, mask=slf_attn_mask)

        decoder_status = decoder_output
        decoder_output, dec_enc_attn = self.enc_attn(decoder_output, encoder_output, encoder_output, mask=dec_enc_attn_mask)

        decoder_context = decoder_output
        decoder_output = self.ffn(decoder_output)
        return decoder_output, dec_slf_attn, dec_enc_attn, decoder_status, decoder_context

'''===============define mask functions ========================'''

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

'''===============define model modules ========================='''

class Encoder(nn.Module):
    def __init__(self, vocab_size, n_layers,
                 n_heads, d_model, d_linear, d_k, d_v,
                 pad_idx, dropout=0.1, max_len=5000):
        super(Encoder, self).__init__()
        self.src_word_emb = Embeddings(d_model, vocab_size, pad_idx=pad_idx)
        self.src_posi_emb = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(n_heads, d_model, d_linear, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_mask, return_attns=True):
        enc_slf_attn_list = []

        encoder_output = self.dropout(self.src_posi_emb(self.src_word_emb(src_seq)))
        encoder_output = self.layer_norm(encoder_output)

        for enc_layer in self.layer_stack:
            encoder_output, enc_slf_attn = enc_layer(encoder_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        #print(encoder_output.size())

        if return_attns:
            return encoder_output, enc_slf_attn_list
        else:
            return encoder_output

class Decoder(nn.Module):
    def __init__(self, vocab_size, n_layers,
                 n_heads, d_model, d_linear, d_k, d_v,
                 pad_idx, dropout=0.1, max_len=5000):
        super(Decoder, self).__init__()
        self.tgt_word_emb = Embeddings(d_model, vocab_size, pad_idx=pad_idx)

        self.tgt_posi_emb = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(n_heads, d_model, d_linear, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, tgt_seq, tgt_mask, enc_output, src_mask, return_attn=True):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        decoder_output = self.dropout(self.tgt_posi_emb(self.tgt_word_emb(tgt_seq)))
        decoder_output = self.layer_norm(decoder_output)

        for dec_layer in self.layer_stack:
            decoder_output, dec_self_attn, dec_enc_attn, decoder_status, deocder_context = dec_layer(
                decoder_output, enc_output, slf_attn_mask=tgt_mask, dec_enc_attn_mask=src_mask
            )
            dec_slf_attn_list += [dec_self_attn] if return_attn else []
            dec_enc_attn_list += [dec_enc_attn] if return_attn else []

        if return_attn:
            return decoder_output, dec_slf_attn_list, dec_enc_attn_list, decoder_status, deocder_context
        else:
            return decoder_output


class GeneraProb(nn.Module):
    def __init__(self, d_model, dropout):
        super(GeneraProb, self).__init__()

        self.w_h = nn.Linear(d_model, 1)
        self.w_s = nn.Linear(d_model, 1)

        self.dropout = nn.Dropout(p=dropout)

    # h : weight sum of encoder output ,(batch,hidden*2)
    # s : decoder state                 (batch,hidden*2)
    # x : decoder input                 (batch,embed)
    def forward(self, h, s):
        h_feature = self.dropout(self.w_h(h))  # (batch,1)
        s_feature = self.dropout(self.w_s(s))  # (batch,1)

        gen_feature = h_feature + s_feature # (batch,1)

        gen_p = torch.sigmoid(gen_feature)

        return gen_p

'''===============define Transformer ==========================='''

class Transformer(pl.LightningModule):
    def __init__(self, opt):
        super(Transformer, self).__init__()
        self.all_mode = ["train", "valid", "decode"]
        self.vocab_size = opt.vocab_size
        self.d_model = opt.d_model
        self.d_linear = opt.d_linear
        self.n_layers = opt.n_layers
        self.n_heads = opt.n_heads
        self.d_k = self.d_model // self.n_heads
        self.d_v = self.d_model // self.n_heads
        self.dropout = opt.dropout
        self.max_encoder_len = opt.max_article_len
        self.max_decoder_len = opt.max_title_len
        self.tgt_emb_proj_weight_sharing = opt.tgt_emb_proj_weight_sharing
        self.emb_src_tgt_weight_sharing = opt.emb_src_tgt_weight_sharing

        self.use_pointer = opt.use_pointer
        self.use_coverage = opt.use_coverage
        self.smoothing = opt.smoothing

        self.init_lr = opt.init_lr
        self.n_warmup_steps = opt.n_warmup_steps

        self.coverage_loss_weight = opt.coverage_loss_weight

        self.pad_idx = opt.pad_idx


        self.encoder = Encoder(
            vocab_size=self.vocab_size,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_model=self.d_model,
            d_linear=self.d_linear,
            d_k=self.d_k,
            d_v=self.d_v,
            pad_idx=self.pad_idx,
            dropout=self.dropout,
            max_len=self.max_encoder_len
        )

        self.decoder = Decoder(
            vocab_size=self.vocab_size,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_model=self.d_model,
            d_linear=self.d_linear,
            d_k=self.d_k,
            d_v=self.d_v,
            pad_idx=self.pad_idx,
            dropout=self.dropout,
            max_len=self.max_decoder_len
        )

        self.GeneraProb = GeneraProb(
            d_model=self.d_model,
            dropout=self.dropout
        )

        self.tgt_word_proj = nn.Linear(self.d_model, self.vocab_size, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.x_logit_scale = 1.
        if self.tgt_emb_proj_weight_sharing:
            self.tgt_word_proj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (self.d_model ** -0.5)

        if self.emb_src_tgt_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight

    def forward(self, encoder_input, encoder_mask, encoder_with_oov,
                oovs_zero, context_vec, coverage,
                decoder_input=None, decoder_mask=None, decoder_target=None,
                mode="train", start_tensor=None, beam_size=4):
        assert mode in self.all_mode
        if mode in ['train', 'valid']:
            model_loss = self._forward(encoder_input, encoder_with_oov, oovs_zero,
                                       decoder_input, decoder_target)

            return model_loss

        elif mode in ['decode']:
            return self._decoder()

    def _forward(self, encoder_input, encoder_with_oov, oovs_zero,
                 decoder_input, decoder_target):
        encoder_mask = get_pad_mask(encoder_input, self.pad_idx)
        # print('src_mask:', src_mask.size())
        decoder_mask = get_pad_mask(decoder_input, self.pad_idx) & get_subsequent_mask(decoder_input)
        # print('tgt_mask:', tgt_mask.size())

        encoder_output, enc_slf_attn_list = self.encoder(encoder_input, encoder_mask)
        # print('encoder:', enc_output.size())
        # decoder_output.size = [B, S, d_model]
        # decoder_status.size = [B, S, d_model]
        # decoder_context.size = [B, S, d_model]
        decoder_output, dec_slf_attn_list, dec_enc_attn_list, decoder_status, decoder_context \
            = self.decoder(decoder_input, decoder_mask, encoder_output, encoder_mask)
        #self.tgt_word_proj = nn.Linear(self.d_model, self.vocab_size, bias=False)
        # [bz, sl, d_model]
        # ---> [bz, tgt_sql, vocab_size]
        gen_logit = self.tgt_word_proj(decoder_output) * self.x_logit_scale
        if self.use_pointer:
            # [bz, tgt_sql, 1]
            gen_prob = self.GeneraProb(decoder_context, decoder_status)
            copy_prob = (1 - gen_prob)
            gen_logit = gen_logit.mul(gen_prob)
            # cal attention_score for copy_dist
            # [B,n,tgt_sql,src_sql] ---> [B,tgt_sql,src_sql]
            copy_dist = dec_enc_attn_list[-1].sum(dim=1)
            copy_logit = copy_dist.mul(copy_prob)
            if oovs_zero is not None:
                tgt_sql = gen_prob.size(1)
                encoder_with_oov = encoder_with_oov.unsqueeze(1).repeat(1, tgt_sql, 1)
                oovs_zero = oovs_zero.unsqueeze(1).repeat(1, tgt_sql, 1)
                gen_logit = torch.cat([gen_logit, oovs_zero], dim=-1)
            final_logit = gen_logit.scatter_add(dim=-1, index=encoder_with_oov, src=copy_logit)
        else:
            final_logit = gen_logit
        # [bz*sql, vocab]
        final_logit = final_logit.view(-1, gen_logit.size(2))
        #print(final_logit.size())
        #print(decoder_target.size())
        loss, n_correct, n_word = cal_performance(final_logit, decoder_target, self.pad_idx, smoothing=self.smoothing)
        return loss
    def _decoder(self):
        exit()

    def configure_optimizers(self):
        optimizer = Optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.98))
        # optimizer = ScheduledOptim(optimizer, self.init_lr, self.d_model, self.n_warmup_steps,
        #                            n_steps = 0)
        return optimizer

    def training_step(self, batch, batch_idx):
        batch = from_batch_get_model_input(batch, self.d_model,
                                           use_pointer=self.use_pointer, use_coverage=self.use_coverage)
        batch[0] = patch_src(batch[0])
        batch[6], batch[8] = patch_trg(batch[6])

        inputs = {'encoder_input': batch[0],
                  'encoder_mask': batch[1],
                  'encoder_with_oov': batch[2],
                  'oovs_zero': batch[3],
                  'context_vec': batch[4],
                  'coverage': batch[5],
                  'decoder_input': batch[6],
                  'decoder_mask': batch[7],
                  'decoder_target': batch[8],
                  'mode': 'train'}


        loss = self(**inputs)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss, prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx):
        batch = from_batch_get_model_input(batch, self.d_model,
                                           use_pointer=self.use_pointer, use_coverage=self.use_coverage)

        batch[0] = patch_src(batch[0])
        batch[6], batch[8] = patch_trg(batch[6])


        inputs = {'encoder_input': batch[0],
                'encoder_mask': batch[1],
                'encoder_with_oov': batch[2],
                'oovs_zero': batch[3],
                'context_vec': batch[4],
                'coverage': batch[5],
                'decoder_input': batch[6],
                'decoder_mask': batch[7],
                'decoder_target': batch[8],
                'mode': 'train'}


        loss = self(**inputs)
        #print(loss)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, prog_bar=True)
        return result

'''===============define extra functions ========================'''

def patch_src(src):
    # src = src.transpose(0, 1)
    return src

def patch_trg(trg):
    # trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold

def cal_performance(pred, gold, tgt_pad_idx, smoothing=False):

    loss = cal_loss(pred, gold, tgt_pad_idx, smoothing=smoothing)
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(tgt_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word

def cal_loss(pred, gold, tgt_pad_idx, smoothing=False):
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(tgt_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        #print(pred.size())
        #print(gold.size())
        loss = F.cross_entropy(pred, gold, ignore_index=tgt_pad_idx, reduction='sum')
    return loss
