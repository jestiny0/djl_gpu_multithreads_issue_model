from .feature import num_embeddings, embedding_dims, feature_meta_size
from .auto_dis import AutoDis

import logging

import torch
import torch.nn as nn


def reduce_mean(embs, idx, dim=-2):
    return torch.mean(embs(idx), dim=dim)


def reduce_sum(embs, ind, dim=-2):
    return torch.sum(embs(ind), dim=dim)


def init_autodis(feature_type):
    return AutoDis(meta_embedding_vocab_size=feature_meta_size[feature_type], dims=embedding_dims[feature_type])


def init_embedding(feature_type, pre_trained=None):
    if pre_trained is None:
        return nn.Embedding(num_embeddings[feature_type], embedding_dims[feature_type], 0)  # , sparse=True)
    else:
        ret = nn.Embedding.from_pretrained(pre_trained, freeze=False, padding_idx=0)  # , sparse=True)
        ret._fill_padding_idx_with_zero()
        return ret


class Ps(nn.Module):
    def __init__(self):
        super(Ps, self).__init__()

        # SIDE INFO EMBEDDING
        self.domain_embs = init_embedding('domain')
        self.cate_embs = init_embedding('cate')
        self.channel_embs = init_embedding('chn')
        self.seg_title_embs = init_embedding('seg_title')
        self.os_embs = init_embedding('os')
        self.doc_embs = init_embedding('doc')
        self.uid_embs = init_embedding('uid')
        self.zip_code_embs = init_embedding('zip_code')
        self.pid_embs = init_embedding('pid')

        logging.info(
            'doc_embs shape: {}, uid_embs shape: {}'.format(self.doc_embs.weight.shape, self.uid_embs.weight.shape))

    def get_x(self, batch, prefix):
        x_doc_embs = self.doc_embs(batch[f'{prefix}_xs'])
        x_domain_embs = self.domain_embs(batch[f'{prefix}_x_domain'])
        x_cate_embs = reduce_sum(self.cate_embs, batch[f'{prefix}_x_cate'])
        x_chn_embs = reduce_sum(self.channel_embs, batch[f'{prefix}_x_chn'])
        x_seg_title_embs = reduce_sum(self.seg_title_embs, batch[f'{prefix}_x_seg_title'])
        x_pid_embs = reduce_mean(self.pid_embs, batch[f'{prefix}_x_pid'])
        x_embs = torch.cat(
            [x_doc_embs, x_domain_embs, x_cate_embs, x_chn_embs, x_seg_title_embs, batch[f'{prefix}_x_semantic_emb'],
             x_pid_embs], dim=-1)
        return x_embs

    def get_y(self, batch):
        y_doc_embs = self.doc_embs(batch[f'ys'])
        y_domain_embs = self.domain_embs(batch[f'y_domain'])
        y_cate_embs = reduce_sum(self.cate_embs, batch[f'y_cate'])
        y_chn_embs = reduce_sum(self.channel_embs, batch[f'y_chn'])
        y_seg_title_embs = reduce_sum(self.seg_title_embs, batch[f'y_seg_title'])
        y_pid_embs = reduce_sum(self.pid_embs, batch[f'y_pid'])
        y_embs = torch.cat(
            [y_doc_embs, y_domain_embs, y_cate_embs, y_chn_embs, y_seg_title_embs, batch['y_semantic_emb'], y_pid_embs],
            dim=-1)
        return y_embs

    def forward(self, batch):
        local_x = self.get_x(batch, 'local')
        nonlocal_x = self.get_x(batch, 'nonlocal')
        y = self.get_y(batch)
        # print max and min value of batch['zip_code']
        zipcode = torch.clip(batch['zip_code'], 0, num_embeddings['zip_code'] - 1)
        d = {
            'local_x': local_x,
            'nonlocal_x': nonlocal_x,
            'y': y,
            'os': self.os_embs(batch['os']),
            'uid': self.uid_embs(batch['hashed_uid']),
            'zip_code': self.zip_code_embs(zipcode),
        }
        return d
