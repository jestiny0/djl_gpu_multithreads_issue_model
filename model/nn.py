
import torch
import torch.nn as nn
from . import dnn, ps
from .rmsnorm import RMSNorm
from .swi_glu import SwiGLU
from collections import namedtuple
from .feature import embedding_dims, combine_doc_emb_len, user_context_feature_emb_len, num_embeddings, cfb_feature_emb_len

AttentionWeights = namedtuple('AttnWeights', ['local_attn_weights', 'nonlocal_attn_weights'])


def get_attention_net(input_shape, context_emb_len):
    layers = []
    layers.append(nn.Linear(4 * input_shape + context_emb_len, 40 * 2))
    layers.append(SwiGLU())
    layers.append(nn.Linear(40, 1))
    return nn.Sequential(*layers)


def get_attention_output(attention_net, x_rmsnorm_net, y_rmsnorm_net, x, y, rdid_mask, context_emb):
    x = x_rmsnorm_net(x)
    y = y_rmsnorm_net(y)
    y_expanded = y.unsqueeze(1).expand(-1, x.shape[1], -1)
    att_side_feature = context_emb.unsqueeze(1).expand(-1, x.shape[1], -1)
    input = torch.cat([x, x*y_expanded, x-y_expanded, y_expanded, att_side_feature], dim=-1)
    p_attn = attention_net(input).squeeze(-1)
    p_attn = p_attn.masked_fill(~rdid_mask.to(bool), 0.)
    attention_output = torch.bmm(p_attn.unsqueeze(1), x).squeeze(1)
    return p_attn, attention_output


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.level_score_autodis = ps.init_autodis('level_score')
        self.on_boarding_days_embs = ps.init_embedding('on_boarding_days')
        self.county_fibs_embs = ps.init_embedding('county_fibs')
        self.state_code_embs = ps.init_embedding('state_code')
        self.city_id_embs = ps.init_embedding('city_id')
        self.dma_code_embs = ps.init_embedding('dma_code')
        self.check_embs = ps.init_embedding('check')
        self.click_embs = ps.init_embedding('click')
        self.ctr_autodis = ps.init_autodis('ctr')
        self.avg_level_score_autodis = ps.init_autodis('avg_level_score')

        self.dnn_fc = dnn.DNN([
            3 * combine_doc_emb_len + embedding_dims['os'] + embedding_dims['uid'] + embedding_dims['zip_code'] + user_context_feature_emb_len + cfb_feature_emb_len, 512, 256, 64,
        ])
        self.local_att_fc = get_attention_net(combine_doc_emb_len, embedding_dims['os'] + embedding_dims['uid'] + embedding_dims['zip_code'])
        self.local_x_rmsnorm = RMSNorm(d=combine_doc_emb_len)
        self.nonlocal_att_fc = get_attention_net(combine_doc_emb_len, embedding_dims['os'] + embedding_dims['uid'] + embedding_dims['zip_code'])
        self.nonlocal_x_rmsnorm = RMSNorm(d=combine_doc_emb_len)
        self.y_rmsnorm = RMSNorm(d=combine_doc_emb_len)
        self.ps = None

    def forward(self, batch):
        concated_embs = self.ps(batch)
        local_x_embs, nonlocal_x_embs, y_embs, os_embs, user_id_embs, zip_code_embs = [concated_embs.get(v) for v in [
                'local_x',
                'nonlocal_x',
                'y',
                'os',
                'uid',
                'zip_code'
            ]
        ]
        bs = y_embs.shape[0]

        user_context_embs = torch.cat([
            self.on_boarding_days_embs(torch.clamp(batch['on_boarding_days'], min=0, max=num_embeddings['on_boarding_days'] - 1)),
            self.county_fibs_embs(batch['county_fibs']),
            self.state_code_embs(batch['state_code']),
            self.city_id_embs(batch['city_id']),
            self.dma_code_embs(batch['dma_code'])
        ], dim=-1)

        user_context_embs = user_context_embs.expand(bs, -1)
        user_context_embs = torch.cat([
            self.level_score_autodis(batch['level_score']),
            user_context_embs
        ], dim=-1)

        cfb_embs = torch.cat([
            self.check_embs(torch.clamp(batch['check'], min=0, max=num_embeddings['check'] - 1)),
            self.click_embs(torch.clamp(batch['click'], min=0, max=num_embeddings['click'] - 1)),
            self.ctr_autodis(batch['ctr']),
            self.avg_level_score_autodis(batch['avg_level_score']),
        ], dim=-1)

        user_id_embs = user_id_embs.expand(bs, -1)
        os_embs = os_embs.expand(bs, -1)
        zip_code_embs = zip_code_embs.expand(bs, -1)
        local_x_embs = local_x_embs.expand(y_embs.shape[0], -1, -1)
        nonlocal_x_embs = nonlocal_x_embs.expand(y_embs.shape[0], -1, -1)
        

        context_embs = torch.cat([user_id_embs, os_embs, zip_code_embs], dim=-1)

        local_p_attn, local_embs = get_attention_output(self.local_att_fc, self.local_x_rmsnorm, self.y_rmsnorm, local_x_embs, y_embs, batch["local_rdid_mask"], context_embs)
        nonlocal_p_attn, nonlocal_embs = get_attention_output(self.nonlocal_att_fc, self.nonlocal_x_rmsnorm, self.y_rmsnorm, nonlocal_x_embs, y_embs, batch["nonlocal_rdid_mask"], context_embs)

        out = self.dnn_fc(torch.cat([local_embs, nonlocal_embs, y_embs, user_id_embs, os_embs, zip_code_embs, user_context_embs, cfb_embs], dim=-1))

        sims = out.squeeze()

        return sims