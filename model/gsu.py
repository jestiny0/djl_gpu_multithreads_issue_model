import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature import combine_doc_emb_len, embedding_dims


def get_attention_net(input_shape):
    layers = []
    layers.append(nn.Linear(4 * input_shape + embedding_dims['os'] + embedding_dims['uid'] + embedding_dims['zip_code'], 40))
    layers.append(nn.PReLU())
    layers.append(nn.Linear(40, 1))
    return nn.Sequential(*layers)


class GSU(nn.Module):
    def __init__(self):
        super(GSU, self).__init__()

        self.head = 2

        self.transY = nn.Sequential(
            nn.Linear(combine_doc_emb_len, self.head * combine_doc_emb_len),
            nn.PReLU()
        )
        self.transLocalX = nn.Sequential(
            nn.Linear(combine_doc_emb_len, self.head * combine_doc_emb_len),
            nn.PReLU()
        )
        self.transNonLocalX = nn.Sequential(
            nn.Linear(combine_doc_emb_len, self.head * combine_doc_emb_len),
            nn.PReLU()
        )

        self.local_att_fc = get_attention_net(combine_doc_emb_len)
        self.nonlocal_att_fc = get_attention_net(combine_doc_emb_len)

    def get_attention_output(self, attention_net, x, y, user_id_embs, os_embs, zip_code_embs):
        # (batch, head, doc_num, doc_dim)
        x_head = x.view(-1, x.shape[1], self.head, combine_doc_emb_len).permute(0, 2, 1, 3)
        # (batch, doc_num, trans_dim)  (candidate_doc_num, doc_num, trans_dim)
        y_expanded = y.unsqueeze(1).expand(-1, x.shape[1], -1)
        # (batch, head, doc_num, doc_dim)  (candidate_doc_num, head, doc_num, doc_dim)
        y_expanded_head = y_expanded.view(-1, y_expanded.shape[1], self.head, combine_doc_emb_len).permute(0, 2, 1, 3)
        # (batch, doc_num, context_dim)
        att_side_feature = torch.cat([user_id_embs, os_embs, zip_code_embs], dim=-1).unsqueeze(1).expand(-1, x.shape[1], -1)
        # (batch, head, doc_num, context_dim)
        att_side_feature_head = att_side_feature.unsqueeze(1).expand(-1, self.head, -1, -1)
        # (batch, head, doc_num)
        p_attn = attention_net(
            torch.cat([x_head, x_head * y_expanded_head, x_head - y_expanded_head, y_expanded_head, att_side_feature_head], dim=-1)).squeeze()
        # (batch, head, doc_dim)
        attention_output_head = torch.sum(x_head * p_attn.unsqueeze(-1), dim=2).squeeze(2)
        # (batch, trans_dim)
        attention_output = attention_output_head.view(-1, self.head * combine_doc_emb_len)
        return p_attn, attention_output

    def forward(self, local_x_embs, nonlocal_x_embs, y_embs, user_id_embs, os_embs, zip_code_embs, is_debug=False, serving=False):
        # (batch, doc_num, trans_dim)
        local_x_trans = F.normalize(self.transLocalX(local_x_embs), p=2, dim=-1)
        # (batch, doc_num, trans_dim)
        nonlocal_x_trans = F.normalize(self.transNonLocalX(nonlocal_x_embs), p=2, dim=-1)
        # (batch, trans_dim)  (candidate_doc_num, trans_dim)
        y_trans = F.normalize(self.transY(y_embs), p=2, dim=-1)

        local_p_attn, local_embs = self.get_attention_output(self.local_att_fc, local_x_trans, y_trans, user_id_embs, os_embs, zip_code_embs)
        nonlocal_p_attn, nonlocal_embs = self.get_attention_output(self.nonlocal_att_fc, nonlocal_x_trans, y_trans, user_id_embs, os_embs, zip_code_embs)
        if is_debug:
            return local_p_attn, local_embs, nonlocal_p_attn, nonlocal_embs
        else:
            return local_embs, nonlocal_embs


