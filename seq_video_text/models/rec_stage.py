import math
import torch
from torch import nn, Tensor
from .FocalTransformer import FocalTransformerBlock
from .roi_seq_predictors import SequencePredictor


class DynamicConv_v2(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dim_dynamic = 64
        self.num_dynamic = 2
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ELU(inplace=True)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (rec_resolution, N * nr_boxes, self.d_model)
        '''
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)
        del parameters

        features = torch.bmm(features, param1)

        del param1
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)

        del param2

        features = self.norm2(features)
        features = self.activation(features)

        return features


class REC_STAGE(nn.Module):

    def __init__(self, args, d_model, num_classes,
                 feat_size=(28, 28), rec_batch_size=128,
                 dim_feedforward=2048, nhead=8, dropout=0.2, activation="relu"):
        super().__init__()

        self.d_model = d_model

        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv_v2(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ELU(inplace=True)

        self.feat_size = feat_size
        self.rec_batch_size = rec_batch_size
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        self.TLSAM = nn.Sequential(
            FocalTransformerBlock(dim=256, input_resolution=self.feat_size, num_heads=8, window_size=7, expand_size=0, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.2,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_method="fc", 
                 focal_level=2, focal_window=3, use_layerscale=False, layerscale_value=1e-4),
                FocalTransformerBlock(dim=256, input_resolution=self.feat_size, num_heads=8, window_size=7, expand_size=0, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.2,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_method="fc", 
                 focal_level=2, focal_window=3, use_layerscale=False, layerscale_value=1e-4),FocalTransformerBlock(dim=256, input_resolution=self.feat_size, num_heads=8, window_size=7, expand_size=0, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.2,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_method="fc", 
                 focal_level=2, focal_window=3, use_layerscale=False, layerscale_value=1e-4)
                 )

        self.pos_encoder = PositionalEncoding(self.d_model, max_len=(self.feat_size[0]//4)*(self.feat_size[1]//4))
        num_channels = d_model
        in_channels = d_model
        mode = 'nearest'
        self.k_encoder = nn.Sequential(
            encoder_layer(num_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2))
        )
        self.k_decoder_det = nn.Sequential(
            decoder_layer_worelu(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer_worelu(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, in_channels, size=(self.feat_size[0], self.feat_size[1]), mode=mode)
        )
        self.k_decoder_rec = nn.Sequential(
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
        )

        self.seq_decoder = SequencePredictor(args, d_model)
        self.rescale = nn.Upsample(size=(self.feat_size[0], self.feat_size[1]), mode="bilinear", align_corners=False)

    def forward(self, roi_features, pro_features, gt_masks, idx=None, targets=None):
        """
        :param gt_masks: # [\sum{box}, 28, 28]
        :param roi_features: [\sum{box}, c, roi_mask_size, roi_mask_size]
        :param pro_features: [bs, nf, num_querries, c]
        """
        assert roi_features.dim() == pro_features.dim() == 4, "dim error"
        features = []
        k = roi_features
        for i in range(0, len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
        n, c, h, w = k.size()
        k = k.view(n, c, -1).permute(2, 0, 1)  # [7*7, n, c]
        # self_att.
        pro_features = pro_features.reshape(-1, *pro_features.shape[-2:]).permute(1, 0, 2)  # [num_querries, bs * nf, c]
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)

        del pro_features2

        pro_features = self.norm1(pro_features)

        # inst_interact.
        if idx:
            pro_features = pro_features.permute(1, 0, 2)[idx]  # [num_insts, c]
            pro_features = pro_features.repeat(2, 1)[:self.rec_batch_size]
        else:
            pro_features = pro_features.permute(1, 0, 2)

        pro_features = pro_features.reshape(1, -1, self.d_model)  # [1, num_insts, c]
        pro_features2 = self.inst_interact(pro_features, k)
        pro_features = k.permute(1, 0, 2) + self.dropout2(pro_features2)

        del pro_features2

        obj_features = self.norm2(pro_features)  # [num_insts, k_h * k_w, c]

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        # RC Module
        del obj_features2
        obj_features = self.norm3(obj_features)
        obj_features = obj_features.permute(1, 0, 2)
        obj_features = self.pos_encoder(obj_features)
        obj_features = self.transformer_encoder(obj_features)
        obj_features = obj_features.permute(1, 2, 0)
        n, c, w = obj_features.shape
        obj_features = obj_features.view(n, c, self.feat_size[0]//4, self.feat_size[1]//4)
        obj_features = obj_features
        k = k.permute(1, 2, 0)
        k = k.view(n, c, self.feat_size[0]//4, self.feat_size[1]//4)
        k_rec = k*obj_features.sigmoid()  # k -> a3
        k_rec = self.k_decoder_rec[0](k_rec)
        k_rec = k_rec + features[0]  # features[0] -> a2

        k_det = obj_features
        k_det = self.k_decoder_det[0](k_det)
        k_det = k_det + features[0]
        k_rec = k_rec * k_det.sigmoid()

        k_rec = self.k_decoder_rec[1](k_rec) + roi_features
        k_det = self.k_decoder_det[1](k_det) + roi_features
        k_rec = k_rec * k_det.sigmoid()

        k_rec = self.k_decoder_det[-1](k_rec)
        k_rec = k_rec.flatten(-2, -1).permute(0, 2, 1)
        k_rec = self.TLSAM(k_rec)
        k_rec = k_rec.permute(0, 2, 1).view(n, c, self.feat_size[0], self.feat_size[1])
        gt_masks = self.rescale(gt_masks.unsqueeze(1))
        k_rec = k_rec * gt_masks
        attn_vecs = self.seq_decoder(k_rec, targets, targets)
        return attn_vecs


def encoder_layer(in_c, out_c, k=3, s=2, p=1):
    return nn.Sequential(nn.Conv2d(in_c, out_c, k, s, p),
                         nn.BatchNorm2d(out_c),
                         nn.ReLU(True))


def decoder_layer(in_c, out_c, k=3, s=1, p=1, mode='nearest', scale_factor=None, size=None):
    align_corners = None if mode == 'nearest' else True
    return nn.Sequential(nn.Upsample(size=size, scale_factor=scale_factor,
                                     mode=mode, align_corners=align_corners),
                         nn.Conv2d(in_c, out_c, k, s, p),
                         nn.BatchNorm2d(out_c),
                         nn.ReLU(True))


def decoder_layer_worelu(in_c, out_c, k=3, s=1, p=1, mode='nearest', scale_factor=None, size=None):
    align_corners = None if mode=='nearest' else True
    return nn.Sequential(nn.Upsample(size=size, scale_factor=scale_factor,
                                   mode=mode, align_corners=align_corners),
                         nn.Conv2d(in_c, in_c, k, s, p),
                         nn.BatchNorm2d(in_c),
                         nn.ReLU(True),
                         nn.Conv2d(in_c, out_c, k, s, p))


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
