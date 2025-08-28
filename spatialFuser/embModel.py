import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphAttention(Module):
    """
    Graph attention layer
    :param args: Global hyperparameters setting
    :param layer: layer index
    """
    def __init__(self, args, layer):
        super(GraphAttention, self).__init__()
        self.args = args

        if self.args.heads > 1 :
            self.V = torch.nn.init.xavier_normal_(
                torch.empty(self.args.heads,
                            2,
                            int(self.args.hidden[layer] / self.args.heads),
                            1),
                gain=1.414)
        else:
            self.V = torch.nn.init.xavier_normal_(
                torch.empty(2,
                            self.args.hidden[layer],
                            1),
                gain=1.414)
        self.V = Parameter(self.V, requires_grad=True)

        attn_dropout = 0
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, local_patten, long_range_patten, weighted_X):
        if self.args.heads > 1 :
            f1 = torch.matmul(weighted_X, self.V[:, 0, :, :])
            f2 = torch.matmul(weighted_X, self.V[:, 1, :, :])
            # construct node_num * node_num att matrix for each head
            ori_att = f1.repeat(1, 1, weighted_X.shape[1]) + f2.repeat(1, 1, weighted_X.shape[1]).permute(0, 2, 1)
            # Transforming any value into the range of (0,1)
            sigmoid_att = torch.sigmoid(ori_att)
            # softmax
            tau = 1
            local_att = torch.sparse.softmax((local_patten.to_dense() * sigmoid_att / tau).to_sparse_coo(), dim=-1)
            if long_range_patten == None:
                return local_att
            else:
                long_range_att = torch.sparse.softmax((long_range_patten.to_dense() * sigmoid_att / tau).to_sparse_coo(),dim=-1)
                att = self.args.alpha * long_range_att + (1 - self.args.alpha) * local_att
                # att = self.dropout(att)
                return att
        else:
            f1 = torch.matmul(weighted_X, self.V[0, :, :])
            f2 = torch.matmul(weighted_X, self.V[1, :, :])
            ori_att = f1.repeat(1, weighted_X.shape[0]) + f2.repeat(1, weighted_X.shape[0]).permute(0, 1)
            sigmoid_att = torch.sigmoid(ori_att)
            tau = 1
            local_att = torch.sparse.softmax( (local_patten.to_dense() * sigmoid_att / tau).to_sparse_coo(), dim=-1)
            if long_range_patten == None:
                return local_att
            else:
                long_range_att = torch.sparse.softmax( (long_range_patten.to_dense() * sigmoid_att / tau).to_sparse_coo(),dim=-1)
                att = self.args.alpha * long_range_att + (1 - self.args.alpha) * local_att
                # att = self.dropout(att)
                return att



class MultiHeadGraphAttentionLayer(Module):
    """
    Multi-head graph attention layer
    :param args: Global hyperparameters setting
    :param layer: layer index
    """
    def __init__(self, args, layer):
        super(MultiHeadGraphAttentionLayer, self).__init__()
        self.args = args
        self.layer = layer

        if self.args.heads > 1 :
            self.W = torch.nn.init.xavier_normal_(
                torch.empty(self.args.heads,
                            int(self.args.hidden[self.layer] / self.args.heads),
                            int(self.args.hidden[self.layer + 1] / self.args.heads)),
                gain=1.414)
        else:
            # expand is unsupported for Sparse tensors
            self.W = torch.nn.init.xavier_normal_(
                torch.empty(self.args.hidden[self.layer],
                            self.args.hidden[self.layer + 1]),
                gain=1.414)

        self.W = Parameter(self.W, requires_grad=True)

        # graph attention layer for each heads
        self.graph_att_layer = GraphAttention(self.args, self.layer + 1)

        self.dropout = nn.Dropout(self.args.dropout)

        self.norm = nn.BatchNorm1d(self.args.hidden[self.layer + 1], eps=1e-6)

    def forward(self, local_patten, long_range_patten, X):
        X = self.dropout(X)
        if self.args.heads > 1 :
            # split into multi-head
            heads_num, nodes_num, feature_num = self.args.heads, X.shape[0], self.args.hidden[self.layer]
            assert feature_num % heads_num == 0, 'feature_num must be a multiple of heads_num'

            X = X.transpose(0, 1).contiguous().view(heads_num, int(feature_num / heads_num),
                                                              nodes_num).permute(0, 2, 1) \
            # linear transform
            X = torch.matmul(X, self.W)

            # compute att value of each head
            self.att = self.graph_att_layer.forward(local_patten, long_range_patten, X).to_dense()

            # apply attention mechanism
            X = torch.matmul(self.att, X)

            # concat multi-Head result
            concated_X = torch.concat([X[x, :, :] for x in range(self.args.heads)], dim=-1)
            return self.norm(concated_X)

        else:
            # linear transform
            X = torch.matmul(X, self.W)
            # compute att value
            self.att = self.graph_att_layer.forward(local_patten, long_range_patten, X).to_dense()
            # apply attention mechanism
            X = torch.matmul(self.att, X)
            return self.norm(X)


class PositionwiseFeedForward(nn.Module):
    """
    A two-layer feed-forward module
    :param args: Global hyperparameters setting
    :param layer: layer index
    :param mod:
    """
    def __init__(self, args, layer, mod):
        super().__init__()
        self.args = args
        self.layer = layer
        if mod == 'encoder':
            self.MLP = nn.Linear(self.args.hidden[self.layer + 1],
                                 int(self.args.mlp_hidden_times * self.args.hidden[self.layer + 1]))  # position-wise

            self.norm = nn.BatchNorm1d(self.args.hidden[self.layer + 1], eps=1e-6)
        elif mod == 'decoder':
            self.MLP = nn.Linear(self.args.hidden[self.layer],
                                 int(self.args.mlp_hidden_times * self.args.hidden[self.layer]))  # position-wise

            self.norm = nn.BatchNorm1d(self.args.hidden[self.layer], eps=1e-6)
        self.actv = torch.nn.ELU()

    def forward(self, x):
        residual = x
        x = self.MLP(x)
        x = self.actv(x)
        x = x + residual
        x = self.norm(x)
        return x


class EncoderLayer(Module):
    """
    Encoder layer
    :param args: Global hyperparameters setting
    :param layer: layer index
    """
    def __init__(self, args, layer):
        super(EncoderLayer, self).__init__()
        self.args = args
        self.layer = layer
        self.multiHeadAtt_layer = MultiHeadGraphAttentionLayer(self.args, self.layer)
        if self.args.encoder_mlp:
            self.MLP = PositionwiseFeedForward(self.args, self.layer, 'encoder')

    def forward(self, local_patten, long_range_patten, X):
        self.input = X
        atted_X = self.multiHeadAtt_layer.forward(local_patten, long_range_patten, self.input)
        if self.args.encoder_mlp:
            enc_output = self.MLP(atted_X)
            return enc_output
        else:
            return atted_X


class Encoder(Module):
    """
    Encoder module, containing  several layers of encoder layer
    :param args: Global hyperparameters setting
    :param layer: layer index
    """
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.n_layers = len(self.args.hidden) - 1
        self.layers = nn.ModuleList([EncoderLayer(self.args, layer) for layer in range(self.n_layers)])

    def forward(self, local_patten, long_range_patten, X):
        # Encoder
        H = X
        for layer in range(self.n_layers):
            # pass into each encoder layer block
            H = self.layers[layer].forward(local_patten, long_range_patten, H)
        # Final node representations
        self.embedding = H
        return self.embedding


class DecoderLayer(Module):
    """
    Decoder layer
    :param encoderLayer: corresponding encoder layer
    """
    def __init__(self, encoderLayer):
        super(DecoderLayer, self).__init__()
        self.EncoderLayer = encoderLayer
        if self.EncoderLayer.args.decoder_mlp:
            self.MLP = PositionwiseFeedForward(self.EncoderLayer.args, self.EncoderLayer.layer, 'decoder')
        # if self.EncoderLayer.layer != 0:
        #     self.norm = nn.BatchNorm1d(self.EncoderLayer.args.hidden[self.EncoderLayer.layer], eps=1e-6)
        self.norm = nn.BatchNorm1d(self.EncoderLayer.args.hidden[self.EncoderLayer.layer], eps=1e-6)

    def forward(self, X):
        if self.EncoderLayer.args.heads > 1:
            heads_num, nodes_num, feature_num = self.EncoderLayer.multiHeadAtt_layer.args.heads, \
                X.shape[0], \
                self.EncoderLayer.multiHeadAtt_layer.args.hidden[self.EncoderLayer.layer + 1]
            # split
            X = X.transpose(0, 1).contiguous().view(heads_num, int(feature_num / heads_num), nodes_num).permute(0, 2, 1)
            # att
            X = torch.matmul(self.EncoderLayer.multiHeadAtt_layer.att, X)
            # linear transfer
            X = torch.matmul(X, self.EncoderLayer.multiHeadAtt_layer.W.permute(0, 2, 1))
            # concat
            X = torch.concat([X[x, :, :] for x in range(self.EncoderLayer.args.heads)], dim=1)
            if self.EncoderLayer.args.nonlinear:
                actv = torch.nn.ELU()
                X = actv(X)
            if self.EncoderLayer.args.decoder_mlp:
                return self.MLP(X)
            else:
                return X
        else:
            # att
            X = torch.matmul(self.EncoderLayer.multiHeadAtt_layer.att, X)
            # linear transfer
            X = torch.matmul(X, self.EncoderLayer.multiHeadAtt_layer.W.permute(1, 0))
            # concat
            if self.EncoderLayer.args.nonlinear:
                actv = torch.nn.ELU()
                X = actv(X)
            if self.EncoderLayer.args.decoder_mlp:
                return self.MLP(X)
            else:
                return X


class Decoder(Module):
    """
    Decoder
    :param encoder: corresponding encoder in MCGATE
    """
    def __init__(self, encoder):
        super(Decoder, self).__init__()
        self.Encoder = encoder
        self.layers = nn.ModuleList([DecoderLayer(model) for model in self.Encoder.layers])

    def forward(self, X):
        # Decoder
        H = X
        for layer in range(self.Encoder.n_layers - 1, -1, -1):
            H = self.layers[layer].forward(H)
        return H

class MCGATE(Module):
    """
    MCGATE
    :param args: Global hyperparameters setting
    """
    def __init__(self, args):
        super(MCGATE, self).__init__()
        self.args = args
        self.encoder = Encoder(self.args)
        self.decoder = Decoder(self.encoder)
        self.input = None
        self.embedding = None
        self.output = None
        self.local_patten = None
        self.long_range_patten = None
        self.mnn_count = torch.tensor([0]).cuda()
        self.epoch = 1

    def identify_sparse_spatial_patten(self, adj):
        # enhance local connection
        self.local_patten = self.strengthen_local_connection(adj)
        # create long-range connection based on MNN
        if (self.epoch % self.args.step == 0) & (self.args.alpha > 0):
            self.long_range_patten = self.get_mnn_long_range_connection( int(self.args.K), adj)
            self.mnn_count = (self.long_range_patten - torch.eye(adj.shape[0]).cuda()).sum(-1).mean()

    def cosine_similarity(self, mod : str = 'multihead'):
        if mod == 'multihead':
            n_heads, n_features, n_nodes = self.args.heads, self.embedding.shape[1], self.embedding.shape[0]
            data = self.embedding.detach().transpose(0, 1).contiguous().view(n_heads, int(n_features / n_heads),
                                                                   n_nodes).permute(0, 2, 1)
            data_L2_normed = F.normalize(data, p=2, dim=-1)
            cosine_similarity = (torch.matmul(data_L2_normed, data_L2_normed.permute(0,2,1)))
        elif mod == 'singlehead':
            data = self.embedding.detach()
            data_L2_normed = F.normalize(data, p=2, dim=-1)
            cosine_similarity = (torch.matmul(data_L2_normed, data_L2_normed.permute(1, 0)))
        return cosine_similarity

    def get_sparse_local_connection(self, k, spatial_adj):
        multi_head_simi = self.cosine_similarity(mod = 'multihead')
        topk_local_index = torch.topk(spatial_adj * multi_head_simi, int(k / 2))[1]
        # generate row index
        row = torch.arange(0, spatial_adj.shape[0]).unsqueeze(-1)
        # get an prune_index
        local_adj = torch.zeros(spatial_adj.size()).cuda()
        local_adj[row, topk_local_index] = 1
        return local_adj*multi_head_simi

    def strengthen_local_connection(self, spatial_adj):
        if self.args.heads > 1:
            multi_head_simi = self.cosine_similarity(mod='multihead')
            tau = 1
            local_patten = torch.sparse.softmax( (spatial_adj * multi_head_simi/tau).to_sparse_coo(), dim = -1 )

            # tau = 1
            # local_patten = (spatial_adj * multi_head_simi/tau)
        else:
            simi = self.cosine_similarity(mod='singlehead')
            tau = 1
            local_patten = torch.sparse.softmax( (spatial_adj * simi/tau).to_sparse_coo(), dim = -1 )
        return local_patten

    def get_mnn_long_range_connection(self, k, spatial_adj):
        simi = self.cosine_similarity(mod='singlehead')
        non_local_simi = (torch.ones(spatial_adj.shape).cuda() - spatial_adj ) * simi
        topk_long_range_index = torch.topk(non_local_simi, k)[1]
        row = torch.arange(0, simi.shape[0]).unsqueeze(-1)
        # get an prune_index
        adj = torch.zeros(simi.size()).cuda()
        adj[row, topk_long_range_index] = 1
        MNN_adj = torch.multiply(adj, adj.T) + torch.eye(spatial_adj.shape[0]).cuda()
        return MNN_adj

    def forward(self, spatial_adj, input_data):
        if self.epoch == 1:
            self.local_patten = spatial_adj
        else:
            self.identify_sparse_spatial_patten(spatial_adj)
        self.embedding = self.encoder.forward(self.local_patten, self.long_range_patten, input_data)
        self.output = self.decoder.forward(self.embedding)
        self.epoch = self.epoch + 1
        return self.embedding, self.output
