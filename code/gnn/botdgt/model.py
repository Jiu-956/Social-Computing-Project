import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import TransformerConv


class NodeFeatureEmbeddingLayer(nn.Module):
    def __init__(self, hidden_dim, numerical_feature_size=5, categorical_feature_size=3,
                 des_feature_size=768, tweet_feature_size=768, dropout=0.3):
        super().__init__()
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.numerical_feature_linear = nn.Sequential(
            nn.Linear(numerical_feature_size, hidden_dim // 4),
            self.activation
        )
        self.categorical_feature_linear = nn.Sequential(
            nn.Linear(categorical_feature_size, hidden_dim // 4),
            self.activation
        )
        self.des_feature_linear = nn.Sequential(
            nn.Linear(des_feature_size, hidden_dim // 4),
            self.activation
        )
        self.tweet_feature_linear = nn.Sequential(
            nn.Linear(tweet_feature_size, hidden_dim // 4),
            self.activation
        )
        self.total_feature_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.activation
        )
        self.init_weights()

    def forward(self, des_tensor, tweet_tensor, num_prop, category_prop):
        num_prop = self.numerical_feature_linear(num_prop)
        category_prop = self.categorical_feature_linear(category_prop)
        des_tensor = self.des_feature_linear(des_tensor)
        tweet_tensor = self.tweet_feature_linear(tweet_tensor)
        x = torch.cat((num_prop, category_prop, des_tensor, tweet_tensor), dim=1)
        x = self.dropout(x)
        x = self.total_feature_linear(x)
        return x

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()


class GraphStructuralLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super().__init__()
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.layer1 = TransformerConv(hidden_dim, hidden_dim // n_heads, heads=n_heads, concat=True, dropout=dropout)
        self.layer2 = TransformerConv(hidden_dim, hidden_dim // n_heads, heads=n_heads, concat=True, dropout=dropout)
        self.init_weights()

    def forward(self, x, edge_index):
        out1 = self.layer1(x, edge_index)
        out1 = self.activation(out1)
        out1 = self.layer2(out1, edge_index)
        out1 += x
        out1 = self.activation(out1)
        return out1

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()


class GraphTemporalLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout, num_time_steps, temporal_module_type):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.temporal_module_type = temporal_module_type
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.feedforward_linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.PReLU()
        self.feedforward_linear_2 = nn.Linear(hidden_dim, 2)
        self.attention_dropout = nn.Dropout(dropout)
        self.num_time_steps = num_time_steps
        self.position_embedding_temporal = nn.Embedding(self.num_time_steps, hidden_dim)
        self.GRU = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.LSTM = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.init_weights()

    def forward(self, structural_output, position_embedding_clustering_coefficient,
                position_embedding_bidirectional_links_ratio, exist_nodes):
        if self.temporal_module_type == 'gru':
            gru_output, _ = self.GRU(structural_output)
            y = structural_output + gru_output
            return self.feed_forward(y)
        elif self.temporal_module_type == 'lstm':
            lstm_output, _ = self.LSTM(structural_output)
            y = structural_output + lstm_output
            return self.feed_forward(y)
        else:
            structural_input = structural_output
            position_inputs = torch.arange(0, self.num_time_steps).reshape(1, -1).repeat(
                structural_output.shape[0], 1).long().to(structural_output.device)
            position_embedding_temporal = self.position_embedding_temporal(position_inputs)
            temporal_inputs = (structural_output + position_embedding_temporal
                             + position_embedding_clustering_coefficient
                             + position_embedding_bidirectional_links_ratio)
            temporal_inputs = self.layer_norm(temporal_inputs)
            q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2], [0]))
            k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2], [0]))
            v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2], [0]))
            split_size = int(q.shape[-1] / self.n_heads)
            q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0)
            k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0)
            v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0)
            outputs = torch.matmul(q_, k_.permute(0, 2, 1))
            outputs = outputs / (split_size ** 0.5)
            diag_val = torch.ones_like(outputs[0])
            tril = torch.tril(diag_val)
            sequence_mask = tril[None, :, :].repeat(outputs.shape[0], 1, 1)
            total_mask = sequence_mask.float()
            padding = torch.ones_like(total_mask) * (-1e9)
            outputs = torch.where(total_mask == 0, padding, outputs)
            outputs = F.softmax(outputs, dim=2)
            outputs = self.attention_dropout(outputs)
            outputs = torch.matmul(outputs, v_)
            multi_head_attention_output = torch.cat(
                torch.split(outputs, split_size_or_sections=int(outputs.shape[0] / self.n_heads), dim=0), dim=2)
            multi_head_attention_output += structural_input
            multi_head_attention_output = self.layer_norm(multi_head_attention_output)
            multi_head_attention_output = self.feed_forward(multi_head_attention_output)
            return multi_head_attention_output

    def init_weights(self):
        nn.init.kaiming_uniform_(self.Q_embedding_weights)
        nn.init.kaiming_uniform_(self.K_embedding_weights)
        nn.init.kaiming_uniform_(self.V_embedding_weights)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()

    def feed_forward(self, inputs):
        out = self.feedforward_linear_1(inputs)
        out = self.activation(out)
        out = self.feedforward_linear_2(out)
        return out


class PositionEncodingClusteringCoefficient(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.clustering_coefficient_linear = nn.Linear(1, hidden_dim)
        self.init_weights()

    def forward(self, clustering_coefficient):
        clustering_coefficient = self.clustering_coefficient_linear(clustering_coefficient)
        return clustering_coefficient

    def init_weights(self):
        nn.init.kaiming_normal_(self.clustering_coefficient_linear.weight)


class PositionEncodingBidirectionalLinks(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.bidirectional_links_ratio_linear = nn.Linear(1, hidden_dim)

    def forward(self, bidirectional_links_ratio):
        bidirectional_links_ratio = self.bidirectional_links_ratio_linear(bidirectional_links_ratio)
        return bidirectional_links_ratio


class BotDyGNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.structural_head_config = args.structural_head_config
        self.structural_drop = args.structural_drop
        self.temporal_head_config = args.temporal_head_config
        self.temporal_drop = args.temporal_drop
        self.window_size = args.window_size
        self.temporal_module_type = args.temporal_module_type
        (self.node_feature_embedding_layer,
         self.position_encoding_clustering_coefficient_layer,
         self.position_encoding_bidirectional_links_ratio_layer,
         self.structural_layer,
         self.temporal_layer) = self.build_model()

    def forward(self, all_snapshots_des_tensor, all_snapshots_tweet_tensor, all_snapshots_num_prop,
                all_snapshots_category_prop, all_snapshots_edge_index, all_snapshots_clustering_coefficient,
                all_snapshots_bidirectional_links_ratio, all_snapshots_exist_nodes, current_batch_size):
        all_snapshots_structural_output = []
        num_of_snapshot = len(all_snapshots_des_tensor)
        for t in range(num_of_snapshot):
            one_snapshot_des_tensor = all_snapshots_des_tensor[t]
            one_snapshot_tweet_tensor = all_snapshots_tweet_tensor[t]
            one_snapshot_num_prop = all_snapshots_num_prop[t]
            one_snapshot_category_prop = all_snapshots_category_prop[t]
            x = self.node_feature_embedding_layer(one_snapshot_des_tensor, one_snapshot_tweet_tensor,
                                                   one_snapshot_num_prop, one_snapshot_category_prop)
            one_snapshot_edge_index = all_snapshots_edge_index[t]
            output = self.structural_layer(x, one_snapshot_edge_index)[:current_batch_size]
            all_snapshots_structural_output.append(output)
        all_snapshots_structural_output = torch.stack(all_snapshots_structural_output, dim=1)
        if torch.any(torch.isnan(all_snapshots_structural_output)):
            print('structural_output has nan')

        position_embedding_clustering_coefficient = [
            self.position_encoding_clustering_coefficient_layer(all_snapshots_clustering_coefficient[t])[
            :current_batch_size] for t in range(num_of_snapshot)]
        position_embedding_bidirectional_links_ratio = [
            self.position_encoding_bidirectional_links_ratio_layer(all_snapshots_bidirectional_links_ratio[t])[
            :current_batch_size] for t in range(num_of_snapshot)]
        position_embedding_clustering_coefficient = torch.stack(position_embedding_clustering_coefficient, dim=1)
        position_embedding_bidirectional_links_ratio = torch.stack(position_embedding_bidirectional_links_ratio, dim=1)
        exist_nodes = all_snapshots_exist_nodes.transpose(0, 1)
        temporal_output = self.temporal_layer(all_snapshots_structural_output,
                                               position_embedding_clustering_coefficient,
                                               position_embedding_bidirectional_links_ratio, exist_nodes)
        if torch.any(torch.isnan(temporal_output)):
            print('temporal_output has nan')
        return temporal_output

    def build_model(self):
        node_feature_embedding_layer = NodeFeatureEmbeddingLayer(hidden_dim=self.hidden_dim)
        position_encoding_clustering_coefficient_layer = PositionEncodingClusteringCoefficient(
            hidden_dim=self.hidden_dim)
        position_encoding_bidirectional_links_ratio_layer = PositionEncodingBidirectionalLinks(
            hidden_dim=self.hidden_dim)
        structural_layer = GraphStructuralLayer(hidden_dim=self.hidden_dim,
                                                 n_heads=self.structural_head_config,
                                                 dropout=self.structural_drop)
        temporal_layer = GraphTemporalLayer(hidden_dim=self.hidden_dim,
                                              n_heads=self.temporal_head_config,
                                              dropout=self.temporal_drop,
                                              num_time_steps=self.window_size,
                                              temporal_module_type=self.temporal_module_type)
        return (node_feature_embedding_layer, position_encoding_clustering_coefficient_layer,
                position_encoding_bidirectional_links_ratio_layer, structural_layer, temporal_layer)
