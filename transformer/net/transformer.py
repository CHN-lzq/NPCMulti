import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn
import os
from models.BA.lib.Cell_DETR_master.segmentation import MultiHeadAttention
from models.BA.lib.Cell_DETR_master.transformer import Transformer



class transformer(nn.Module):
    def __init__(self,
                 point_pred,
                 in_channels=2048,
                 transformer_attention_heads=8,
                 num_encoder_layers=3,
                 num_decoder_layers=2,
                 hidden_features=128,
                 number_of_query_positions=1,
                 segmentation_attention_heads=8,
                 dropout=0):

        super(transformer, self).__init__()

        self.point_pred = point_pred


        self.transformer_attention_heads = transformer_attention_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.hidden_features = hidden_features
        self.number_of_query_positions = number_of_query_positions
        self.transformer_activation = nn.LeakyReLU
        self.segmentation_attention_heads = segmentation_attention_heads

        # in_channels = 2048
        self.convolution_mapping = nn.Conv2d(in_channels=in_channels,
                                             out_channels=hidden_features,
                                             kernel_size=(1, 1),
                                             stride=(1, 1),
                                             padding=(0, 0),
                                             bias=True)

        self.query_positions = nn.Parameter(data=torch.randn(
            number_of_query_positions, hidden_features, dtype=torch.float),
                                            requires_grad=True)

        self.row_embedding = nn.Parameter(data=torch.randn(100,
                                                           hidden_features //
                                                           2,
                                                           dtype=torch.float),
                                          requires_grad=True)
        self.column_embedding = nn.Parameter(data=torch.randn(
            100, hidden_features // 2, dtype=torch.float),
                                             requires_grad=True)
        # self.row_embedding = nn.Parameter(data=torch.randn(200,
        #                                                    hidden_features //
        #                                                    2,
        #                                                    dtype=torch.float),
        #                                   requires_grad=True)
        # self.column_embedding = nn.Parameter(data=torch.randn(
        #     200, hidden_features // 2, dtype=torch.float),
        #                                      requires_grad=True)

        self.transformer = Transformer(d_model=hidden_features,
                                       nhead=transformer_attention_heads,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dropout=dropout,
                                       dim_feedforward=4 * hidden_features,
                                       activation=self.transformer_activation)

        self.trans_out_conv = nn.Conv2d(
            hidden_features + segmentation_attention_heads, in_channels, 1, 1)

        self.segmentation_attention_head = MultiHeadAttention(
            query_dimension=hidden_features,
            hidden_features=hidden_features,
            number_of_heads=segmentation_attention_heads,
            dropout=dropout)
        self.point_pre_layer = nn.Conv2d(hidden_features, 1, kernel_size=1)

    def forward(self, feature_map):


        features = self.convolution_mapping(feature_map)
        height, width = features.shape[2:]
        batch_size = features.shape[0]
        positional_embeddings = torch.cat([self.column_embedding[:height].unsqueeze(dim=0).repeat(height, 1, 1),self.row_embedding[:width].unsqueeze(dim=1).repeat(1, width, 1)],dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        boundary_embedding, features_encoded = self.transformer(features, None, self.query_positions, positional_embeddings)
        boundary_embedding = boundary_embedding.permute(2, 0, 1)

        if self.point_pred == 1:
            point_map = self.point_pre_layer(features_encoded)
            point_map = torch.sigmoid(point_map)
            features_encoded = point_map * features_encoded + features_encoded

        point_map_2 = self.segmentation_attention_head(boundary_embedding, features_encoded.contiguous())

        trans_feature_maps = torch.cat((features_encoded, point_map_2[:, 0]),dim=1)
        trans_feature_maps = self.trans_out_conv(trans_feature_maps)

        return trans_feature_maps

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = transformer(1)
    print(model)
    d = torch.rand((2, 2048, 28, 28))
    o = model(d)
    print(o.size())