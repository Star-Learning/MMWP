import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

from stgcn import STGCN
from models.encoder_decoder import EncoderDecoder
from ode_grid import ODEForecastModelGrid
from ode_graph import ODEForecastModelGraph
from models.common.modules import DecoderGrid, DecoderGraph, MLPTimeGrid, MLPTimeGraph, SpatioTemporalDecoderGraph, SpatioTemporalDecoderGrid


class MMWP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, in_t, out_t, input_nodes, target_nodes):
        super(MMWP, self).__init__()
        self.grid_embedding = EncoderDecoder(
                input_dim=in_channels * 2, 
                hidden_dim=hidden_channels,
                output_dim=out_channels,
                input_seq_len=in_t,
                output_seq_len=out_t
            )


        self.graph_embedding = STGCN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_nodes=input_nodes, 
                target_nodes=target_nodes, 
                t_in=in_t, 
                t_out=out_t
        )

        self.grid_temporal_embedding = ODEForecastModelGrid(channels=hidden_channels, T_out=out_t)
        self.graph_temporal_embedding = ODEForecastModelGraph(feature_dim=hidden_channels, T_out=out_t)

        self.grid_temporal_embedding = MLPTimeGrid(t_in=in_t, t_out=out_t)
        self.graph_temporal_embedding = MLPTimeGraph(t_in=in_t, t_out=out_t) 

        self.grid_out = SpatioTemporalDecoderGrid(in_channels=out_channels, hidden_channels=out_channels, dropout=0.2)
        self.graph_out = SpatioTemporalDecoderGraph(in_dim=out_channels, hidden_dim=out_channels, dropout=0.2)

    def forward(self, grid, graph):
        grid_x = grid
        graph_x, graph_adj = graph

        grid_x = self.grid_embedding(grid_x)
        graph_x = self.graph_embedding(graph_x, graph_adj)

        grid_x = self.grid_temporal_embedding(grid_x)
        graph_x = self.graph_temporal_embedding(graph_x)

        grid_x = self.grid_out(grid_x)
        graph_x = self.graph_out(graph_x)

        return grid_x, graph_x