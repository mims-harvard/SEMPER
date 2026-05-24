import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.nn import Dropout, Linear, Sequential, ReLU
# from torch_geometric.nn.conv import GPSConv, GINEConv
from torch_geometric.nn.conv import GPSConv
from torch_geometric.nn.inits import glorot, zeros


class PCTConv(nn.Module):
    def __init__(self, in_channels, num_ppi_relations, num_mg_relations, ppi_data, out_channels, sem_att_channels, pc_att_channels, node_heads=3, tissue_update = 100):
        super().__init__()
        
        self.ppi_data = ppi_data
        self.in_channels = in_channels
        self.num_ppi_relations = num_ppi_relations
        self.num_mg_relations = num_mg_relations
        self.out_channels = out_channels
        self.sem_att_channels = sem_att_channels
        self.node_heads = node_heads
        self.tissue_update = tissue_update
        self.tissue_update = 100

        # Cell-type specific PPI weights
        self.ppi_attn = dict()

        # Independent GAT per cell type specific PPI network
        self.ppi_w = torch.nn.ModuleList()

        for celltype, ppi in ppi_data.items():
            self.ppi_w.append(GPSConv(
                channels=in_channels,  # Size of each input sample
                conv=GATv2Conv(in_channels, in_channels // node_heads, heads=node_heads),
                # conv=GATv2Conv(in_channels, out_channels, heads=node_heads),  # Local message passing layer
                heads=node_heads,  # Number of attention heads
                dropout=0.0,  # Dropout
                act='relu',  # Activation function
                norm='batch_norm',  # Normalization function
                attn_type='multihead'  # Global attention type (can also be 'performer')
            ))

        # Linear layers after PPI-specific GPSConv
        self.ppi_linear_layers = torch.nn.ModuleList()
        for celltype, ppi in ppi_data.items():
            self.ppi_linear_layers.append(Linear(in_channels, out_channels * node_heads))  # Adjust output to desired shape

        
        # Independent GAT for metagraph
        self.mg_conv_in = GPSConv(
                channels=in_channels,  # Size of each input sample
                conv=GATv2Conv(in_channels, in_channels // node_heads, heads=node_heads),
                # conv=GATv2Conv(in_channels, out_channels, heads=node_heads),  # Local message passing layer
                heads=node_heads,  # Number of attention heads
                dropout=0.0,  # Dropout
                act='relu',  # Activation function
                norm='batch_norm',  # Normalization function
                attn_type='multihead'  # Global attention type (can also be 'performer')
            )

        # Linear layer to reshape the output of GPSConv
        self.linear_mg = Linear(in_channels, out_channels * node_heads)  # Adjust output to desired shape

        self.mg_conv_out = GPSConv(
                channels=out_channels * node_heads,  # Size of each input sample
                conv=GATv2Conv(out_channels * node_heads, out_channels, heads=node_heads),
                # conv=GATv2Conv(in_channels, out_channels, heads=node_heads),  # Local message passing layer
                heads=node_heads,  # Number of attention heads
                dropout=0.0,  # Dropout
                act='relu',  # Activation function
                norm='batch_norm',  # Normalization function
                attn_type='multihead'  # Global attention type (can also be 'performer')
            )

        self.linear_mg_out = Linear(out_channels * node_heads, out_channels * node_heads)


        # Semantic attention (shared across networks)
        self.W = nn.Parameter(torch.Tensor(1, 1, out_channels * node_heads, sem_att_channels))
        self.b = nn.Parameter(torch.Tensor(1, 1, sem_att_channels))
        self.q = nn.Parameter(torch.Tensor(1, 1, sem_att_channels))
        nn.init.xavier_uniform_(self.W, gain = nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.b, gain = nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.q, gain = nn.init.calculate_gain('leaky_relu'))

        # Protein - Cell type attention (shared across networks)
        self.pc_W = nn.Parameter(torch.Tensor(1, 1, out_channels * node_heads, pc_att_channels))
        self.pc_b = nn.Parameter(torch.Tensor(1, 1, pc_att_channels))
        self.pc_q = nn.Parameter(torch.Tensor(1, 1, pc_att_channels))
        nn.init.xavier_uniform_(self.pc_W, gain = nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.pc_b, gain = nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.pc_q, gain = nn.init.calculate_gain('leaky_relu'))
        
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.W)
        zeros(self.b)
        glorot(self.q)

    def _per_data_forward(self, x, edgetypes, node_conv, linear_layer=None):
        out = []
        for i, edgetype in enumerate(edgetypes):
            if isinstance(edgetype, torch.Tensor) and len(edgetype.shape) > 1 and edgetype.shape[1] > 0:
                try:
                    result = node_conv(x, edgetype)
                    out.append(result)
                except RuntimeError as e:
                    print(f"Error in node_conv with edgetype shape {edgetype.shape} and x shape {x.shape}: {e}")

        # Calculate node-level attention representations
        # out = [node_conv(x, edgetype) for edgetype in edgetypes if edgetype.shape[1] > 0]
        out = torch.stack(out, dim=1).to(x.device)
    
        # Apply non-linearity
        out = F.leaky_relu(out)

        # Apply the linear layer to reshape the output (optional)
        if linear_layer:
            out = linear_layer(out)  # Reshape and apply the linear layer

        # Aggregate node-level representation using semantic level attention        
        w = torch.sum(self.W * out.unsqueeze(-1), dim=-2) + self.b
        w = torch.tanh(w)
        beta = torch.sum(self.q * w, dim=-1)
        beta = torch.softmax(beta, dim=1)
        z = torch.sum(out * beta.unsqueeze(-1), dim=1)
        return z

    def forward(self, ppi_x, mg_x, ppi_metapaths, mg_metapaths, ppi_edge_index, mg_edge_index, tissue_neighbors, init_cci=False):
        
        if init_cci: mg_x_list = []
        else: # Project metagraph embeddings to the same dimension as PPI
            mg_x = self._per_data_forward(mg_x, mg_metapaths, self.mg_conv_in, self.linear_mg)

        for celltype, x in ppi_x.items(): # Iterate through cell-type specific PPI layers
            if len(ppi_metapaths[celltype]) == 0:
                ppi_x[celltype] = []
            else:
                ppi_x[celltype] = self._per_data_forward(x, ppi_metapaths[celltype], self.ppi_w[celltype], self.ppi_linear_layers[celltype])

            # Attention on PPI nodes per cell type
            w = torch.sum(self.pc_W * ppi_x[celltype].unsqueeze(-1), dim=-2) + self.pc_b
            w = torch.tanh(w)
            gamma = torch.sum(self.pc_q * w, dim=-1)

            gamma = torch.softmax(gamma, dim=1)
            self.ppi_attn[celltype] = gamma.squeeze(0)

            if init_cci: # Initialize CCI embeddings using PPI embeddings
                weighted_x = torch.sum(ppi_x[celltype] * self.ppi_attn[celltype].unsqueeze(-1), dim=0)
                mg_x_list.append(weighted_x)
            else: # Update CCI embeddings
                mg_x[celltype, :] += torch.sum(ppi_x[celltype] * self.ppi_attn[celltype].unsqueeze(-1), dim=0)

        if init_cci: # Concatenate initialized metagraph embeddings
            mg_x = torch.stack(mg_x_list)
            bto = torch.zeros(len(tissue_neighbors), mg_x.shape[1])
            mg_x = torch.cat((mg_x, torch.normal(bto, std=1).to(mg_x.device)))
        for i in range(self.tissue_update): # Initialize tissue embeddings in a more meaningful way
            for t in sorted(tissue_neighbors):
                assert len(tissue_neighbors[t]) != 0
                mg_x[t, :] = torch.mean(mg_x[tissue_neighbors[t]], 0)
        
        mg_x = self._per_data_forward(mg_x, mg_metapaths, self.mg_conv_out, self.linear_mg_out)
        
        return ppi_x, mg_x


class PPIConv(nn.Module):
    def __init__(self, in_channels, num_ppi_relations, out_channels, ppi_data, sem_att_channels, node_heads=3):
        super().__init__()
        self.in_channels = in_channels
        self.num_ppi_relations = num_ppi_relations
        self.out_channels = out_channels
        self.sem_att_channels = sem_att_channels
        self.node_heads = node_heads

        # Independent GAT per cell type specific PPI network
        self.ppi_w = torch.nn.ModuleList()
        for celltype, ppi in ppi_data.items():
            self.ppi_w.append(GPSConv(
                channels=in_channels,
                conv=GATv2Conv(in_channels, in_channels // node_heads, heads=node_heads),
                heads=node_heads,
                dropout=0.0,
                act='relu',
                norm='batch_norm',
                attn_type='multihead'
            ))

        # Linear layers after PPI-specific GPSConv
        self.ppi_linear_layers = torch.nn.ModuleList()
        for celltype, ppi in ppi_data.items():
            self.ppi_linear_layers.append(Linear(in_channels, out_channels * node_heads))  # Adjust output to desired shape

        self.W = nn.Parameter(torch.Tensor(1, 1, out_channels * node_heads, sem_att_channels))

        self.b = nn.Parameter(torch.Tensor(1, 1, sem_att_channels))
        self.q = nn.Parameter(torch.Tensor(1, 1, sem_att_channels))
        nn.init.xavier_uniform_(self.W, gain = nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.b, gain = nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.q, gain = nn.init.calculate_gain('leaky_relu'))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.W)
        zeros(self.b)
        glorot(self.q)

    def _per_data_forward(self, x, metapaths, node_conv, linear_layer = None):

        # Calculate node-level attention representations
        out = [node_conv(x, metapath) for metapath in metapaths if metapath.shape[1] > 0]
        out = torch.stack(out, dim=1).to(x.device)
        
        # Apply non-linearity
        out = F.leaky_relu(out)

        # Apply the linear layer to reshape the output (optional)
        if linear_layer:
            out = linear_layer(out)  # Reshape and apply the linear layer

        # Aggregate node-level representation using semantic level attention
        w = torch.sum(self.W * out.unsqueeze(-1), dim=-2) + self.b
        w = torch.tanh(w)
        beta = torch.sum(self.q * w, dim=-1)
        beta = torch.softmax(beta, dim=1)
        z = torch.sum(out * beta.unsqueeze(-1), dim=1)

        return z

    def forward(self, ppi_x, ppi_metapaths, mg_x, ppi_attn):
        
        for celltype, x in ppi_x.items(): # Iterate through cell-type specific PPI layers
            if len(ppi_metapaths[celltype]) == 0: ppi_x[celltype] = []
            else: # Update using meta-path attention
                ppi_x[celltype] = self._per_data_forward(x, ppi_metapaths[celltype], self.ppi_w[celltype], self.ppi_linear_layers[celltype])

            # Downpool using cell-type embedding
            gamma = ppi_attn[celltype] # (n,) where n = number of proteins in the cell type (in the batch)
            ppi_x[celltype] += (mg_x[celltype, :].repeat(len(gamma), 1) * gamma.unsqueeze(-1))
        
        return ppi_x