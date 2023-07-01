import pickle
import torch
from dataset import *
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
torch.manual_seed(12)

dataset = HW3Dataset(root='data/hw3/')
dataset = dataset[0]

NUM_FEATURES = 128
NUM_CLASSES = 40


# our final model
class GAT(torch.nn.Module):
    # Graph Attention Network Model with 2 layers
    def __init__(self, dim_in, dim_h, dim_out, heads=8):
        super().__init__()
        self.gat_l1 = GATv2Conv(dim_in, dim_h, heads=8)
        self.gat_l2 = GATv2Conv(dim_h * heads, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.01,
                                          weight_decay=5e-4)

    def forward(self, x, edge_inx):
        h = self.gat_l1(x, edge_inx)
        h = F.elu(h)
        h = self.gat_l2(h, edge_inx)
        return F.log_softmax(h, dim=1)

# get data
x = dataset.x
edge_index = dataset.edge_index

# define model
model = GAT(NUM_FEATURES, 8, NUM_CLASSES)

# load model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


# make prediction
model.eval()
with torch.no_grad():
    _, pred = model(x, edge_index).max(dim=1)

# write to prediction file
with open("prediction.csv", "w") as f:
    # headers
    f.write("idx,prediction\n")
    for i, p in enumerate(pred):
        f.write(f"{int(i)},{int(p)}\n")

print('Done')
