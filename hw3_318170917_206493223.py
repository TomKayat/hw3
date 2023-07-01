import dataset
import torch
import os
import pickle
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv

torch.manual_seed(12)
NUM_FEATURES = 128
NUM_CLASSES = 40


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


def train(model, data, optimizer, criterion):
    # Training the model
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)  # Doing a forward pass o the model
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask].squeeze(1))  # Computing the train loss
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients (optimizer)
    return loss


def test(model, data):
    # Testing the model
    with torch.no_grad():
        model.eval()
        # get predictions vectors
        out = model(data.x, data.edge_index)
        predictons = out.argmax(dim=1)
        test_correct = predictons[data.val_mask] == data.y[data.val_mask].reshape(-1)  # Compute num of correct samples
        test_acc = int(test_correct.sum()) / len(data.val_mask)  # Computing accuracy
    return test_acc


def run_model(p_dropout=None, hidden_channels_size=None, lr=None):
    # Servers as a main: gets the data, train the model, and test it

    # Get the data
    dataset_ojb = dataset.HW3Dataset(root='data/hw3/')
    data = dataset_ojb[0]

    # Define the model
    model = GAT(NUM_FEATURES, 8, NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Result txt file
    if os.path.isfile(os.path.join("final", 'model_final.txt')):
        os.remove(os.path.join("final", 'model_final.txt'))

    # epochs of training
    for epoch in range(1, 441):
        loss = train(model, data, optimizer, criterion)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    # saving the model pickle and results txt file
    test_accurcy = test(model, data)
    if os.path.isfile('model.pkl'):
        os.remove('model.pkl')
    pickle.dump(model, open('model.pkl', 'wb'))
    print(f'Test Accuracy: {test_accurcy:.4f}')
    with open('model_final.txt', 'a') as f:
        f.write(f'Test Accuracy: {test_accurcy:.4f} \n')

    return test_accurcy


if __name__ == '__main__':
    run_model(hidden_channels_size=256, lr=0.01)
