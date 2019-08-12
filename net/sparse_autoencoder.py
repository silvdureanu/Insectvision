import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset, Dataset, DataLoader
import numpy as np

lr = 1e-3
nr_epochs = 200
nr_pn = 360
nr_kc = 20000
kc_ratio = 0.01

class SparseAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pn2kc = nn.Linear(nr_pn,nr_kc)
        self.kc2pn = nn.Linear(nr_kc,nr_pn)
    def forward(self, pn):
        kc_activation = torch.sigmoid(self.pn2kc(pn))
        restored_input = torch.sigmoid(self.kc2pn(kc_activation))
        return kc_activation, restored_input


if __name__ == "__main__":
    torch.manual_seed(2019)
    f1 = np.load('random_im_x_and_y.npz')
    f2 = np.load('sideways_im_x_and_y.npz')
    f3 = np.load('rotating_im_x_and_y.npz')
    x1 = f1['ims'] / 255
    y1 = f1['ims'] / 255
    x2 = f2['imsx'] / 255
    y2 = f2['imsy'] / 255
    x3 = f3['ims'] / 255
    y3 = f3['ims'] / 255
    x_data = np.vstack((x1,x2,x3))
    y_data = np.vstack((y1,y2,y3))
    x_torch = torch.from_numpy(x_data).float()
    y_torch = torch.from_numpy(y_data).float()
    dataset = TensorDataset(x_torch, y_torch)
    train_set, val_set, test_set = random_split(dataset, [12800,3200,4000])
    train_batcher = DataLoader(dataset=train_set, batch_size=10, shuffle=True)
    val_batcher = DataLoader(dataset=val_set, batch_size=3200, shuffle=False)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ae = (SparseAutoencoder()).to(device)

    mse_loss = nn.MSELoss(reduction='mean')
    l1_loss = nn.L1Loss()
    optimiser = optim.SGD(ae.parameters(), lr=lr)

    for i in range(nr_epochs):
        nr = 0
        for x, y in train_batcher:
            nr+=1
            print(i)
            print(nr)
            x = x.to(device)
            y = y.to(device)

            ae.train()
            kc, y_predict = ae(x)
            loss = mse_loss(y_predict,y) + 0.05* torch.sum(kc)
            actualp = torch.sum(kc) / kc.nelement()
            #kl_loss = kc_ratio * torch.log(kc_ratio / actualp) + (1-kc_ratio) * torch.log((1-kc_ratio) / (1-actualp))
            #loss = mse_loss(y_predict,y) + 8000*kl_loss
            print(actualp)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            #npkc = kc.cpu().detach().numpy()
            #print(((npkc>0.01).sum()) / npkc.size)
            print(loss.data)

        for x,y in val_batcher:
            x = x.to(device)
            y = y.to(device)
            ae.eval()
            kc, y_predict = ae(x)
            loss = mse_loss(y_predict,y)
            print(loss.data)

    torch.save(ae.state_dict(), './ae.params')
