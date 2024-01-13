import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import TensorDataset, DataLoader
from Model import PIGAT

train_set = torch.load('train_set_state_201908_201910.pth')
val_set = torch.load('val_set_state_201908_201910.pth')

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=True)

adj_1 = torch.load('adj_distance.pt')
adj_2 = torch.load('adj_flow.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

adj_1 = adj_1.to(device)
adj_2 = adj_2.to(device)

in_features = 6
embed_size = 24
hidden_dim = 48
num_nodes = 207
spatial_heads = 2
temporal_heads = 2
graph_heads = 2
num_time = 20
num_time_out = 10
num_layers= 2
alpha = 0.01
beta = 0.005

max_0, max_1,max_2,max_3 = 167,152,32,35.1167

model = PIGAT(in_features,
        embed_size,
        hidden_dim,
        num_nodes,
        spatial_heads,
        temporal_heads,
        graph_heads,
        num_time,
        num_time_out,
        num_layers).to(device)

loss_function = torch.nn.MSELoss()
loss_function = loss_function.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def weight_schedule(epoch, max_val=0.1, mult=-5, max_epochs=100):
    if epoch == 0:
        return 0.
    w = max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)
    w = float(w)
    if epoch > max_epochs:
        return max_val
    return w

# Define early stopping parameters
best_val_loss = np.inf
patience = 10  # Number of epochs to wait for improvement before stopping
patience_counter = 0

for epoch in range(100):  # Number of epochs
    w = weight_schedule(epoch)
    model.train()  # Set the model to training mode
    for batch_x, batch_y in train_loader:
        # Assuming the output of the model is the same shape as input
        optimizer.zero_grad()
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        output,norm_loss = model(batch_x, adj_1, adj_2, a=0.5)

        qt = output[:,2,:,:]
        adqt_dt = torch.gradient(output,dim=2)[0]
        da_dt = adqt_dt[:,0,:,:]
        dd_dt = adqt_dt[:,1,:,:]
        dq_dt = adqt_dt[:,2,:,:]
        dw_dt = adqt_dt[:,3,:,:]

        loss_0 = da_dt*0
        loss_p1 = loss_function((da_dt*max_0-dd_dt*max_1)-dq_dt*max_2,loss_0)
        loss_p2 = loss_function(dw_dt*max_3*20-qt*max_2,loss_0)    

        mse = loss_function(output, batch_y)
        loss = mse/mse.detach() + w*norm_loss/norm_loss.detach() + alpha *loss_p1/loss_p1.detach() + beta *loss_p2/loss_p2.detach()
        loss.backward()
        optimizer.step()

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_x = val_x.float().to(device)
            val_y = val_y.float().to(device)
            val_output, val_norm_loss = model(val_x, adj_1, adj_2, a=0.5)
            # Compute validation loss
            val_loss += loss_function(val_output, val_y).item()
    
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}, Training Loss: {mse.item()}, Validation Loss: {val_loss}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the best model if desired
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Stopping early due to no improvement")
            break
