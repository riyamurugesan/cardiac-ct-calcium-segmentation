import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
import json
import torch
import matplotlib.pyplot as plt

class CalciumDataset(Dataset):
    def __init__(self, data_dir,pt_ids):
        self.data_dir = Path(data_dir)
        self.pt_ids = pt_ids

    def __len__(self):
        return len(self.pt_ids)

    def __getitem__(self,ind):
        pt_id = self.pt_ids[ind]
        pt_dir = self.data_dir / str(pt_id)

        ct = np.load(pt_dir / 'ct_volume.npy')
        mask = np.load(pt_dir / 'mask.npy')

        ct = np.clip(ct,-1000,3000)
        ct = (ct +1000)/4000.0
        ct = ct[np.newaxis,...]
        mask = mask[np.newaxis,...]

        ct = torch.from_numpy(ct).float()
        mask = torch.from_numpy(mask).float()
        return ct, mask
    
def training_epoch(model,loader,optimizer,loss_func, device):
    model.train()
    epoch_loss = 0

    for ct,mask in loader:
        ct = ct.to(device)
        mask = mask.to(device)
        # go forward and get loss
        optimizer.zero_grad()
        output = model(ct)
        loss = loss_func(output,mask)
        # go backwards
        loss.backward()
        optimizer.step()

        #add loss to epoch loss
        epoch_loss = epoch_loss + loss.item()

    avg_loss = epoch_loss / len(loader)
    return avg_loss

def validate(model, loader, metric,device):
    model.eval()
    metric.reset()

    with torch.no_grad():
        for ct, mask in loader:
            ct = ct.to(device)
            mask = mask.to(device)

            #forward
            output = model(ct)

            #activation and conversion to 0/1 state (calcium or no calcium)
            output = torch.sigmoid(output)
            output = (output > 0.5).float()

            metric(y_pred = output, y=mask)
    dice = metric.aggregate().item()
    metric.reset()
    return dice

def main():
    data_dir = Path('training_data')
    output_dir = Path('trained_model')
    output_dir.mkdir(exist_ok=True)

    #use gpu if available, cpu if not
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    #set hyperparameters
    batch_size = 1
    num_epochs = 50 # go thru training set 50 times
    step_size = 0.0001

    
    all_pts = []
    for f in data_dir.iterdir():
        if f.is_dir():
            all_pts.append(f)

    all_pts = sorted(all_pts,key=int)

    #split 80/20 train test
    train_ind = int(0.8*len(all_pts))
    training_pts = all_pts[:train_ind]
    val_pts = all_pts[train_ind:]
    #create datasets
    training_data = CalciumDataset(data_dir,training_pts)
    val_data = CalciumDataset(data_dir,val_pts)

    # create data loaders
    training_loader = DataLoader(training_data,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_data,batch_size=batch_size,shuffle=False)

    # Create model! 
    model = UNet(spatial_dims=3,in_channels=1,out_channels=1,
                 channels=(16,32,64,128,256),strides=(2,2,2,2),
                 num_res_units=2, norm=Norm.BATCH,).to(device)
    
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    
    loss_func = DiceLoss(sigmoid=True)

    optimizer = torch.optim.Adam(model.parameters(),lr=step_size)
    dice_metric = DiceMetric(include_background=True,reduction="mean")

    # training loop
    best_dice = 0.0
    hist = {'training_loss': [], 'val_dice': []}

    for epoch in range(num_epochs):
        training_loss = training_epoch(model,training_loader,optimizer,loss_func,device)

        val_dice = validate(model,val_loader,dice_metric,device)

        hist['training_loss'].append(training_loss)
        hist['val_dice'].append(val_dice)
        # save model with new best dice
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(),output_dir / 'best_model.pth')
        
    # save final model
    torch.save(model.state_dict(),output_dir / 'final_model.pth')

    # plot loss by epoch and dice by epoch
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
    # Loss
    ax1.plot(hist['training_loss'])
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Dice Loss')
    ax1.grid(True)
    
    # Dice
    ax2.plot(hist['val_dice'])
    ax2.set_title('Validation Dice Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    main()
    
        
    


    