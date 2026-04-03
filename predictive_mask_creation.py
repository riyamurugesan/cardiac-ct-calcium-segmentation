import numpy as np
import torch
from pathlib import Path
from monai.networks.nets import UNet
from monai.networks.layers import Norm

MODEL_PATH = 'Output/best_model.pth'

class CalciumPredictor:
    def __init__(self,model_path,device):
        #use gpu if available, cpu if not
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # same architecture that was used for training
        self.model = UNet(spatial_dims = 3,in_channels=1,out_channels=1,channels=(16,32,64,128,256),strides=(2,2,2,2),
                    num_res_units=2,norm=Norm.BATCH,).to(self.device)

        #load weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def preprocess(self, ct_arr):
        self.original_shape = ct_arr.shape
        ct = np.clip(ct_arr,-1000,3000)
        ct = (ct +1000)/4000.0

        h, l, w = ct.shape
        self.padding_h = (16 - h%16) % 16
        self.padding_l = (16 - l%16) % 16
        self.padding_w = (16 - w%16) % 16
        padding = ((self.padding_h // 2, self.padding_h - self.padding_h // 2),
                    (self.padding_l // 2, self.padding_l - self.padding_l // 2),
                    (self.padding_w // 2, self.padding_w - self.padding_w // 2))
        ct = np.pad(ct,padding,mode='constant',constant_values=0)

        ct = ct[np.newaxis,...]
        ct = torch.from_numpy(ct).float()
        ct = ct.unsqueeze(0)
        return ct

    def postprocess(self, pred):
        #remove padding to get back to original dims

        pred = pred[0,0].cpu().numpy()
        h, l, w = self.original_shape
        start_h = self.padding_h // 2 
        start_l = self.padding_l // 2
        start_w = self.padding_w // 2

        pred = pred[start_h: start_h+h, start_l: start_l+l, start_w: start_w+w]
        return pred

    
    def predict(self, ct_arr):
        ct_tensor = self.preprocess(ct_arr)
        ct_tensor = ct_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(ct_tensor)
            output = torch.sigmoid(output)
            pred = (output > 0.5).float()
        
        calcium_mask = self.postprocess(pred)
        return calcium_mask



        


