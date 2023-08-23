from .dataloader import Dataset_Infer
from .misc import list_img
from .model import Unet
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

def train(batch_size, t1_dir, t1ce_dir, t2_dir,flair_dir, save_path,encoder='resnet34', device='cuda',compiler=False, num_workers=4, model_path='' ):

    # create segmentation model with pretrained encoder
    model = Unet(
        encoder_name=encoder, 
        encoder_depth = 4,
        classes=4, 
        activation=None,
        in_channels=1
    )
    if compiler:
        model = torch.compile(model)

    t1_list = list_img(t1_dir)
    t1ce_list= list_img(t1ce_dir)
    t2_list = list_img(t2_dir)
    flair_list = list_img(flair_dir)

    dataset = Dataset_Infer(
        t1_list,
        t1ce_list,
        t2_list,
        flair_list,
        preprocessing=True
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    ##load weights to model and run inference just to get output
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    with tqdm(
            dataloader
        ) as iterator:
            for x1, x2, x3, x4, index in iterator:
                x1, x2, x3, x4 = x1.to(device), x2.to(device), x3.to(device), x4.to(device)
                y_pred = model(x1,x2,x3,x4)
                mask = torch.argmax(y_pred, dim=1)
                mask = mask.cpu().detach().numpy()
                for i in range(0,mask.shape[0]):
                    name = t1_list[index[i]].split('/')
                    directory_path = save_path+name[-2]+'/'
                    if not os.path.exists(directory_path):
                        os.mkdir(directory_path)
                    np.save(directory_path +name[-1].split('.')[0]+'.npy',mask[i])

def predict(configs):
    train(configs['batch_size'], configs['t1_dir'],configs['t1ce_dir'], configs['t2_dir'],configs['flair_dir'],configs['save_path'],
          configs['encoder'], configs['device'], configs['compile'], configs['num_workers'], configs['model_path'])
