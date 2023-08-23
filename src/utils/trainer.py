import wandb
import segmentation_models_pytorch as smp
from .train_utils import TrainEpoch, ValidEpoch
from .loss import custom_loss
from .dataloader import Dataset
from .transformations import get_training_augmentation, get_validation_augmentation
from .misc import list_img
from .model import Unet
from .loss import DiceLoss
from .metrics import IoU
from torchmetrics import JaccardIndex
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def train(epochs, batch_size, t1_dir, t1ce_dir, t2_dir,flair_dir, seg_dir,encoder='resnet34', encoder_weights='imagenet', device='cuda', lr=1e-4,beta=1, contrastive=True, compiler=False, num_workers=4, checkpoint='' ):

    # create segmentation model with pretrained encoder
    model = Unet(
        encoder_name=encoder, 
        encoder_weights=encoder_weights, 
        encoder_depth = 4,
        classes=4, 
        activation=None,
        in_channels=1,
        #fusion=True,
        contrastive=contrastive
    )
    if compiler:
        model = torch.compile(model)

    t1_list = list_img(t1_dir)
    t1ce_list= list_img(t1ce_dir)
    t2_list = list_img(t2_dir)
    flair_list = list_img(flair_dir)
    seg_list = list_img(seg_dir)

    t1_train_list, t1_val_list = train_test_split(t1_list, test_size=0.2, random_state = 42)
    t1ce_train_list, t1ce_val_list = train_test_split(t1ce_list, test_size=0.2, random_state = 42)
    t2_train_list, t2_val_list = train_test_split(t2_list, test_size=0.2, random_state = 42)
    flair_train_list, flair_val_list = train_test_split(flair_list, test_size=0.2, random_state = 42)
    seg_train_list, seg_val_list = train_test_split(seg_list, test_size=0.2, random_state = 42)
 
    # preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    train_dataset = Dataset(
        t1_train_list,
        t1ce_train_list,
        t2_train_list,
        flair_train_list,
        seg_train_list,
        augmentation=get_training_augmentation(), 
        preprocessing=True
    )

    valid_dataset = Dataset(
       t1_val_list,
       t1ce_val_list,
       t2_val_list,
       flair_val_list,
       seg_val_list,
       augmentation=get_validation_augmentation(), 
       preprocessing= True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True, persistent_workers=True)

    loss = custom_loss(batch_size, beta=beta, contrastive=contrastive)


    D = DiceLoss(mode= 'multiclass', from_logits=False, metric=True)
    J = IoU(num_classes=4)
    D.__name__ = 'dice'
    J.__name__ = 'jaccard'
    metrics = [
        D,
        J,
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=lr),
    ])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,250)
    if checkpoint != '':
        model.load_state_dict(torch.load(checkpoint))
        print('Checkpoint Loaded!')
        
    train_epoch = TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=device,
        verbose=True,
        contrastive=contrastive
    )
    valid_epoch = ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=device,
        verbose=True,
        contrastive=contrastive
    )

    max_dice = 0
    max_jaccard = 0
    counter = 0
    for i in range(0, epochs):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        # scheduler.step()
        #wandb.log({'epoch':i+1,'t_loss':train_logs['custom_loss'],'t_dice':train_logs['dice'],'t_jaccard':train_logs['jaccard']})
        train_logs['dice'] = 1-train_logs['dice']
        valid_logs['dice'] = 1-valid_logs['dice']
        wandb.log({'epoch':i+1,'t_loss':train_logs['custom_loss'],'t_dice':train_logs['dice'],'v_loss':valid_logs['custom_loss'],'v_dice':valid_logs['dice'],'v_jaccard':valid_logs['jaccard'],'t_jaccard':train_logs['jaccard']})
        # do something (save model, change lr, etc.)
        if max_dice <= valid_logs['dice']:
            max_dice = valid_logs['dice']
            max_jaccard = valid_logs['jaccard']
            wandb.config.update({'max_dice':max_dice, 'max_jaccard':max_jaccard}, allow_val_change=True)
            torch.save(model.state_dict(), './best_model.pth')
            print('Model saved!')
            counter = 0
        counter = counter+1

        if counter>10:
            break
         
    print(f'max dice: {max_dice} max jaccard: {max_jaccard}')

def train_model(configs):
    train(configs['epochs'], configs['batch_size'], configs['t1_dir'],configs['t1ce_dir'], configs['t2_dir'],configs['flair_dir'],configs['seg_dir'],
          configs['encoder'],configs['encoder_weights'], configs['device'], configs['lr'], configs['beta'], (configs['contrastive']), configs['compile'], configs['num_workers'], configs['checkpoint'])