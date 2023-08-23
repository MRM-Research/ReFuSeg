import argparse
from utils.trainer import train_model
import wandb

def main(args):
    config = {
        't1_dir': args.t1_dir,
        't2_dir': args.t2_dir,
        't1ce_dir': args.t1ce_dir,
        'flair_dir': args.flair_dir,
        'seg_dir': args.seg_dir,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'device':args.device,
        'encoder': args.encoder,
        'encoder_weights': args.encoder_weights,
        'lr': args.lr,
        'beta': args.beta,
        'contrastive': args.contrastive,
        'compile': args.compile,
        'num_workers': args.num_workers,
        'checkpoint': args.checkpoint
    }
    wandb.init(project="BRATS CoReFusion", entity="kasliwal17",
               config={'model':'resnet34 depth4','beta':args.beta,
                'lr':args.lr, 'max_dice':0, 'max_jaccard':0, 'contrastive':args.contrastive, 'encoder':args.encoder, 'encoder_weights':args.encoder_weights, 'fusion_technique':'maximization'})
    #ssl_module._create_default_https_context = ssl_module.create_default_context(cafile=certifi.where())
    train_model(config)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--t1_dir', type=str, required=False, default='/Users/sankysagaram/MedSeg/Sample/t1')
    parser.add_argument('--t1ce_dir', type=str, required=False, default='/Users/sankysagaram/MedSeg/Sample/t1ce')
    parser.add_argument('--flair_dir', type=str, required=False, default='/Users/sankysagaram/MedSeg/Sample/flair')
    parser.add_argument('--t2_dir', type=str, required=False, default='/Users/sankysagaram/MedSeg/Sample/t2')
    parser.add_argument('--seg_dir', type=str, required=False, default='/Users/sankysagaram/MedSeg/Sample/mask')
    parser.add_argument('--batch_size', type=int, required=False, default=1)
    parser.add_argument('--epochs', type=int, required=False, default=250)
    parser.add_argument('--device', type=str, required=False, default='cuda')
    parser.add_argument('--encoder', type=str, required=False, default='resnet34')
    parser.add_argument('--encoder_weights', type=str, required=False, default='imagenet')
    parser.add_argument('--lr', type=float, required=False, default=1e-4)
    parser.add_argument('--beta', type=float, required=False, default=1)
    parser.add_argument('--contrastive', type=bool, required=False, default=False)
    parser.add_argument('--compile', type=bool, required=False, default=False)
    parser.add_argument('--num_workers', type=int, required=False, default=12)
    parser.add_argument('--checkpoint', type=str, required=False, default='')
    arguments = parser.parse_args()
    main(arguments)

