import argparse
from utils.predict import predict

def main(args):
    config = {
        't1_dir': args.t1_dir,
        't2_dir': args.t2_dir,
        't1ce_dir': args.t1ce_dir,
        'flair_dir': args.flair_dir,
        'batch_size': args.batch_size,
        'device':args.device,
        'encoder': args.encoder,
        'compile': args.compile,
        'num_workers': args.num_workers,
        'model_path': args.model_path,
        'save_path': args.save_path
    }
    predict(config)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--t1_dir', type=str, required=False, default='/Users/sankysagaram/MedSeg/Sample/t1')
    parser.add_argument('--t1ce_dir', type=str, required=False, default='/Users/sankysagaram/MedSeg/Sample/t1ce')
    parser.add_argument('--flair_dir', type=str, required=False, default='/Users/sankysagaram/MedSeg/Sample/flair')
    parser.add_argument('--t2_dir', type=str, required=False, default='/Users/sankysagaram/MedSeg/Sample/t2')
    parser.add_argument('--batch_size', type=int, required=False, default=1)
    parser.add_argument('--device', type=str, required=False, default='cuda')
    parser.add_argument('--encoder', type=str, required=False, default='resnet34')
    parser.add_argument('--compile', type=bool, required=False, default=False)
    parser.add_argument('--num_workers', type=int, required=False, default=12)
    parser.add_argument('--model_path', type=str, required=True, default='./best_model.pth')
    parser.add_argument('--save_path', type=str, required=True, default='./predicted_masks/')
    parser.add_argument('--contrastive', type=bool, required=False, default=False)
    arguments = parser.parse_args()
    main(arguments)

