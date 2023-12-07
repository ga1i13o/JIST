import argparse
import torch
import os


def _get_device_count():
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return -1


def parse_arguments():
    parser = argparse.ArgumentParser(description="Sequence Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # reproducibility
    parser.add_argument('--seed', type=int, default=0)
    
    # dataset
    parser.add_argument("--city", type=str, default='', help='subset of cities from train set')
    parser.add_argument("--seq_length", type=int, default=5,
                        help="Number of images in each sequence")
    parser.add_argument("--reverse", action='store_true', default=False, help='reverse DB sequences frames')
    parser.add_argument("--cut_last_frame", action='store_true', default=False, help='cut last sequence frame')
    parser.add_argument("--val_posDistThr", type=int, default=25, help="_")
    parser.add_argument("--train_posDistThr", type=int, default=10, help="_")
    parser.add_argument("--negDistThr", type=int, default=25, help="_")
    parser.add_argument('--img_shape', type=int, default=[480, 640], nargs=2,
                        help="Resizing shape for images (HxW).")

    # about triplets and mining
    parser.add_argument("--nNeg", type=int, default=5,
                        help="How many negatives to consider per each query in the loss")
    parser.add_argument("--cached_negatives", type=int, default=3000,
                        help="How many negatives to use to compute the hardest ones")
    parser.add_argument("--cached_queries", type=int, default=1000,
                        help="How many queries to keep cached")
    parser.add_argument("--queries_per_epoch", type=int, default=5000,
                        help="How many queries to consider for one epoch. Must be multiple of cached_queries")

    # models
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")
    parser.add_argument("--pretrain_model", type=str, default=None,
                        help="Path to load pretrained model from.")

    # training pars
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=8,
                        help="num_workers for all dataloaders")

    parser.add_argument("--num_sub_epochs", type=int, default=10,
                        help="How many times to recompute cache per epoch.")
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Number of triplets: (query + pos + negs) * seq_length.")
    parser.add_argument("--infer_batch_size", type=int, default=8,
                        help="Batch size for inference (caching and testing)")

    parser.add_argument("--epochs_num", type=int, default=5,
                        help="number of epochs to train for")

    parser.add_argument("--margin", type=float, default=0.1,
                        help="margin for the triplet loss")
    parser.add_argument("--lambda_triplet", type=float, default=10000, help="weight to triplet loss")
    parser.add_argument("--lambda_im2im", type=float, default=100, help="weight to cosplace loss")
    
    # PATHS
    parser.add_argument("--seq_dataset_path", type=str, required=True,
                        help="Path of the seq2seq dataset")
    parser.add_argument("--dataset_folder", type=str, required=True, # should end in 'sf_xl/processed"
                        help="path of the SF-XL processed folder with train/val/test sets")
    parser.add_argument("--exp_name", type=str, default="default",
                        help="Folder name of the current run (saved in ./runs/)")

    # CosPlace Groups parameters
    parser.add_argument("--M", type=int, default=10, help="_")
    parser.add_argument("--alpha", type=int, default=30, help="_")
    parser.add_argument("--N", type=int, default=5, help="_")
    parser.add_argument("--L", type=int, default=2, help="_")
    parser.add_argument("--groups_num", type=int, default=8, help="_")
    parser.add_argument("--min_images_per_class", type=int, default=10, help="_")
    # Model parameters
    parser.add_argument("--backbone", type=str, default="ResNet18",
                        choices=["ResNet18", "ResNet50", "ResNet101", "VGG16"], help="_")
    parser.add_argument("--aggregation_type", type=str, default="seqgem",
                        choices=["concat", "mean", "max", "simplefc", "conv1d", "meanfc", "seqgem"], help="_")
    parser.add_argument("--fc_output_dim", type=int, default=512,
                        help="Output dimension of final fully connected layer")
    parser.add_argument("--augmentation_device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="on which device to run data augmentation")
    parser.add_argument("--cp_batch_size", type=int, default=64, help="_")
    parser.add_argument("--lr", type=float, default=0.00001, help="_")
    parser.add_argument("--classifiers_lr", type=float, default=0.01, help="_")
    # Data augmentation
    parser.add_argument("--brightness", type=float, default=0.7, help="_")
    parser.add_argument("--contrast", type=float, default=0.7, help="_")
    parser.add_argument("--hue", type=float, default=0.5, help="_")
    parser.add_argument("--saturation", type=float, default=0.7, help="_")
    parser.add_argument("--random_resized_crop", type=float, default=0.5, help="_")
    # Resume parameters
    parser.add_argument("--resume_train", type=str, default=None,
                        help="path to checkpoint to resume, e.g. logs/.../last_checkpoint.pth")
    parser.add_argument("--resume_model", type=str, default=None,
                        help="path to model to resume, e.g. logs/.../best_model.pth")
    
    args = parser.parse_args()

    args.train_set_folder = os.path.join(args.dataset_folder, "train")
    if not os.path.exists(args.train_set_folder):
        raise FileNotFoundError(f"Folder {args.train_set_folder} does not exist")
    
    args.val_set_folder = os.path.join(args.dataset_folder, "val")
    if not os.path.exists(args.val_set_folder):
        raise FileNotFoundError(f"Folder {args.val_set_folder} does not exist")
    
    args.test_set_folder = os.path.join(args.dataset_folder, "test")
    if not os.path.exists(args.test_set_folder):
        raise FileNotFoundError(f"Folder {args.test_set_folder} does not exist")
            
    if args.queries_per_epoch % args.cached_queries != 0:
        raise ValueError("Please ensure that queries_per_epoch is divisible by cache_refresh_rate, " +
                         f"because {args.queries_per_epoch} is not divisible by {args.cached_queries}")
    
    return args

