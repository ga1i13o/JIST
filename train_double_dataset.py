
import sys
import torch
import logging
import torchmetrics
from torch import nn
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import torchvision.transforms as T
from torch.utils.data.dataloader import DataLoader
from os.path import join
torch.backends.cudnn.benchmark= True  # Provides a speedup

# ours
from jist.datasets import (BaseDataset, TrainDataset, collate_fn,
                                CosplaceTrainDataset, TestDataset)
from jist import utils
from jist.models import JistModel
from jist import evals
from jist.utils import (parse_arguments, setup_logging, MarginCosineProduct,
                        configure_transform, delete_model_gradients,
                        InfiniteDataLoader, make_deterministic, move_to_device)

args = parse_arguments()
start_time = datetime.now()
output_folder = f"logs/{args.exp_name}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
setup_logging(output_folder)
make_deterministic(args.seed)
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {output_folder}")

# Model
model = JistModel(args, agg_type=args.aggregation_type)
logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

if args.resume_model != None:
    logging.debug(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)
    model.load_state_dict(model_state_dict)

model = model.to(args.device).train()

#### Optimizer
criterion = torch.nn.CrossEntropyLoss()
model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion_triplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")

#|||||||||||||||||||||||| Datasets 
#### Datasets cosplace
groups = [CosplaceTrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                       current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]
# Each group has its own classifier, which depends on the number of classes in the group
classifiers = [MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]
classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers]

logging.info(f"Using {len(groups)} groups")
logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")
logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

cp_val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.val_posDistThr)
logging.info(f"Validation set: {cp_val_ds}")

#### Datasets sequence
# get transform
meta = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
img_shape = (args.img_shape[0], args.img_shape[1])
transform = configure_transform(image_dim=img_shape, meta=meta)

logging.info("Loading train set...")
triplets_ds = TrainDataset(cities=args.city, dataset_folder=args.seq_dataset_path, split='train',
                            base_transform=transform, seq_len=args.seq_length,
                            pos_thresh=args.train_posDistThr, neg_thresh=args.negDistThr, infer_batch_size=args.infer_batch_size,
                            num_workers=args.num_workers, img_shape=args.img_shape,
                            cached_negatives=args.cached_negatives,
                            cached_queries=args.cached_queries, nNeg=args.nNeg)

logging.info(f"Train set: {triplets_ds}")
logging.info("Loading val set...")
val_ds = BaseDataset(dataset_folder=args.seq_dataset_path, split='val',
                     base_transform=transform, seq_len=args.seq_length,
                     pos_thresh=args.val_posDistThr)
logging.info(f"Val set: {val_ds}")

logging.info("Loading test set...")
test_ds = BaseDataset(dataset_folder=args.seq_dataset_path, split='test',
                      base_transform=transform, seq_len=args.seq_length,
                      pos_thresh=args.val_posDistThr)
logging.info(f"Test set: {test_ds}")


#### Resume
if args.resume_train:
    model, model_optimizer, classifiers, classifiers_optimizers, sequence_best_r5, start_epoch_num = \
        utils.cp_utils.resume_train(args, output_folder, model, model_optimizer, classifiers, classifiers_optimizers)

    epoch_num = start_epoch_num - 1
    best_val_recall1 = 0
    logging.info(f"Resuming from epoch {start_epoch_num} with best seq R@5 {sequence_best_r5:.1f} from checkpoint {args.resume_train}")
else:
    best_val_recall1 = start_epoch_num = sequence_best_r5 = 0

#### Train / evaluation loop
iterations_per_epoch = args.cached_queries // args.train_batch_size * args.num_sub_epochs
logging.info("Start training ...")
logging.info(f"There are {len(groups[0])} classes for the first group, " +
             f"each epoch has {iterations_per_epoch} iterations " +
             f"with batch_size {args.cp_batch_size}, therefore the model sees each class (on average) " +
             f"{iterations_per_epoch * args.cp_batch_size / len(groups[0]):.1f} times per epoch")
logging.info(f"Backbone output channels are {model.features_dim}, features descriptor dim is {model.fc_output_dim}, "
             f"sequence descriptor dim is {model.aggregation_dim}")

gpu_augmentation = T.Compose([
            utils.augmentations.DeviceAgnosticColorJitter(brightness=args.brightness, contrast=args.contrast,
                                                          saturation=args.saturation, hue=args.hue),
            utils.augmentations.DeviceAgnosticRandomResizedCrop([512, 512],
                                                                scale=[1-args.random_resized_crop, 1]),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

scaler = torch.cuda.amp.GradScaler()
for epoch_num in range(start_epoch_num, args.epochs_num):

    epoch_start_time = datetime.now()
    # Select classifier and dataloader according to epoch
    current_group_num = epoch_num % args.groups_num
    classifiers[current_group_num] = classifiers[current_group_num].to(args.device)
    move_to_device(classifiers_optimizers[current_group_num], args.device)
    dataloader = InfiniteDataLoader(groups[current_group_num], num_workers=args.num_workers, drop_last=True,
                                    batch_size=args.cp_batch_size, shuffle=True, pin_memory=(args.device == "cuda"))
    dataloader_iterator = iter(dataloader)
    model = model.train()

    sequence_mean_loss = torchmetrics.MeanMetric()
    cosplace_mean_loss = torchmetrics.MeanMetric()

    seq_epoch_losses = np.zeros((0, 1), dtype=np.float32)

    for num_sub_epoch in range(args.num_sub_epochs):
        logging.debug(f"Cache: {num_sub_epoch + 1} / {args.num_sub_epochs}")

        # creates triplets on the smaller cache set
        triplets_ds.compute_triplets(model)
        triplets_dl = DataLoader(dataset=triplets_ds, num_workers=args.num_workers,
                                 batch_size=args.train_batch_size,
                                 collate_fn=collate_fn,
                                 pin_memory=(args.device == "cuda"),
                                 drop_last=True)

        model = model.train()
        tqdm_bar = tqdm(triplets_dl, ncols=100)
        for images, _, _ in tqdm_bar:
            model_optimizer.zero_grad()
            classifiers_optimizers[current_group_num].zero_grad()

            if args.lambda_triplet != 0:
                #### ITERATION ON SEQUENCES
                # images shape: (bsz, seq_len*(nNeg + 2), 3, H, W)
                # triplets_local_indexes shape: (bsz, nNeg+2) -> contains -1 for query, 1 for pos, 0 for neg
                # reshape images to only have 4-d
                images = images.view(-1, 3, *img_shape)
                # features : (bsz*(nNeg+2), model_output_size)
                with torch.cuda.amp.autocast():
                    features = model(images.to(args.device))
                    features = model.aggregate(features)

                    # Compute loss by passing the triplets one by one
                    sequence_loss = 0
                    features = features.view(args.train_batch_size, -1, model.aggregation_dim)
                    for b in range(args.train_batch_size):
                        query = features[b:b + 1, 0]  # size (1, output_dim)
                        pos = features[b:b + 1, 1]  # size (1, output_dim)
                        negatives = features[b, 2:]  # size (nNeg, output_dim)
                        # negatives has 10 images , pos and query 1 but
                        # the loss yields same result as calling it 10 times
                        sequence_loss += criterion_triplet(query, pos, negatives)
                del images, features
                sequence_loss /= (args.train_batch_size * args.nNeg)
                sequence_loss *= args.lambda_triplet
                scaler.scale(sequence_loss).backward()
                sequence_mean_loss.update(sequence_loss.item())
                del sequence_loss
            else:
                sequence_mean_loss.update(-1)

            if args.lambda_im2im != 0:
                #### ITERATION ON COSPLACE
                images, targets, _ = next(dataloader_iterator)
                images, targets = images.to(args.device), targets.to(args.device)

                if args.augmentation_device == "cuda":
                    images = gpu_augmentation(images)

                with torch.cuda.amp.autocast():
                    descriptors = model(images)
                    output = classifiers[current_group_num](descriptors, targets)
                    cosplace_loss = criterion(output, targets)
                del output, images, descriptors, targets
                cosplace_loss *= args.lambda_im2im
                scaler.scale(cosplace_loss).backward()
                cosplace_mean_loss.update(cosplace_loss.item())
                del cosplace_loss
            else:
                cosplace_mean_loss.update(-1)

            scaler.step(model_optimizer)
            if args.lambda_im2im != 0:
                scaler.step(classifiers_optimizers[current_group_num])
            scaler.update()
            tqdm_bar.set_description(f"seq_loss: {sequence_mean_loss.compute():.4f} - cos_loss: {cosplace_mean_loss.compute():.2f}")

        logging.debug(f"Epoch[{epoch_num:02d}]({num_sub_epoch + 1}/{args.num_sub_epochs}): " +
                      f"epoch sequence loss = {sequence_mean_loss.compute():.4f} - " +
                      f"epoch cosplace loss = {cosplace_mean_loss.compute():.4f}")

    classifiers[current_group_num] = classifiers[current_group_num].cpu()
    move_to_device(classifiers_optimizers[current_group_num], "cpu")
    delete_model_gradients(model)

    #### Evaluation CosPlace
    cosplace_recalls, cosplace_recalls_str = evals.cosplace_test(args, cp_val_ds, model)
    logging.info(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, cosPlace {cp_val_ds}: {cosplace_recalls_str[:20]}")
    cosplace_is_best = cosplace_recalls[0] > best_val_recall1
    cosplace_best_val_recall1 = max(cosplace_recalls[0], best_val_recall1)

    #### Evaluation Sequence
    sequence_recalls, sequence_recalls_str = evals.test(args, val_ds, model)
    logging.info(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, sequence {val_ds}: {sequence_recalls_str}")
    sequence_is_best = sequence_recalls[1] > sequence_best_r5

    if sequence_is_best:
        logging.info(f"Improved: previous best R@5 = {sequence_best_r5:.1f}, current R@5 = {sequence_recalls[1]:.1f}")
        sequence_best_r5 = sequence_recalls[1]
    else:
        logging.info(f"Not improved: best R@5 = {sequence_best_r5:.1f}, current R@5 = {sequence_recalls[1]:.1f}")

    utils.cp_utils.save_checkpoint({
        "epoch_num": epoch_num + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model_optimizer.state_dict(),
        "classifiers_state_dict": [c.state_dict() for c in classifiers],
        "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
        "best_seq_val_recall5": sequence_best_r5
    }, sequence_is_best, output_folder)
    recalls, recalls_str = evals.test(args, test_ds, model)
    logging.info(f"Recalls on test set: {recalls_str}")

logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")
#### Test best model on test set
best_model_state_dict = torch.load(join(output_folder, "best_model.pth"))
model.load_state_dict(best_model_state_dict)

recalls, recalls_str = evals.test(args, test_ds, model)
logging.info(f"Recalls on test set: {recalls_str}")
