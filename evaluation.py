import logging
from datetime import datetime
import torch

from jist.datasets import BaseDataset
from jist import utils, evals
from jist.models import JistModel


def evaluation(args):
    start_time = datetime.now()
    args.output_folder = f"test/{args.exp_name}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    utils.setup_logging(args.output_folder, console="info")
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.output_folder}")

    ### Definition of the model
    model = JistModel(args, agg_type=args.aggregation_type)

    if args.resume_model != None:
        logging.debug(f"Loading model from {args.resume_model}")
        model_state_dict = torch.load(args.resume_model)
        model.load_state_dict(model_state_dict)

    model = model.to(args.device)

    meta = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    img_shape = (args.img_shape[0], args.img_shape[1])
    transform = utils.configure_transform(image_dim=img_shape, meta=meta)

    eval_ds = BaseDataset(dataset_folder=args.seq_dataset_path, split='test',
                          base_transform=transform, seq_len=args.seq_length,
                          pos_thresh=args.val_posDistThr, reverse_frames=args.reverse)
    logging.info(f"Test set: {eval_ds}")

    logging.info(f"Backbone output channels are {model.features_dim}, features descriptor dim is {model.fc_output_dim}, "
             f"sequence descriptor dim is {model.aggregation_dim}")

    _, recalls_str = evals.test(args, eval_ds, model)
    logging.info(f"Recalls on test set: {recalls_str}")
    logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")


if __name__ == "__main__":
    args = utils.parse_arguments()
    evaluation(args)
