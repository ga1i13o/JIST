import torch
import logging
import numpy as np
from tqdm import tqdm
import faiss
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset


def test(args, eval_ds, model):
    model = model.eval()

    query_num = eval_ds.queries_num
    gallery_num = eval_ds.database_num
    all_features = np.empty((query_num + gallery_num, model.aggregation_dim), dtype=np.float32)

    with torch.no_grad():
        logging.debug("Extracting gallery features for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=4,
                                         batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))

        for images, indices, _ in tqdm(database_dataloader, ncols=100):
            images = images.contiguous().view(-1, 3, args.img_shape[0], args.img_shape[1])
            frames_features = model(images.to(args.device))
            aggregated_features = model.aggregate(frames_features)
            all_features[indices.numpy(), :] = aggregated_features.cpu().numpy()

        logging.debug("Extracting queries features for evaluation/testing")
        queries_subset_ds = Subset(eval_ds,
                                   list(range(eval_ds.database_num, eval_ds.database_num + eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=4,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))

        for images, _, indices in tqdm(queries_dataloader, ncols=100):
            images = images.contiguous().view(-1, 3, args.img_shape[0], args.img_shape[1])
            frames_features = model(images.to(args.device))
            aggregated_features = model.aggregate(frames_features)
            all_features[indices.numpy(), :] = aggregated_features.cpu().numpy()

    torch.cuda.empty_cache()
    queries_features = all_features[eval_ds.database_num:]
    gallery_features = all_features[:eval_ds.database_num]

    faiss_index = faiss.IndexFlatL2(model.aggregation_dim)
    faiss_index.add(gallery_features)

    logging.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_features, 10)

    # For each query, check if the predictions are correct
    positives_per_query = eval_ds.pIdx
    recall_values = [1, 5, 10]  # recall@1, recall@5, recall@10
    recalls = np.zeros(len(recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / len(eval_ds.qIdx) * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(recall_values, recalls)])
    return recalls, recalls_str
