import os
import torch
import faiss
import logging
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from os.path import join
import torch.utils.data as data
from torch.utils.data.dataset import Subset
from sklearn.neighbors import NearestNeighbors
from torch.utils.data.dataloader import DataLoader

from jist.utils import RAMEfficient2DMatrix


def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (images,
        triplets_local_indexes, triplets_global_indexes).
        triplets_local_indexes are the indexes referring to each triplet within images.
        triplets_global_indexes are the global indexes of each image.
    Args:
        batch: list of tuple (images, triplets_local_indexes, triplets_global_indexes).
            considering each query to have 10 negatives (negs_num_per_query=10):
            - images: torch tensor of shape (12, 3, h, w).
            - triplets_local_indexes: torch tensor of shape (10, 3).
            - triplets_global_indexes: torch tensor of shape (12).
    Returns:
        images: torch tensor of shape (batch_size*12, 3, h, w).
        triplets_local_indexes: torch tensor of shape (batch_size*10, 3).
        triplets_global_indexes: torch tensor of shape (batch_size, 12).
    """
    images = torch.cat([e[0] for e in batch])
    triplets_local_indexes = torch.cat([e[1][None] for e in batch])
    triplets_global_indexes = torch.cat([e[2][None] for e in batch])
    for i, (local_indexes, global_indexes) in enumerate(zip(triplets_local_indexes, triplets_global_indexes)):
        local_indexes += len(global_indexes) * i  # Increment local indexes by offset (len(global_indexes) is 12)
    return images, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes


class BaseDataset(data.Dataset):
    def __init__(self, cities='', dataset_folder="datasets", split="train", base_transform=None,
                 seq_len=3, pos_thresh=25, neg_thresh=25, reverse_frames=False):
        super().__init__()
        self.dataset_folder = join(dataset_folder, split)
        self.seq_len = seq_len

        # the algorithm to create sequences works with odd-numbered sequences only
        cut_last_frame = ((seq_len % 2) == 0) 
        if cut_last_frame:
            self.seq_len += 1
        self.base_transform = base_transform

        if not os.path.exists(self.dataset_folder): raise FileNotFoundError(
            f"Folder {self.dataset_folder} does not exist")

        self.init_data(cities, split, pos_thresh, neg_thresh)
        if reverse_frames:
            self.db_paths = [",".join(path.split(',')[::-1]) for path in self.db_paths]
        self.images_paths = self.db_paths + self.q_paths
        self.database_num = len(self.db_paths)
        self.queries_num = len(self.qIdx)

        if cut_last_frame:
            self.__cut_last_frame()

    def init_data(self, cities, split, pos_thresh, neg_thresh):
        if ('msls' in self.dataset_folder) and (split == 'train'):
            os.makedirs('cache', exist_ok=True)
            cache_file = f'cache/msls_seq{self.seq_len}_{cities}.torch'
            if os.path.isfile(cache_file):
                logging.info(f'Loading cached data from {cache_file}...')
                cache_dict = torch.load(cache_file)
                self.__dict__.update(cache_dict)
                return
            else:
                logging.info('Data structures were not cached, building them now...')
        #### Read paths and UTM coordinates for all images.
        database_folder = join(self.dataset_folder, "database")
        queries_folder = join(self.dataset_folder, "queries")

        cities = cities
        self.db_paths, all_db_paths, db_idx_frame_to_seq = build_sequences(database_folder,
                                                                           seq_len=self.seq_len, cities=cities,
                                                                           desc='loading database...')
        self.q_paths, all_q_paths, q_idx_frame_to_seq = build_sequences(queries_folder,
                                                                        seq_len=self.seq_len, cities=cities,
                                                                        desc='loading queries...')

        q_unique_idxs = np.unique([idx for seq_frames_idx in q_idx_frame_to_seq for idx in seq_frames_idx])
        db_unique_idxs = np.unique([idx for seq_frames_idx in db_idx_frame_to_seq for idx in seq_frames_idx])

        self.database_utms = np.array(
            [(path.split("@")[1], path.split("@")[2]) for path in all_db_paths[db_unique_idxs]]).astype(np.float64)
        self.queries_utms = np.array(
            [(path.split("@")[1], path.split("@")[2]) for path in all_q_paths[q_unique_idxs]]).astype(
            np.float64)

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.hard_positives_per_query = knn.radius_neighbors(self.queries_utms,
                                                             radius=pos_thresh,
                                                             return_distance=False)
        if split == 'train':
            # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 25 meters)
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.database_utms)
            self.soft_positives_per_query = knn.radius_neighbors(self.queries_utms,
                                                                 radius=neg_thresh,
                                                                 return_distance=False)
        self.qIdx = []
        self.pIdx = []
        self.nonNegIdx = []
        self.q_without_pos = 0
        for q in tqdm(range(len(q_idx_frame_to_seq)), ncols=100, desc='Finding positives and negatives...'):
            q_frame_idxs = q_idx_frame_to_seq[q]
            unique_q_frame_idxs = np.where(np.in1d(q_unique_idxs, q_frame_idxs))

            p_uniq_frame_idxs = np.unique(
                [p for pos in self.hard_positives_per_query[unique_q_frame_idxs] for p in pos])

            if len(p_uniq_frame_idxs) > 0:
                # p_seq_idx = np.where(np.in1d(db_unique_idxs, p_uniq_frame_idxs))[0]
                p_seq_idx = np.where(np.in1d(db_idx_frame_to_seq, db_unique_idxs[p_uniq_frame_idxs])
                              .reshape(db_idx_frame_to_seq.shape))[0]

                self.qIdx.append(q)
                self.pIdx.append(np.unique(p_seq_idx))

                if split == 'train':
                    nonNeg_uniq_frame_idxs = np.unique(
                        [p for pos in self.soft_positives_per_query[unique_q_frame_idxs] for p in pos])
                    #nonNeg_seq_idx = np.where(np.in1d(db_unique_idxs, nonNeg_uniq_frame_idxs))
                    #self.nonNegIdx.append(nonNeg_seq_idx)
                    nonNeg_seq_idx = np.where(np.in1d(db_idx_frame_to_seq, db_unique_idxs[nonNeg_uniq_frame_idxs])
                              .reshape(db_idx_frame_to_seq.shape))[0]
                    self.nonNegIdx.append(np.unique(nonNeg_seq_idx))
            else:
                self.q_without_pos += 1
        
        self.qIdx = np.array(self.qIdx)
        self.pIdx = np.array(self.pIdx, dtype=object)
        if split == 'train':
            save_dict = {
                'db_paths': self.db_paths,
                'q_paths': self.q_paths,
                'database_utms': self.database_utms,
                'queries_utms': self.queries_utms, 
                'hard_positives_per_query': self.hard_positives_per_query, 
                'soft_positives_per_query': self.soft_positives_per_query, 
                'qIdx': self.qIdx, 
                'pIdx': self.pIdx,
                'nonNegIdx': self.nonNegIdx, 
                'q_without_pos': self.q_without_pos
            }
            torch.save(save_dict, cache_file)

    def __cut_last_frame(self):
        for i, seq in enumerate(self.images_paths):
            self.images_paths[i] = ','.join((seq.split(',')[:-1]))
        for i, seq in enumerate(self.db_paths):
            self.db_paths[i] = ','.join((seq.split(',')[:-1]))
        for i, seq in enumerate(self.q_paths):
            self.q_paths[i] = ','.join((seq.split(',')[:-1]))

    def __getitem__(self, index):
        old_index = index
        if index >= self.database_num:
            q_index = index - self.database_num
            index = self.qIdx[q_index] + self.database_num

        img = torch.stack([self.base_transform(Image.open(join(self.dataset_folder, im))) for im in self.images_paths[index].split(',')])

        return img, index, old_index

    def __len__(self):
        return len(self.images_paths)

    def __repr__(self):
        return (
            f"< {self.__class__.__name__}, ' #database: {self.database_num}; #queries: {self.queries_num} >")

    def get_positives(self):
        return self.pIdx


def filter_by_cities(x, cities):
    for city in cities:
        if x.find(city) > 0:
            return True
    return False


def build_sequences(folder, seq_len=3, cities='', desc='loading'):
    if cities != '':
        if not isinstance(cities, list):
            cities = [cities]
    base_path = os.path.dirname(folder)
    paths = []
    all_paths = []
    idx_frame_to_seq = []
    seqs_folders = sorted(glob(join(folder, '*'), recursive=True))
    for seq in tqdm(seqs_folders, ncols=100, desc=desc):
        start_index = len(all_paths)
        frame_nums = np.array(list(map(lambda x: int(x.split('@')[4]), sorted(glob(join(seq, '*'))))))
        # seq_paths = np.array(sorted(glob(join(seq, '*'))))
        # read the full paths, then keep only relative ones. allows caching only rel. paths
        full_seq_paths = sorted(glob(join(seq, '*')))
        seq_paths = np.array([s_p.replace(f'{base_path}/', '') for s_p in full_seq_paths])
        
        if cities != '':
            sample_path = seq_paths[0]
            if not filter_by_cities(sample_path, cities):
                continue

        sorted_idx_frames = np.argsort(frame_nums)
        all_paths += list(seq_paths[sorted_idx_frames])
        for idx, frame_num in enumerate(frame_nums):
            if idx < (seq_len // 2) or idx >= (len(frame_nums) - seq_len // 2): continue
            # find surrounding frames in sequence
            seq_idx = np.arange(-seq_len // 2, seq_len // 2) + 1 + idx
            if (np.diff(frame_nums[sorted_idx_frames][seq_idx]) == 1).all():
                paths.append(",".join(seq_paths[sorted_idx_frames][seq_idx]))
                idx_frame_to_seq.append(seq_idx + start_index)

    return paths, np.array(all_paths), np.array(idx_frame_to_seq)


class TrainDataset(BaseDataset):
    def __init__(self, cities='', dataset_folder="datasets", split="train", base_transform=None,
                 seq_len=3, pos_thresh=25, neg_thresh=25, infer_batch_size=8,
                 num_workers=3, img_shape=(480, 640), 
                 cached_negatives=1000, cached_queries=1000, nNeg=10):
        super().__init__(dataset_folder=dataset_folder, split=split, cities=cities, base_transform=base_transform,
                         seq_len=seq_len, pos_thresh=pos_thresh, neg_thresh=neg_thresh)
        self.cached_negatives = cached_negatives  # Number of negatives to randomly sample
        self.num_workers = num_workers
        self.cached_queries = cached_queries
        self.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
        self.bs = infer_batch_size
        self.img_shape = img_shape
        self.nNeg = nNeg  # Number of negatives per query in each batch
        self.is_inference = False
        self.query_transform = self.base_transform

    def __getitem__(self, index):
        if self.is_inference:
            # At inference time return the single image. This is used for caching or computing NetVLAD's clusters
            return super().__getitem__(index)
        query_index, best_positive_index, neg_indexes = torch.split(self.triplets_global_indexes[index],
                                                                    (1, 1, self.nNeg))

        query = torch.stack(
            [self.base_transform(Image.open(join(self.dataset_folder, im))) for im in self.q_paths[query_index].split(',')])

        positive = torch.stack(
            [self.base_transform(Image.open(join(self.dataset_folder, im))) for im in self.db_paths[best_positive_index].split(',')])

        negatives = [torch.stack([self.base_transform(Image.open(join(self.dataset_folder, im)))for im in self.db_paths[idx].split(',')])
                        for idx in neg_indexes]

        images = torch.stack((query, positive, *negatives), 0)
        triplets_local_indexes = torch.empty((0, 3), dtype=torch.int)
        for neg_num in range(len(neg_indexes)):
            triplets_local_indexes = torch.cat(
                (triplets_local_indexes, torch.tensor([0, 1, 2 + neg_num]).reshape(1, 3)))
        return images, triplets_local_indexes, self.triplets_global_indexes[index]

    def __len__(self):
        if self.is_inference:
            # At inference time return the number of images. This is used for caching or computing NetVLAD's clusters
            return super().__len__()
        else:
            return len(self.triplets_global_indexes)

    def compute_triplets(self, model):
        self.is_inference = True
        self.compute_triplets_partial(model)
        self.is_inference = False

    def compute_cache(self, model, subset_ds, cache_shape):
        subset_dl = DataLoader(dataset=subset_ds, num_workers=self.num_workers,
                               batch_size=self.bs, shuffle=False,
                               pin_memory=(self.device == "cuda"))
        model = model.eval()
        cache = RAMEfficient2DMatrix(cache_shape, dtype=np.float32)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for images, indexes, _ in tqdm(subset_dl, desc="cache feat extraction", ncols=100):
                    images = images.view(-1, 3, self.img_shape[0], self.img_shape[1])
                    frames_features = model(images.to(self.device))
                    aggregated_features = model.aggregate(frames_features)
                    cache[indexes.numpy()] = aggregated_features.cpu().numpy()
        return cache

    def get_best_positive_index(self, qidx, cache, query_features):
        positives_features = cache[self.pIdx[qidx]]
        faiss_index = faiss.IndexFlatL2(len(query_features))
        faiss_index.add(positives_features)
        # Search the best positive (within 10 meters AND nearest in features space)
        _, best_positive_num = faiss_index.search(query_features.reshape(1, -1), 1)
        best_positive_index = self.pIdx[qidx][best_positive_num[0]]
        return best_positive_index

    def get_hardest_negatives_indexes(self, cache, query_features, neg_indexes):
        neg_features = cache[neg_indexes]

        faiss_index = faiss.IndexFlatL2(len(query_features))
        faiss_index.add(neg_features)

        _, neg_nums = faiss_index.search(query_features.reshape(1, -1), self.nNeg)
        neg_nums = neg_nums.reshape(-1)
        neg_idxs = neg_indexes[neg_nums].astype(np.int32)

        return neg_idxs

    def compute_triplets_partial(self, model):
        self.triplets_global_indexes = []
        # Take 1000 random queries
        sampled_queries_indexes = np.random.choice(self.queries_num, self.cached_queries, replace=False)
        # Sample 1000 random database images for the negatives
        sampled_database_indexes = np.random.choice(self.database_num, self.cached_negatives, replace=False)

        positives_indexes = np.unique([idx for db_idx in self.pIdx[sampled_queries_indexes] for idx in db_idx])
        database_indexes = list(sampled_database_indexes) + list(positives_indexes)
        subset_ds = Subset(self, database_indexes + list(sampled_queries_indexes + self.database_num))
        cache = self.compute_cache(model, subset_ds, cache_shape=[len(self), model.aggregation_dim])

        for q in tqdm(sampled_queries_indexes, desc="computing hard negatives", ncols=100):
            qidx = self.qIdx[q] + self.database_num
            query_features = cache[qidx]

            best_positive_index = self.get_best_positive_index(q, cache, query_features)
            if isinstance(best_positive_index, np.ndarray):
                best_positive_index = best_positive_index[0]
            # Choose the hardest negatives within sampled_database_indexes, ensuring that there are no positives
            soft_positives = self.nonNegIdx[q]
            neg_indexes = np.setdiff1d(sampled_database_indexes, soft_positives, assume_unique=True)
            # Take all database images that are negatives and are within the sampled database images
            neg_indexes = self.get_hardest_negatives_indexes(cache, query_features, neg_indexes)
            self.triplets_global_indexes.append((self.qIdx[q], best_positive_index, *neg_indexes))

        # self.triplets_global_indexes is a tensor of shape [1000, 12]
        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes)

