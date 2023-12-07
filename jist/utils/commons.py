
import torch
import random
import numpy as np


class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iterator = super().__iter__()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


def delete_model_gradients(model):
    """Set gradients to None to free some GPU memory. Useful before inference.
    Note that using optimizer.zero_grad() sets them to 0, which keeps using GPU space.
    """
    for param in model.parameters():
        param.grad = None


def make_deterministic(seed=0):
    """Make results deterministic. If seed == -1, do not make deterministic.
        Running your script in a deterministic way might slow it down.
        Note that for some packages (eg: sklearn's PCA) this function is not enough.
    """
    seed = int(seed)
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
