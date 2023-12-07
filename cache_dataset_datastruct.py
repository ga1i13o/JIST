import argparse
from jist.datasets import TrainDataset


def main(args):
    folder = args.msls_folder
    seq_len = args.seq_len
    
    print(f'Caching dataset with seq len {seq_len}')
    triplets_ds = TrainDataset(cities='', dataset_folder=folder, split='train',
                            seq_len=seq_len, pos_thresh=10, neg_thresh=25)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sequence Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--msls_folder', type=str, required=True)
    parser.add_argument('--seq_len', type=int, default=5)
    args = parser.parse_args()
    
    main(args)