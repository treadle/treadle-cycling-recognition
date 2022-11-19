import argparse
import json
import os

import numpy as np

from datasets.data_record import DataRecord, ORDER


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # data
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--destination_path', required=True)

    # splits
    parser.add_argument('--val_sub', nargs='+', help='subjects to be used for validation')
    parser.add_argument('--test_sub', nargs='+', required=True, help='subjects to be used for test')

    # sampling
    parser.add_argument('--len_tw_sec', type=int, default=1, help='length of time windows')
    parser.add_argument('--overlap_ratio', type=float, default=0, help='overlapping percentage')

    # normalization
    parser.add_argument('--get_norm_vals', default=True, action='store_true', help='flag for computing mean and std based on train data')


    return parser.parse_args()


def save_sampled_record(sampled_record, record_name, destination):
    """Saves sampled records
    """
    for i, rec in enumerate(sampled_record):
        np.save(os.path.join(destination, f'{record_name}_{i}.npy'), rec)


def sample_and_split(args):
    """Main function for sampling. Iterates through all records, sample them into short time-windows and split based on subjects; computes mean and std values on train data for further normalization.
    """
    if args.get_norm_vals:
        running_train_mean = []
        running_train_std = []

    records_dirs = [os.path.join(args.data_path, dir) for dir in os.listdir(args.data_path)]

    for dir in records_dirs:
        record_name = os.path.basename(os.path.normpath(dir))
        data_record = DataRecord(os.path.join(dir, f'{record_name}.csv'), os.path.join(dir, f'meta_{record_name}.csv'))
        sampled_record = data_record.sample(args.len_tw_sec, args.overlap_ratio)
        if data_record.subject in args.test_sub or data_record.label != 'cycle':
            save_sampled_record(sampled_record, record_name, os.path.join(args.destination_path, 'test/'))
        elif args.val_sub is not None and data_record.subject in args.val_sub:
            save_sampled_record(sampled_record, record_name, os.path.join(args.destination_path, 'val/'))
        else:
            if data_record.label == 'cycle':
                if args.get_norm_vals:
                    cur_mean, cur_std = data_record.get_mean_std()
                    running_train_mean.append(cur_mean)
                    running_train_std.append(cur_std)
                save_sampled_record(sampled_record, record_name, os.path.join(args.destination_path, 'train/'))

    if args.get_norm_vals:
        ovr_mean = np.stack(running_train_mean).mean(axis=0)
        ovr_std = np.stack(running_train_std).mean(axis=0)

        means = {}
        stds = {}

        for i, device in enumerate(ORDER):
            means[device] = ovr_mean[i]
            stds[device] = ovr_std[i]
            
            with open(os.path.join(args.destination_path, 'ovr_mean.json'), 'w') as fp:
                json.dump(means, fp)

            with open(os.path.join(args.destination_path, 'ovr_std.json'), 'w') as fp:
                json.dump(stds, fp)


        
        np.save(os.path.join(args.destination_path, 'ovr_mean.npy'), ovr_mean)
        np.save(os.path.join(args.destination_path, 'ovr_std.npy'), ovr_std)
    

def main():
    args = parse_arguments()
    os.makedirs(args.destination_path, exist_ok=True)
    os.makedirs(os.path.join(args.destination_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.destination_path, 'val'), exist_ok=True)
    os.makedirs(os.path.join(args.destination_path, 'test'), exist_ok=True)
    sample_and_split(args)


if __name__ == '__main__':
    main()