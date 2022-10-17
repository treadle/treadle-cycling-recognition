# Cycling Recognition Project

Python version: `3.9.3`

The conda environment can be created and activated as follows:
```
$ conda env create -n cycling_rec -f environment.yml 
$ conda activate cycling_rec
```

In order to run sampling and splitting of data, use the following command:
```
python preprocessing.py --data_path ./data/ride_data/ --destination_path ./sampled_data/sec1_ovr0_test_volodymyr/ --test_sub volodymyr --get_norm_vals
```

The script splits data from subject `volodymyr` into the test set, and from all the remaining subjects into the train set. The time-windows are sampled into 1 second time-windows with no overlapping (default parameters). You can specify custom length of time window in seconds using `--len_tw_sec` argument and overlapping ratio could be set with `--overlap_ratio`. Argument `--get_norm_vals` computes and saves mean and std per channel based on the train set for further normalization of data.