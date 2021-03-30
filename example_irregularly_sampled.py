import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Run on CPU

import tensorflow as tf
import argparse


class_map = {
    'lying down': 0,
    'lying': 0,
    'sitting down': 1,
    'sitting': 1,
    'standing up from lying': 2,
    'standing up from sitting': 2,
    'standing up from sitting on the ground': 2,
    "walking": 3,
    "falling": 4,
    'on all fours': 5,
    'sitting on the ground': 6,
}

sensor_ids = {
    "010-000-024-033": 0,
    "010-000-030-096": 1,
    "020-000-033-111": 2,
    "020-000-032-221": 3
}


def one_hot(x, n):
    y = np.zeros(n, dtype=np.float32)
    y[x] = 1
    return y


def load_crappy_formated_csv():
    all_x = []
    all_y = []
    all_t = []

    series_x = []
    series_t = []
    series_y = []

    all_feats = []
    all_elapsed = []
    last_millis = None
    if (not os.path.isfile("data/person/ConfLongDemo_JSI.txt")):
        print("ERROR: File 'data/person/ConfLongDemo_JSI.txt' not found")
        print("Please execute the command")
        print("source download_dataset.sh")
        import sys
        sys.exit(-1)
    with open("data/person/ConfLongDemo_JSI.txt", "r") as f:
        current_person = "A01"

        for line in f:
            arr = line.split(",")
            if (len(arr) < 6):
                break
            if (arr[0] != current_person):
                # Enque and reset
                series_x = np.stack(series_x, axis=0)
                series_t = np.stack(series_t, axis=0)
                series_y = np.array(series_y, dtype=np.int32)
                all_x.append(series_x)
                all_t.append(series_t)
                all_y.append(series_y)
                last_millis = None
                series_x = []
                series_y = []
                series_t = []

            millis = np.int64(arr[2]) / (100 * 1000)
            # 100ms will be normalized to 1.0
            millis_mapped_to_1 = 10.0
            if (last_millis is None):
                elasped_sec = 0.05
            else:
                elasped_sec = float(millis - last_millis) / 1000.0
            elasped = elasped_sec * 1000 / millis_mapped_to_1

            last_millis = millis
            all_elapsed.append(elasped)
            current_person = arr[0]
            sensor_id = sensor_ids[arr[1]]
            label_col = class_map[arr[7].replace("\n", "")]
            feature_col_2 = np.array(arr[4:7], dtype=np.float32)
            # Last 3 entries of the feature vector contain sensor value

            # First 4 entries of the feature vector contain sensor ID
            feature_col_1 = np.zeros(4, dtype=np.float32)
            feature_col_1[sensor_id] = 1

            feature_col = np.concatenate([feature_col_1, feature_col_2])
            series_x.append(feature_col)
            series_t.append(elasped)
            all_feats.append(feature_col)
            series_y.append(label_col)

    all_feats = np.stack(all_feats, axis=0)
    print("all_feats.shape: ", str(all_feats.shape))
    all_elapsed = np.stack(all_elapsed, axis=0)
    print("all_elapsed.mean", str(np.mean(all_elapsed)))
    print("all_elapsed.med ", str(np.median(all_elapsed)))

    # No normalization
    # all_mean = np.mean(all_feats,axis=0)
    # all_std = np.std(all_feats,axis=0)
    # all_mean[3:] = 0
    # all_std[3:] = 1
    # print("all_mean: ",str(all_mean))
    # print("all_std: ",str(all_std))
    # for i in range(len(all_x)):
    #     all_x[i] -= all_mean
    #     all_x[i] /= all_std

    return all_x, all_t, all_y


def cut_in_sequences(all_x, all_t, all_y, seq_len, inc=1):
    sequences_x = []
    sequences_t = []
    sequences_y = []

    for i in range(len(all_x)):
        x, t, y = all_x[i], all_t[i], all_y[i]

        for s in range(0, x.shape[0] - seq_len, inc):
            start = s
            end = start + seq_len
            sequences_x.append(x[start:end])
            sequences_t.append(t[start:end])
            sequences_y.append(y[start:end])

    return np.stack(sequences_x, axis=0), np.stack(sequences_t, axis=0).reshape([-1, seq_len, 1]), np.stack(sequences_y,
                                                                                                            axis=0)


class PersonData:

    def __init__(self, seq_len=32):
        self.seq_len = seq_len
        self.num_classes = 7
        all_x, all_t, all_y = load_crappy_formated_csv()
        all_x, all_t, all_y = cut_in_sequences(all_x, all_t, all_y, seq_len=seq_len, inc=seq_len // 2)

        print("all_x.shape: ", str(all_x.shape))
        print("all_t.shape: ", str(all_t.shape))
        print("all_y.shape: ", str(all_y.shape))
        total_seqs = all_x.shape[0]
        print("Total number of sequences: {}".format(total_seqs))
        permutation = np.random.RandomState(98841).permutation(total_seqs)
        test_size = int(0.2 * total_seqs)

        self.test_x = all_x[permutation[:test_size]]
        self.test_y = all_y[permutation[:test_size]]
        self.test_t = all_t[permutation[:test_size]]
        self.train_x = all_x[permutation[test_size:]]
        self.train_t = all_t[permutation[test_size:]]
        self.train_y = all_y[permutation[test_size:]]

        self.feature_size = int(self.train_x.shape[-1])

        print("train_x.shape: ", str(self.train_x.shape))
        print("train_t.shape: ", str(self.train_t.shape))
        print("train_y.shape: ", str(self.train_y.shape))
        print("Total number of train sequences: {}".format(self.train_x.shape[0]))
        print("Total number of test  sequences: {}".format(self.test_x.shape[0]))


data = PersonData()