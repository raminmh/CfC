###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Authors: Yulia Rubanova and Ricky Chen
###########################

import os

from sklearn import model_selection

import duv_utils as utils
import numpy as np
import tarfile
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url

# Adapted from: https://github.com/rtqichen/time-series-datasets


class PersonActivity(object):
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00196/ConfLongDemo_JSI.txt",
    ]

    tag_ids = [
        "010-000-024-033",  # "ANKLE_LEFT",
        "010-000-030-096",  # "ANKLE_RIGHT",
        "020-000-033-111",  # "CHEST",
        "020-000-032-221",  # "BELT"
    ]

    tag_dict = {k: i for i, k in enumerate(tag_ids)}

    label_names = [
        "walking",
        "falling",
        "lying down",
        "lying",
        "sitting down",
        "sitting",
        "standing up from lying",
        "on all fours",
        "sitting on the ground",
        "standing up from sitting",
        "standing up from sit on grnd",
    ]

    # label_dict = {k: i for i, k in enumerate(label_names)}

    # Merge similar labels into one class
    label_dict = {
        "walking": 0,
        "falling": 1,
        "lying": 2,
        "lying down": 2,
        "sitting": 3,
        "sitting down": 3,
        "standing up from lying": 4,
        "standing up from sitting": 4,
        "standing up from sit on grnd": 4,
        "on all fours": 5,
        "sitting on the ground": 6,
    }

    def __init__(
        self,
        root,
        download=False,
        reduce="average",
        max_seq_length=50,
        n_samples=None,
        device=torch.device("cpu"),
    ):

        self.root = root
        self.reduce = reduce
        self.max_seq_length = max_seq_length

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        if device == torch.device("cpu"):
            self.data = torch.load(
                os.path.join(self.processed_folder, self.data_file), map_location="cpu"
            )
        else:
            self.data = torch.load(os.path.join(self.processed_folder, self.data_file))

        if n_samples is not None:
            self.data = self.data[:n_samples]

    def download(self):
        if self._check_exists():
            return

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        def save_record(records, record_id, tt, vals, mask, labels):
            tt = torch.tensor(tt).to(self.device)

            vals = torch.stack(vals)
            mask = torch.stack(mask)
            labels = torch.stack(labels)

            # flatten the measurements for different tags
            vals = vals.reshape(vals.size(0), -1)
            mask = mask.reshape(mask.size(0), -1)
            assert len(tt) == vals.size(0)
            assert mask.size(0) == vals.size(0)
            assert labels.size(0) == vals.size(0)

            # records.append((record_id, tt, vals, mask, labels))

            seq_length = len(tt)
            # split the long time series into smaller ones
            offset = 0
            slide = self.max_seq_length // 2

            while offset + self.max_seq_length < seq_length:
                idx = range(offset, offset + self.max_seq_length)

                first_tp = tt[idx][0]
                records.append(
                    (record_id, tt[idx] - first_tp, vals[idx], mask[idx], labels[idx])
                )
                offset += slide

        for url in self.urls:
            filename = url.rpartition("/")[2]
            download_url(url, self.raw_folder, filename, None)

            print("Processing {}...".format(filename))

            dirname = os.path.join(self.raw_folder)
            records = []
            first_tp = None

            for txtfile in os.listdir(dirname):
                with open(os.path.join(dirname, txtfile)) as f:
                    lines = f.readlines()
                    prev_time = -1
                    tt = []

                    record_id = None
                    for l in lines:
                        (
                            cur_record_id,
                            tag_id,
                            time,
                            date,
                            val1,
                            val2,
                            val3,
                            label,
                        ) = l.strip().split(",")
                        value_vec = torch.Tensor(
                            (float(val1), float(val2), float(val3))
                        ).to(self.device)
                        time = float(time)

                        if cur_record_id != record_id:
                            if record_id is not None:
                                save_record(records, record_id, tt, vals, mask, labels)
                            tt, vals, mask, nobs, labels = [], [], [], [], []
                            record_id = cur_record_id

                            tt = [torch.zeros(1).to(self.device)]
                            vals = [torch.zeros(len(self.tag_ids), 3).to(self.device)]
                            mask = [torch.zeros(len(self.tag_ids), 3).to(self.device)]
                            nobs = [torch.zeros(len(self.tag_ids)).to(self.device)]
                            labels = [
                                torch.zeros(len(self.label_names)).to(self.device)
                            ]

                            first_tp = time
                            time = round((time - first_tp) / 10 ** 5)
                            prev_time = time
                        else:
                            # for speed -- we actually don't need to quantize it in Latent ODE
                            time = round(
                                (time - first_tp) / 10 ** 5
                            )  # quatizing by 100 ms. 10,000 is one millisecond, 10,000,000 is one second

                        if time != prev_time:
                            tt.append(time)
                            vals.append(
                                torch.zeros(len(self.tag_ids), 3).to(self.device)
                            )
                            mask.append(
                                torch.zeros(len(self.tag_ids), 3).to(self.device)
                            )
                            nobs.append(torch.zeros(len(self.tag_ids)).to(self.device))
                            labels.append(
                                torch.zeros(len(self.label_names)).to(self.device)
                            )
                            prev_time = time

                        if tag_id in self.tag_ids:
                            n_observations = nobs[-1][self.tag_dict[tag_id]]
                            if (self.reduce == "average") and (n_observations > 0):
                                prev_val = vals[-1][self.tag_dict[tag_id]]
                                new_val = (prev_val * n_observations + value_vec) / (
                                    n_observations + 1
                                )
                                vals[-1][self.tag_dict[tag_id]] = new_val
                            else:
                                vals[-1][self.tag_dict[tag_id]] = value_vec

                            mask[-1][self.tag_dict[tag_id]] = 1
                            nobs[-1][self.tag_dict[tag_id]] += 1

                            if label in self.label_names:
                                if torch.sum(labels[-1][self.label_dict[label]]) == 0:
                                    labels[-1][self.label_dict[label]] = 1
                        else:
                            assert (
                                tag_id == "RecordID"
                            ), "Read unexpected tag id {}".format(tag_id)
                    save_record(records, record_id, tt, vals, mask, labels)

            torch.save(records, os.path.join(self.processed_folder, "data.pt"))

        print("Done!")

    def _check_exists(self):
        for url in self.urls:
            filename = url.rpartition("/")[2]
            if not os.path.exists(os.path.join(self.processed_folder, "data.pt")):
                return False
        return True

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def data_file(self):
        return "data.pt"

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.root)
        fmt_str += "    Max length: {}\n".format(self.max_seq_length)
        fmt_str += "    Reduce: {}\n".format(self.reduce)
        return fmt_str


def get_person_id(record_id):
    # The first letter is the person id
    person_id = record_id[0]
    person_id = ord(person_id) - ord("A")
    return person_id


def variable_time_collate_fn_activity(
    batch, args, device=torch.device("cpu"), data_type="train"
):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
        - record_id is a patient id
        - tt is a 1-dimensional tensor containing T time values of observations.
        - vals is a (T, D) tensor containing observed values for D variables.
        - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
        - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
        combined_tt: The union of all time observations.
        combined_vals: (M, T, D) tensor containing the observed values.
        combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0][2].shape[1]
    N = batch[0][-1].shape[1]  # number of labels

    combined_tt, inverse_indices = torch.unique(
        torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True
    )
    combined_tt = combined_tt.to(device)

    offset = 0
    combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_labels = torch.zeros([len(batch), len(combined_tt), N]).to(device)

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        tt = tt.to(device)
        vals = vals.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        indices = inverse_indices[offset : offset + len(tt)]
        offset += len(tt)

        combined_vals[b, indices] = vals
        combined_mask[b, indices] = mask
        combined_labels[b, indices] = labels

    combined_tt = combined_tt.float()

    if torch.max(combined_tt) != 0.0:
        combined_tt = combined_tt / torch.max(combined_tt)

    breakpoint()

    data_dict = {
        "data": combined_vals,
        "time_steps": combined_tt,
        "mask": combined_mask,
        "labels": combined_labels,
    }

    data_dict = utils.split_and_subsample_batch(data_dict, args, data_type=data_type)
    return data_dict


def get_person_dataset(args):
    n_samples = min(10000, args.n)
    device = torch.device("cpu")
    dataset_obj = PersonActivity(
        "data/PersonActivity", download=True, n_samples=n_samples, device=device
    )
    print(dataset_obj)
    # Use custom collate_fn to combine samples with arbitrary time observations.
    # Returns the dataset along with mask and time steps

    # Shuffle and split
    train_data, test_data = model_selection.train_test_split(
        dataset_obj, train_size=0.8, random_state=42, shuffle=True
    )

    train_data = [
        train_data[i] for i in np.random.choice(len(train_data), len(train_data))
    ]
    test_data = [test_data[i] for i in np.random.choice(len(test_data), len(test_data))]

    record_id, tt, vals, mask, labels = train_data[0]
    input_dim = vals.size(-1)

    batch_size = min(min(len(dataset_obj), args.batch_size), args.n)
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        # collate_fn=lambda batch: variable_time_collate_fn_activity(
        #     batch, args, device, data_type="train"
        # ),
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=n_samples,
        num_workers=4,
        shuffle=False,
        # collate_fn=lambda batch: variable_time_collate_fn_activity(
        #     batch, args, device, data_type="test"
        # ),
    )

    data_objects = {
        "dataset_obj": dataset_obj,
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "input_dim": input_dim,
        "n_train_batches": len(train_dataloader),
        "n_test_batches": len(test_dataloader),
        "classif_per_tp": True,  # optional
        "n_labels": labels.size(-1),
    }

    return data_objects


if __name__ == "__main__":
    torch.manual_seed(1991)

    class FakeArg:
        batch_size = 32
        classif = True
        extrap = False
        sample_tp = None
        cut_tp = None
        n = 10000

    ds = get_person_dataset(FakeArg())
    for batch in ds["train_dataloader"]:
        breakpoint()
    # dataset = PersonActivity("data/PersonActivity", download=True)
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=30,
    #     shuffle=True,
    #     collate_fn=variable_time_collate_fn_activity,
    # )
    # dataloader.__iter__().next()