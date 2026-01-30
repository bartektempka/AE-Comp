import re
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils import create_sequences


def read_dataset(
    dataset_name: str,
    return_windowed: bool = False,
    size_limit: int | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:

    dataset_filesnames = []

    try:
        dataset_dir = os.listdir("../TSB-AD-M")
        dataset_path = "../TSB-AD-M"
    except FileNotFoundError:
        dataset_dir = os.listdir("TSB-AD-M")
        dataset_path = "TSB-AD-M"
    for filename in dataset_dir:
        match = re.match(r"(.*)_" + re.escape(dataset_name) + r"_id_.+_.*", filename)
        if match:
            dataset_filesnames.append(filename)
    dataset_dict = {}

    for filename in (
        dataset_filesnames[:size_limit]
        if size_limit is not None
        else dataset_filesnames
    ):
        print(filename)
        traning_index = filename.split(".")[0].split("_")[-3]
        df = pd.read_csv(os.path.join(dataset_path, filename)).dropna()
        train_data = df.iloc[: int(traning_index), :-1].values.astype(float)
        test_data = df.iloc[int(traning_index) :, :-1].values.astype(float)
        test_labels = df.iloc[int(traning_index) :, -1].values.astype(int)

        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)

        dataset_name = filename[:-4]
        data_set_name_split = dataset_name.split("_")
        dataset_name_formatted = f"{data_set_name_split[1]}_id_{data_set_name_split[3]}"

        if return_windowed:
            dataset_dict[dataset_name_formatted] = (
                create_sequences(train_data, 50),
                create_sequences(test_data, 50),
                create_sequences(test_labels, 50),
            )
        else:
            dataset_dict[dataset_name_formatted] = (
                train_data,
                test_data,
                test_labels,
            )

    return dataset_dict


def read_dataset_semisupervised(
    dataset_name: str,
    size_limit: int | None = None,
    anomaly_fraction: float = 0.01,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:

    dataset_dict = read_dataset(
        dataset_name, return_windowed=False, size_limit=size_limit
    )
    # move some anomalies to training set
    for key in dataset_dict.keys():
        train_data, test_data, test_labels = dataset_dict[key]
        n_anomalies = np.sum(test_labels)
        if n_anomalies == 0:
            continue
        anomaly_indices = np.where(test_labels == 1)[0]
        n_to_move = max(1, int(anomaly_fraction * n_anomalies))
        last_anomaly_index = anomaly_indices[
            min(n_to_move - 1, len(anomaly_indices) - 1)
        ]

        normal_train_seq = create_sequences(train_data, 50)
        normal_train_labels = np.zeros(normal_train_seq.shape[:2])
        train_data_with_anomalies = test_data[: last_anomaly_index + 1]
        train_labels_with_anomalies = test_labels[: last_anomaly_index + 1]
        train_data_with_anomalies_seq = create_sequences(train_data_with_anomalies, 50)
        train_labels_with_anomalies_seq = create_sequences(
            train_labels_with_anomalies, 50
        )

        full_train_data = np.vstack((normal_train_seq, train_data_with_anomalies_seq))
        full_train_labels = np.vstack(
            (normal_train_labels, train_labels_with_anomalies_seq)
        )

        test_data = test_data[last_anomaly_index + 1 :]
        test_labels = test_labels[last_anomaly_index + 1 :]

        test_data_seq = create_sequences(test_data, 50)
        test_labels_seq = create_sequences(test_labels, 50)

        dataset_dict[key] = (
            full_train_data,
            full_train_labels,
            test_data_seq,
            test_labels_seq,
        )

    return dataset_dict
