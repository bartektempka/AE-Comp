import re
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def read_dataset(
    dataset_name: str,
    size_limit: int | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:

    dataset_filesnames = []

    for filename in os.listdir("../TSB-AD-M"):
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
        df = pd.read_csv(os.path.join("../TSB-AD-M", filename)).dropna()
        train_data = df.iloc[: int(traning_index), :-1].values.astype(float)
        test_data = df.iloc[int(traning_index) :, :-1].values.astype(float)
        test_labels = df.iloc[int(traning_index) :, -1].values.astype(int)

        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)

        dataset_name = filename[:-4]
        data_set_name_split = dataset_name.split("_")
        dataset_name_formatted = f"{data_set_name_split[1]}_id_{data_set_name_split[3]}"

        dataset_dict[dataset_name_formatted] = (train_data, test_data, test_labels)

    return dataset_dict
