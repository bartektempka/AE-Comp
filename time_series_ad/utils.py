import numpy as np

def create_sequences(data, seq_len: int, step: int = 1):
    data = np.asarray(data, dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(data, seq_len, axis=0)
    return windows[::step].transpose(0, 2, 1) if len(data.shape) == 2 else windows[::step]


def go_back_to_original_shape(
    predictions: np.ndarray, original_data_len: int, seq_step: int = 1
) -> np.ndarray:
    seq_len = predictions.shape[1]
    aggregated_preds = np.zeros((original_data_len, predictions.shape[2]))
    counts = np.zeros((original_data_len, 1))
    print(f"aggregated_preds.shape {aggregated_preds.shape}")
    for start_idx in range(0, original_data_len - seq_len + 1, seq_step):
        end_idx = start_idx + seq_len
        aggregated_preds[start_idx:end_idx] += predictions[start_idx // seq_step]
        counts[start_idx:end_idx] += 1

    print(f"aggregated_preds.shape {aggregated_preds.shape}")

    # Avoid division by zero
    counts[counts == 0] = 1
    averaged_preds = aggregated_preds / counts
    return averaged_preds