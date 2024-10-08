"""
    File contains infinite sample dataloader for 2D point datasets (e.g., radial checkerboard density)
    for use in training mlp diffusion models.
"""

import blobfile as bf
import numpy as np
import torch as th
from torch.utils.data import DataLoader, TensorDataset


def load_data(
    *,
    data_dir,
    train_batch_size,
    test_batch_size,
    deterministic=False,
):
    """
    For a dataset, create a generator over (points, kwargs) pairs.

    Each datapoint is a (x,y) float tensor, and hte kwargs dict contains zero
    or more keys about the classes information.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param deterministic: if True, yield datapoints in a deterministic order.
    """

    if not data_dir:
        raise ValueError("unspecified data directory")
    
    dataset = TensorDataset(th.from_numpy(np.load(data_dir)["data"]).type(th.float32))
    
    if deterministic:
        train_loader = DataLoader(
            dataset, batch_size=train_batch_size, shuffle=False, num_workers=1, drop_last=True
        )
        test_loader = DataLoader(
            dataset, batch_size=test_batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        train_loader = DataLoader(
            dataset, batch_size=train_batch_size, shuffle=True, num_workers=1, drop_last=True
        )
        test_loader = DataLoader(
            dataset, batch_size=test_batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    
    return train_loader, test_loader