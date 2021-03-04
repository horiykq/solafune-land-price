import random
import numpy as np
import torch
import tensorflow as tf


def fix_seed(seed: int):
    try:
        # random
        random.seed(seed)

        # numpy
        np.random.seed(seed)

        # pytorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        # tensorflow
        tf.random.set_seed(seed)

        return True

    except Exception as error:
        print(error)
        return False


if __name__ == "__main__":
    fix_seed()
