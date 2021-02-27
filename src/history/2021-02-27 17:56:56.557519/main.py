import sys

import numpy as np
import pandas as pd
import seaborn as sns

from constants import DATA_DIR
from modules.save_history import save_history


def main():
    success_save_history = save_history()
    if not success_save_history:
        sys.exit()


if __name__ == "__main__":
    main()
