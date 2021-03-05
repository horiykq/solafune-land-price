import datetime
import glob
import os
import shutil

from constants import BASE_DIR


def save_history() -> bool:
    try:
        now = datetime.datetime.now()
        sources = glob.glob(f"{BASE_DIR}/*.py")
        modules = glob.glob(f"{BASE_DIR}/**/*.py")

        history_dir = f"{BASE_DIR}/history"
        backup_dir = f"{history_dir}/{now}"
        if not os.path.exists(backup_dir):
            os.mkdir(backup_dir)
        for target in sources:
            shutil.copy(target, backup_dir)
        if not os.path.exists(f"{backup_dir}/modules"):
            os.mkdir(f"{backup_dir}/modules")
        for target in modules:
            shutil.copy(target, f"{backup_dir}/modules")
        return True

    except Exception as error:
        print(error)
        return False


if __name__ == "__main__":
    save_history()
