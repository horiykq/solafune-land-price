import datetime
import glob
import os
import shutil

BASE_DIR = "."


def save_history():
    try:
        now = datetime.datetime.now()
        targets = glob.glob(f"{BASE_DIR}/*.py")

        history_dir = f"{BASE_DIR}/history"
        backup_dir = f"{history_dir}/{now}"
        if not os.path.exists(backup_dir):
            os.mkdir(backup_dir)
        for target in targets:
            shutil.copy(target, backup_dir)
        return True

    except Exception as error:
        print(error)
        return False


if __name__ == "__main__":
    save_history()
