import numpy as np
import datetime

from constants import SUBMITS_DIR, SUBMIT_TEMPLATE


def create_submit_file(preds: np.array) -> bool:
    try:
        template = open(SUBMIT_TEMPLATE, "r")
        submit_data = template.readlines()

        now = datetime.datetime.now()
        output = open(f"{SUBMITS_DIR}/{now}.csv", "a")

        for index in range(len(preds)):
            this_line = submit_data[index]
            if index != 0:
                this_line = this_line.replace("0\n", "")
                this_line = this_line + str(preds[index]) + "\n"
            output.write(this_line)

        output.close()
        template.close()

        return True

    except Exception as error:
        print(error)
        return False


if __name__ == "__main__":
    create_submit_file()
