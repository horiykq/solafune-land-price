from tensorflow.python.client import device_lib


def verify_gpu():
    device_lib.list_local_devices()


if __name__ == "__main__":
    verify_gpu()
