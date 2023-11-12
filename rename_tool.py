import os
import time


def path(path_name, t):
    directory = os.path.dirname(os.path.abspath(__file__))
    timestamp = int(time.time())
    file_path = f"./output/{path_name}/{timestamp}.{t}"
    file_name = os.path.join(directory, file_path)
    return file_name
