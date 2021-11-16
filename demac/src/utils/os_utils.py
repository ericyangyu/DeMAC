"""
Helper function library that deal with the OS.
"""
import shutil
import os


def init_dir(path):
    """
    Hard initializes a directory given a path.

    Parameters:
        path - a directory path to initialize

    Return:
    """
    shutil.rmtree(path, ignore_errors=True)
    os.mkdir(path)
