"""
This file provides the necessary tools to log generic information, as well as
specific tensorboard information
"""
import logging


def create_std_logger(logger_name, filepath, level):
    """
    Creates a logger with a specified name and log file.

    Parameters:
        logger_name - the name of the logger
        filepath - where the file handler should point to
        level - the logging level

    Return:
        logger - a logger with the input specifications
    """
    # create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # create file handler 
    fh = logging.FileHandler(filepath)
    fh.setLevel(level)
    # print to terminal if we get ERROR or CRITICAL
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the self.logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
