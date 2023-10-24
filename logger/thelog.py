import logging
import sys
import os

_ALL_LOGGERS = {}


def get_logger(name=None, dir=None, logFileName=None):
    logger = logging.getLogger(name)
    if name in _ALL_LOGGERS:
        return logger
    # fmt = "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] %(thread)d: %(message)s"
    fmt = "%(asctime)s %(levelname)s [%(name)s] : %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(formatter)
    std_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(
        os.path.join(dir, logFileName), encoding='utf8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.handlers = []
    # logger.addHandler(std_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.DEBUG)
    _ALL_LOGGERS[name] = logger
    return logger

