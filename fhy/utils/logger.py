import logging


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Constructs a Logger given a name and level."""
    log: logging.Logger = logging.getLogger(name)
    log.setLevel(level)
    stream = logging.StreamHandler()
    form = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s | %(funcName)s():%(lineno)d"
    )
    stream.setLevel(level)
    stream.setFormatter(form)
    log.addHandler(stream)

    return log
