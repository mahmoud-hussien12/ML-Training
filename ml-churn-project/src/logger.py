import logging

logger = None
def getLogger():
    global logger
    if logger is None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        logger = logging.getLogger(__name__)
    return logger

def log_info(message: str):
    logger = getLogger()
    logger.info(message)

def log_error(message: str):
    logger = getLogger()
    logger.error(message)

