import logging
from asgi_correlation_id import correlation_id
logger = None
def getLogger():
    global logger
    if logger is None:
        logging.basicConfig(
            filename="api/logs/info.log",
            filemode="a",
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(request_id)s %(message)s %(data)s'
        )
        logger = logging.getLogger(__name__)
    return logger

def log_info(message: str, data: dict = None):
    logger = getLogger()
    logger.info(message, extra={"request_id": correlation_id.get(), "data": data})

def log_error(message: str, data: dict = None):
    logger = getLogger()
    logger.error(message, extra={"request_id": correlation_id.get(), "data": data})

