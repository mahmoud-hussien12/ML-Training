import logging
from asgi_correlation_id import correlation_id
logger = None
def getLogger():
    global logger
    handler = logging.FileHandler("api/logs/info.log", mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            '%(asctime)s [%(levelname)s] %(correlation_id)s %(message)s %(data)s',
            defaults={"correlation_id": correlation_id.get() or "" , "data": ""}
        )
    )
    if logger is None:
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                handler
            ]
        )
        logger = logging.getLogger(__name__)
    return logger

def log_info(message: str, data: dict = ""):
    logger = getLogger()
    logger.info(message, extra={"correlation_id": correlation_id.get(), "data": data})

def log_error(message: str, data: dict = ""):
    logger = getLogger()
    logger.error(message, extra={"correlation_id": correlation_id.get(), "data": data})

