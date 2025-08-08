from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="INFO", enqueue=True, backtrace=False, diagnose=False,
           format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | "
                  "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                  "<level>{message}</level>")

def get_logger(name: str):
    return logger.bind(mod=name)
