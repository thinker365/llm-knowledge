# -*- coding: utf-8 -*-
# @Project: LLM-demo
# @FileName: logger.py
# @Author: LiuLinYuan
# @Time: 2024/5/22 10:17


import datetime
import loguru
from common.file_path import LOG_DIR

run_date = datetime.datetime.now().strftime("%Y_%m_%d")


class Logger:
    """封装日志库"""

    def __init__(self):
        loguru.logger.remove()  # 移除默认的输出处理器，避免重复输出
        loguru.logger.add(
            f"{LOG_DIR}/runtime_{run_date}.log",
            format="{time:YYYY-MM-DD HH:mm:ss}-{level}-{message}",
        )
        loguru.logger.add(
            sink=lambda msg: print(msg, end=""),  # 控制台输出
            format="{time:YYYY-MM-DD HH:mm:ss}-{level}-{message}",
        )

    @staticmethod
    def info(message):
        loguru.logger.info(message)

    @staticmethod
    def error(message):
        loguru.logger.error(message)

    @staticmethod
    def debug(message):
        loguru.logger.debug(message)

    @staticmethod
    def success(message):
        loguru.logger.success(message)

    @staticmethod
    def warning(message):
        loguru.logger.warning(message)

    @staticmethod
    def critical(message):
        loguru.logger.critical(message)


if __name__ == "__main__":
    Logger().info("info")
    Logger().debug("debug")
    Logger().error("error")
    Logger().success("success")
    Logger().warning("warning")
    Logger().critical("critical")
