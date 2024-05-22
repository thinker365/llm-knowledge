# -*- coding: utf-8 -*-
# @Project: LLM-demo
# @FileName: file_path.py
# @Author: LiuLinYuan
# @Time: 2024/5/22 10:20


from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

CONF_DIR = BASE_DIR.joinpath("config")
LOG_DIR = BASE_DIR.joinpath("logs")
DOC_DIR = BASE_DIR.joinpath("doc")
DB_DIR = BASE_DIR.joinpath("db")

if __name__ == "__main__":
    print(BASE_DIR)
    print(Path(__file__))
    print(DB_DIR.joinpath('chroma'))
