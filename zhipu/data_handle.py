# -*- coding: utf-8 -*-
# @Project: LLM-demo
# @FileName: data_handle.py
# @Author: LiuLinYuan
# @Time: 2024/5/21 17:22

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from common.file_path import DOC_DIR
from utils.logger import Logger

log = Logger()


class DocHandle:
    def __init__(self, file_path):
        self.file_path = file_path
        self.CHUNK_SIZE = 500
        self.OVERLAP_SIZE = 50

    def data_load(self):
        loader = PyMuPDFLoader(DOC_DIR.joinpath(self.file_path))
        pdf_pages = loader.load()
        return pdf_pages

    def data_split(self, doc_data):
        # 使用递归字符文本分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.OVERLAP_SIZE
        )
        return text_splitter.split_documents(doc_data)


def data_clean(pdf_page):
    import re
    pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
    pdf_page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''),
                                   pdf_page.page_content).replace('.', '').replace(' ', '')
    return pdf_page.page_content


if __name__ == '__main__':
    doc = DocHandle('pumpkin_book.pdf')
    data = doc.data_load()
    print(doc.data_split(data))
    # for page in data:
    #     data = data_clean(page)
    #     print(doc.data_split(data))
