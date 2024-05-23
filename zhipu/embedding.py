# -*- coding: utf-8 -*-
# @Project: LLM-demo
# @FileName: embedding.py
# @Author: LiuLinYuan
# @Time: 2024/5/21 16:49

import os
from zhipuai import ZhipuAI
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import Chroma
from common.file_path import DB_DIR
from zhipu.data_handle import DocHandle
from utils.logger import Logger

from typing import Dict, List, Any
from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator

load_dotenv(find_dotenv('../.env'))
log = Logger()

def zhipu_embedding(text: str):
	# api_key = os.environ['ZHIPUAI_API_KEY']
	api_key = '2dbd072adc31da81e6e05a65ab6ced94.lL7e507FPgrTKb9p'
	client = ZhipuAI(api_key=api_key)
	return client.embeddings.create(
		model="embedding-2",
		input=text,
	)

class ZhipuAIEmbeddings(BaseModel, Embeddings):
	"""`Zhipuai Embeddings` embedding models."""
	
	client: Any
	"""`zhipuai.ZhipuAI"""
	
	# @root_validator()
	def validate_environment(cls, values: Dict) -> Dict:
		from zhipuai import ZhipuAI
		values["client"] = ZhipuAI()
		return values
	
	def embed_query(self, text: str) -> List[float]:
		embeddings = self.client.embeddings.create(
			model="embedding-2",
			input=text
		)
		return embeddings.data[0].embedding
	
	def embed_documents(self, texts: List[str]) -> List[List[float]]:
		return [self.embed_query(text) for text in texts]

embedding = ZhipuAIEmbeddings()
persist_directory = DB_DIR.joinpath('chroma')

def embedding_persist(split_docs):
	# 向量持久化
	vectordb = Chroma.from_documents(
		documents=split_docs[:300],
		embedding=embedding,
		persist_directory=persist_directory
	)
	vectordb.persist()

def embedding_load():
	# 加载向量数据
	vectordb = Chroma(
		persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
		embedding_function=embedding
	)
	log.info(f'向量库中存储的数量：{vectordb.collection.count()}')
	
	sim_docs = vectordb.similarity_search('南瓜书', k=3)
	log.info(f"检索到的内容数：{len(sim_docs)}")
	for index, sim_doc in enumerate(sim_docs, start=1):
		log.info(f"检索到的第{index}个，内容: \n {sim_doc.page_content}")

def db():
	return Chroma(
		persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
		embedding_function=embedding
	)

if __name__ == '__main__':
	# response = zhipu_embedding(text='text')
	#
	# print(f'response类型为：{type(response)}')
	# print(f'embedding类型为：{response.object}')
	# print(f'生成embedding的model为：{response.model}')
	# print(f'生成的embedding长度为：{len(response.data[0].embedding)}')
	# print(f'embedding（前10）为: {response.data[0].embedding[:10]}')
	
	# doc = DocHandle('pumpkin_book.pdf')
	# data = doc.data_load()
	# split_docs = doc.data_split(data)
	# print(embedding_persist(split_docs))
	
	embedding_load()
