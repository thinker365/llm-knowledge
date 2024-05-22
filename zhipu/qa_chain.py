# -*- coding: utf-8 -*-
# Author：LiuLinYuan
# Datetime：2024/5/22 15:16
# Project：LLM-demo
# File：qa_chain.py

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from llm import ZhipuAILLM
from embedding import db
from utils.logger import Logger

log = Logger()
llm = ZhipuAILLM()

template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
{context}
问题: {question}
"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       retriever=db().as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
# question_1 = "什么是南瓜书？"
# question_2 = "王阳明是谁？"
# result = qa_chain({"query": question_1})
# log.info(f'大模型+知识库后回答 question_1 的结果：{result["result"]}')
#
# result2 = qa_chain({"query": question_2})
# log.info(f'大模型+知识库后回答 question_2 的结果：{result2["result"]}')

memory = ConversationBufferMemory(
	memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
	return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
)

retriever=db().as_retriever()

qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)
question = "我可以学习到关于提示工程的知识吗？"
result = qa({"question": question})
log.info(result['answer'])
question = "为什么这门课需要教这方面的知识？"
result = qa({"question": question})
log.info(result['answer'])