# -*- coding: utf-8 -*-
# @Project: LLM-demo
# @FileName: llm.py
# @Author: LiuLinYuan
# @Time: 2024/5/21 16:48

import os
from abc import ABC
from typing import Any, List, Mapping, Optional, Dict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from zhipuai import ZhipuAI

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv('.env'))

client = ZhipuAI(
	api_key=os.environ["ZHIPUAI_API_KEY"]
)

# def gen_glm_params(prompt):
# 	"""
# 	构造 GLM 模型请求参数 messages
#
# 	请求参数：
# 		prompt: 对应的用户提示词
# 	"""
# 	messages = [{"role": "user", "content": prompt}]
# 	return messages
#
# def get_completion(prompt, model="glm-4", temperature=0.95):
# 	"""
# 	获取 GLM 模型调用结果
#
# 	请求参数：
# 		prompt: 对应的提示词
# 		model: 调用的模型，默认为 glm-4，也可以按需选择 glm-3-turbo 等其他模型
# 		temperature: 模型输出的温度系数，控制输出的随机程度，取值范围是 0~1.0，且不能设置为 0。温度系数越低，输出内容越一致。
# 	"""
#
# 	messages = gen_glm_params(prompt)
# 	response = client.chat.completions.create(
# 		model=model,
# 		messages=messages,
# 		temperature=temperature
# 	)
# 	if len(response.choices) > 0:
# 		return response.choices[0].message.content
# 	return "generate answer error"

class ZhipuAILLM(LLM, ABC):
	# 默认选用 glm-4 模型
	model: str = "glm-4"
	# 温度系数
	temperature: float = 0.1
	# API_Key
	api_key: str = os.environ["ZHIPUAI_API_KEY"]
	
	def _call(self, prompt: str, stop: Optional[List[str]] = None,
	          run_manager: Optional[CallbackManagerForLLMRun] = None,
	          **kwargs: Any):
		def gen_glm_params(prompt):
			'''
			构造 GLM 模型请求参数 messages

			请求参数：
				prompt: 对应的用户提示词
			'''
			messages = [{"role": "user", "content": prompt}]
			return messages
		
		client = ZhipuAI(
			api_key=self.api_key
		)
		
		messages = gen_glm_params(prompt)
		response = client.chat.completions.create(
			model=self.model,
			messages=messages,
			temperature=self.temperature
		)
		
		if len(response.choices) > 0:
			return response.choices[0].message.content
		return "generate answer error"
	
	# 首先定义一个返回默认参数的方法
	@property
	def _default_params(self) -> Dict[str, Any]:
		"""获取调用Ennie API的默认参数。"""
		normal_params = {
			"temperature": self.temperature,
		}
		# print(type(self.model_kwargs))
		return {**normal_params}
	
	@property
	def _llm_type(self) -> str:
		return "Zhipu"
	
	@property
	def _identifying_params(self) -> Mapping[str, Any]:
		"""Get the identifying parameters."""
		return {**{"model": self.model}, **self._default_params}