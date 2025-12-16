from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar, overload

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletionContentPartTextParam
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.shared.chat_model import ChatModel
from openai.types.shared_params.reasoning_effort import ReasoningEffort
from openai.types.shared_params.response_format_json_schema import JSONSchema, ResponseFormatJSONSchema
from pydantic import BaseModel

from agent.llm.base import BaseChatModel
from agent.llm.exceptions import ModelProviderError, ModelRateLimitError
from agent.llm.messages import BaseMessage
from agent.llm.openai.serializer import OpenAIMessageSerializer
from agent.llm.schema import SchemaOptimizer
from agent.llm.views import ChatInvokeCompletion, ChatInvokeUsage

T = TypeVar('T', bound=BaseModel)


@dataclass
class ChatOpenAI(BaseChatModel):
	"""
	A wrapper around AsyncOpenAI that implements the BaseLLM protocol.

	This class accepts all AsyncOpenAI parameters while adding model
	and temperature parameters for the LLM interface (if temperature it not `None`).
	"""

	# Model configuration
	model: ChatModel | str

	# Model params
	temperature: float | None = 0.2
	frequency_penalty: float | None = 0.3  # this avoids infinite generation of \t for models like 4.1-mini
	reasoning_effort: ReasoningEffort = 'low'
	seed: int | None = None
	service_tier: Literal['auto', 'default', 'flex', 'priority', 'scale'] | None = None
	top_p: float | None = None
	add_schema_to_system_prompt: bool = False  # Add JSON schema to system prompt instead of using response_format
	dont_force_structured_output: bool = False  # If True, the model will not be forced to output a structured output
	remove_min_items_from_schema: bool = (
		False  # If True, remove minItems from JSON schema (for compatibility with some providers)
	)
	remove_defaults_from_schema: bool = (
		False  # If True, remove default values from JSON schema (for compatibility with some providers)
	)

	# Client initialization parameters
	api_key: str | None = None
	organization: str | None = None
	project: str | None = None
	base_url: str | httpx.URL | None = None
	websocket_base_url: str | httpx.URL | None = None
	timeout: float | httpx.Timeout | None = None
	max_retries: int = 5  # Increase default retries for automation reliability
	default_headers: Mapping[str, str] | None = None
	default_query: Mapping[str, object] | None = None
	http_client: httpx.AsyncClient | None = None
	_strict_response_validation: bool = False
	max_completion_tokens: int | None = 4096
	reasoning_models: list[ChatModel | str] | None = field(
		default_factory=lambda: [
			'o4-mini',
			'o3',
			'o3-mini',
			'o1',
			'o1-pro',
			'o3-pro',
			'gpt-5',
			'gpt-5-mini',
			'gpt-5-nano',
		]
	)

	# Static
	@property
	def provider(self) -> str:
		return 'openai'

	def _get_client_params(self) -> dict[str, Any]:
		"""Prepare client parameters dictionary."""
		# Define base client params
		base_params = {
			'api_key': self.api_key,
			'organization': self.organization,
			'project': self.project,
			'base_url': self.base_url,
			'websocket_base_url': self.websocket_base_url,
			'timeout': self.timeout,
			'max_retries': self.max_retries,
			'default_headers': self.default_headers,
			'default_query': self.default_query,
			'_strict_response_validation': self._strict_response_validation,
		}

		# Create client_params dict with non-None values
		client_params = {k: v for k, v in base_params.items() if v is not None}

		# Add http_client if provided
		if self.http_client is not None:
			client_params['http_client'] = self.http_client

		return client_params

	def _normalize_messages_for_claude(self, messages: list[Any]) -> list[Any]:
		"""Нормализует сообщения для Claude через HydraAI.
		
		HydraAI требует, чтобы content был строкой, а не объектом с type='text'.
		Конвертирует списки с одним текстовым элементом в простые строки.
		Также обрабатывает случаи, когда content уже является строкой (оставляет как есть).
		"""
		normalized = []
		for msg in messages:
			# Создаем копию сообщения
			if isinstance(msg, dict):
				normalized_msg = msg.copy()
			else:
				normalized_msg = dict(msg)
			
			# Нормализуем content
			if 'content' in normalized_msg:
				content = normalized_msg['content']
				
				# Если content уже строка - оставляем как есть
				if isinstance(content, str):
					pass  # Уже правильный формат
				# Если content - список с одним текстовым элементом
				elif isinstance(content, list):
					if len(content) == 1 and isinstance(content[0], dict):
						content_obj = content[0]
						if content_obj.get('type') == 'text' and 'text' in content_obj:
							normalized_msg['content'] = content_obj['text']
					# Если список с несколькими элементами - объединяем текст
					elif len(content) > 1:
						text_parts = []
						for part in content:
							if isinstance(part, dict) and part.get('type') == 'text' and 'text' in part:
								text_parts.append(part['text'])
							elif isinstance(part, str):
								text_parts.append(part)
						if text_parts:
							normalized_msg['content'] = '\n'.join(text_parts)
				# Если content - объект с type='text'
				elif isinstance(content, dict) and content.get('type') == 'text' and 'text' in content:
					normalized_msg['content'] = content['text']
			
			normalized.append(normalized_msg)
		
		return normalized

	def get_client(self) -> AsyncOpenAI:
		"""
		Returns an AsyncOpenAI client.

		Returns:
			AsyncOpenAI: An instance of the AsyncOpenAI client.
		"""
		client_params = self._get_client_params()
		return AsyncOpenAI(**client_params)

	@property
	def name(self) -> str:
		return str(self.model)

	def _get_usage(self, response: ChatCompletion) -> ChatInvokeUsage | None:
		if response.usage is not None:
			completion_tokens = response.usage.completion_tokens
			completion_token_details = response.usage.completion_tokens_details
			if completion_token_details is not None:
				reasoning_tokens = completion_token_details.reasoning_tokens
				if reasoning_tokens is not None:
					completion_tokens += reasoning_tokens

			usage = ChatInvokeUsage(
				prompt_tokens=response.usage.prompt_tokens,
				prompt_cached_tokens=response.usage.prompt_tokens_details.cached_tokens
				if response.usage.prompt_tokens_details is not None
				else None,
				prompt_cache_creation_tokens=None,
				prompt_image_tokens=None,
				# Completion
				completion_tokens=completion_tokens,
				total_tokens=response.usage.total_tokens,
			)
		else:
			usage = None

		return usage

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		"""
		Invoke the model with the given messages.

		Args:
			messages: List of chat messages
			output_format: Optional Pydantic model class for structured output

		Returns:
			Either a string response or an instance of output_format
		"""

		openai_messages = OpenAIMessageSerializer.serialize_messages(messages)
		
		# Нормализация сообщений для Claude через HydraAI
		# HydraAI требует, чтобы content был строкой, а не объектом с type='text'
		if self.base_url and 'hydraai.ru' in str(self.base_url).lower():
			is_claude = any(claude_name in str(self.model).lower() for claude_name in ['claude', 'sonnet', 'haiku', 'opus'])
			if is_claude:
				openai_messages = self._normalize_messages_for_claude(openai_messages)

		try:
			model_params: dict[str, Any] = {}

			if self.temperature is not None:
				model_params['temperature'] = self.temperature

			if self.frequency_penalty is not None:
				model_params['frequency_penalty'] = self.frequency_penalty

			if self.max_completion_tokens is not None:
				model_params['max_completion_tokens'] = self.max_completion_tokens

			if self.top_p is not None:
				model_params['top_p'] = self.top_p

			if self.seed is not None:
				model_params['seed'] = self.seed

			if self.service_tier is not None:
				model_params['service_tier'] = self.service_tier

			if self.reasoning_models and any(str(m).lower() in str(self.model).lower() for m in self.reasoning_models):
				model_params['reasoning_effort'] = self.reasoning_effort
				model_params.pop('temperature', None)
				model_params.pop('frequency_penalty', None)

			if output_format is None:
				# Return string response
				response = await self.get_client().chat.completions.create(
					model=self.model,
					messages=openai_messages,
					**model_params,
				)

				usage = self._get_usage(response)
				return ChatInvokeCompletion(
					completion=response.choices[0].message.content or '',
					usage=usage,
					stop_reason=response.choices[0].finish_reason if response.choices else None,
				)

			else:
				response_format: JSONSchema = {
					'name': 'agent_output',
					'strict': True,
					'schema': SchemaOptimizer.create_optimized_json_schema(
						output_format,
						remove_min_items=self.remove_min_items_from_schema,
						remove_defaults=self.remove_defaults_from_schema,
					),
				}

				# Add JSON schema to system prompt if requested
				if self.add_schema_to_system_prompt and openai_messages and openai_messages[0]['role'] == 'system':
					schema_text = f'\n<json_schema>\n{response_format}\n</json_schema>'
					if isinstance(openai_messages[0]['content'], str):
						openai_messages[0]['content'] += schema_text
					elif isinstance(openai_messages[0]['content'], Iterable):
						openai_messages[0]['content'] = list(openai_messages[0]['content']) + [
							ChatCompletionContentPartTextParam(text=schema_text, type='text')
						]

				if self.dont_force_structured_output:
					response = await self.get_client().chat.completions.create(
						model=self.model,
						messages=openai_messages,
						**model_params,
					)
				else:
					# Return structured response
					response = await self.get_client().chat.completions.create(
						model=self.model,
						messages=openai_messages,
						response_format=ResponseFormatJSONSchema(json_schema=response_format, type='json_schema'),
						**model_params,
					)

				if response.choices[0].message.content is None:
					raise ModelProviderError(
						message='Failed to parse structured output from model response',
						status_code=500,
						model=self.name,
					)

				usage = self._get_usage(response)

				content = response.choices[0].message.content
				
				import logging
				logger = logging.getLogger(__name__)
				
				# Сначала пытаемся парсить напрямую (быстрее для чистого JSON)
				parsed = None
				try:
					parsed = output_format.model_validate_json(content)
				except Exception:
					# Если прямой парсинг не удался, пытаемся извлечь JSON из markdown обертки
					# Это нормальная ситуация, когда модель возвращает JSON в ```json ... ```
					from agent.utils import extract_json_from_text
					try:
						json_str = extract_json_from_text(content)
						parsed = output_format.model_validate_json(json_str)
						# Если извлечение успешно, логируем как предупреждение (не ошибка)
						logger.warning(f'[DEBUG] LLM returned JSON wrapped in markdown, extracted successfully')
					except Exception as extract_error:
						# Если и извлечение не помогло, это реальная ошибка
						logger.error(f'[DEBUG] LLM returned invalid JSON. Full response: {content[:500]}...')
						logger.error(f'[DEBUG] Parse error: {str(extract_error)[:500]}')
						error_msg = str(extract_error)
						# Обрезаем огромные пасты валидации до разумного размера
						if len(error_msg) > 200:
							error_msg = error_msg[:200] + '...'
						raise ModelProviderError(
							message=f'Failed to parse JSON: {error_msg}',
							status_code=500,
							model=self.name,
						) from extract_error

				return ChatInvokeCompletion(
					completion=parsed,
					usage=usage,
					stop_reason=response.choices[0].finish_reason if response.choices else None,
				)

		except RateLimitError as e:
			raise ModelRateLimitError(message=e.message, model=self.name) from e

		except APIConnectionError as e:
			raise ModelProviderError(message=str(e), model=self.name) from e

		except APIStatusError as e:
			raise ModelProviderError(message=e.message, status_code=e.status_code, model=self.name) from e

		except Exception as e:
			raise ModelProviderError(message=str(e), model=self.name) from e
