import asyncio
import gc
import inspect
import json
import logging
import re
import tempfile
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast
from urllib.parse import urlparse

if TYPE_CHECKING:
	pass

from dotenv import load_dotenv
from agent.agent.message_manager.utils import save_conversation
from agent.llm.base import BaseChatModel
from agent.llm.exceptions import ModelProviderError, ModelRateLimitError
from agent.llm.messages import BaseMessage, ContentPartImageParam, ContentPartTextParam, UserMessage
from agent.tokens.service import TokenCost

load_dotenv()

from bubus import EventBus
from pydantic import BaseModel, ValidationError
from uuid_extensions import uuid7str

from agent.browser import BrowserSession, BrowserProfile
Browser = BrowserSession  # Псевдоним

# Judge опционален - нужен только если включена оценка judge
try:
	from agent.agent.judge import construct_judge_messages
except ImportError:
	def construct_judge_messages(*args, **kwargs):
		raise NotImplementedError("Judge functionality is not available")

# Ленивый импорт для gif, чтобы избежать тяжелого импорта agent.views при запуске
# from agent.agent.gif import create_history_gif
from agent.agent.message_manager.service import (
	MessageManager,
)
from agent.agent.prompts import SystemPrompt
from agent.agent.views import (
	ActionResult,
	AgentError,
	AgentHistory,
	AgentHistoryList,
	AgentOutput,
	AgentSettings,
	AgentState,
	AgentStepInfo,
	AgentStructuredOutput,
	BrowserStateHistory,
	DetectedVariable,
	JudgementResult,
	StepMetadata,
)
from agent.browser.session import DEFAULT_BROWSER_PROFILE
from agent.browser.views import BrowserStateSummary
from agent.config import CONFIG
from agent.dom.views import DOMInteractedElement
from agent.observability import observe, observe_debug
from agent.tools.registry.views import ActionModel
from agent.tools.service import Tools
from agent.subagents.email_subagent import EmailSubAgent
from agent.utils import (
	URL_PATTERN,
	_log_pretty_path,
	check_latest_agent_version,
	get_agent_version,
	time_execution_async,
	time_execution_sync,
)

logger = logging.getLogger(__name__)


def log_response(response: AgentOutput, registry=None, logger=None) -> None:
	"""Утилита для логирования ответа модели."""

	# Используем модульный логгер, если не предоставлен
	if logger is None:
		logger = logging.getLogger(__name__)

	# Логируем рассуждения только если они присутствуют
	if response.current_state.thinking:
		logger.debug(f'💡 Рассуждения:\n{response.current_state.thinking}')

	# Логируем оценку только если она не пустая
	eval_goal = response.current_state.evaluation_previous_goal
	if eval_goal:
		if 'success' in eval_goal.lower() or 'успех' in eval_goal.lower():
			emoji = '👍'
			# Зеленый цвет для успеха
			logger.info(f'  \033[32m{emoji} Оценка: {eval_goal}\033[0m')
		elif 'failure' in eval_goal.lower() or 'неудача' in eval_goal.lower():
			emoji = '⚠️'
			# Красный цвет для неудачи
			logger.info(f'  \033[31m{emoji} Оценка: {eval_goal}\033[0m')
		else:
			emoji = '❔'
			# Без цвета для неизвестного/нейтрального
			logger.info(f'  {emoji} Оценка: {eval_goal}')

	# Всегда логируем память, если она присутствует
	if response.current_state.memory:
		logger.info(f'  🧠 Память: {response.current_state.memory}')

	# Логируем следующую цель только если она не пустая
	next_goal = response.current_state.next_goal
	if next_goal:
		# Синий цвет для следующей цели
		logger.info(f'  \033[34m🎯 Следующая цель: {next_goal}\033[0m')


Context = TypeVar('Context')


AgentHookFunc = Callable[['Agent'], Awaitable[None]]


class Agent(Generic[Context, AgentStructuredOutput]):
	@time_execution_sync('--init')
	def __init__(
		self,
		task: str,
		llm: BaseChatModel | None = None,
		# Необязательные параметры
		browser_profile: BrowserProfile | None = None,
		browser_session: BrowserSession | None = None,
		browser: Browser | None = None,  # Псевдоним для browser_session
		tools: Tools[Context] | None = None,
		controller: Tools[Context] | None = None,  # Псевдоним для tools
		user_input_callback: Callable[[str], str] | None = None,  # Callback для запроса ввода от пользователя (для капчи)
		# Параметры начального запуска агента
		sensitive_data: dict[str, str | dict[str, str]] | None = None,
		initial_actions: list[dict[str, dict[str, Any]]] | None = None,
		# Облачные колбэки
		register_new_step_callback: (
			Callable[['BrowserStateSummary', 'AgentOutput', int], None]  # Sync callback
			| Callable[['BrowserStateSummary', 'AgentOutput', int], Awaitable[None]]  # Async callback
			| None
		) = None,
		register_done_callback: (
			Callable[['AgentHistoryList'], Awaitable[None]]  # Async Callback
			| Callable[['AgentHistoryList'], None]  # Sync Callback
			| None
		) = None,
		register_external_agent_status_raise_error_callback: Callable[[], Awaitable[bool]] | None = None,
		register_should_stop_callback: Callable[[], Awaitable[bool]] | None = None,
		# Настройки агента
		output_model_schema: type[AgentStructuredOutput] | None = None,
		use_vision: bool | Literal['auto'] = True,
		save_conversation_path: str | Path | None = None,
		save_conversation_path_encoding: str | None = 'utf-8',
		max_failures: int = 3,
		override_system_message: str | None = None,
		extend_system_message: str | None = None,
		generate_gif: bool | str = False,
		available_file_paths: list[str] | None = None,
		include_attributes: list[str] | None = None,
		max_actions_per_step: int = 3,
		use_thinking: bool = True,
		flash_mode: bool = False,
		demo_mode: bool | None = None,
		max_history_items: int | None = None,
		page_extraction_llm: BaseChatModel | None = None,
		fallback_llm: BaseChatModel | None = None,
		ground_truth: str | None = None,
		use_judge: bool = False,
		injected_agent_state: AgentState | None = None,
		source: str | None = None,
		file_system_path: str | None = None,
		task_id: str | None = None,
		calculate_cost: bool = False,
		display_files_in_done_text: bool = True,
		include_tool_call_examples: bool = False,
		vision_detail_level: Literal['auto', 'low', 'high'] = 'auto',
		llm_timeout: int | None = None,
		step_timeout: int = 120,
		directly_open_url: bool = True,
		include_recent_events: bool = False,
		sample_images: list[ContentPartTextParam | ContentPartImageParam] | None = None,
		final_response_after_failure: bool = True,
		llm_screenshot_size: tuple[int, int] | None = None,
		_url_shortening_limit: int = 25,
		**kwargs,
	):
		# Проверка llm_screenshot_size
		if llm_screenshot_size is not None:
			if not isinstance(llm_screenshot_size, tuple) or len(llm_screenshot_size) != 2:
				raise ValueError('llm_screenshot_size must be a tuple of (width, height)')
			width, height = llm_screenshot_size
			if not isinstance(width, int) or not isinstance(height, int):
				raise ValueError('llm_screenshot_size dimensions must be integers')
			if width < 100 or height < 100:
				raise ValueError('llm_screenshot_size dimensions must be at least 100 pixels')
			self.logger.info(f'🖼️  Изменение размера скриншотов для LLM включено: {width}x{height}')
		if llm is None:
			default_llm_name = CONFIG.DEFAULT_LLM
			if default_llm_name:
				from agent.llm.models import get_llm_by_name

				llm = get_llm_by_name(default_llm_name)
			else:
				# LLM по умолчанию не указан, используем оригинальный по умолчанию
				# В упрощённой версии требуется явный llm через настройки среды / конструктор
				raise ValueError('LLM не указан и значение по умолчанию не настроено. Передайте llm явно.')

		# Автоматическая настройка llm_screenshot_size для моделей Claude Sonnet
		if llm_screenshot_size is None:
			model_name = getattr(llm, 'model', '')
			if isinstance(model_name, str) and model_name.startswith('claude-sonnet'):
				llm_screenshot_size = (1400, 850)
				logger.info('🖼️  Автоматически настроен размер скриншотов для LLM (Claude Sonnet): 1400x850')

		if page_extraction_llm is None:
			page_extraction_llm = llm
		if available_file_paths is None:
			available_file_paths = []

		# Установка таймаута на основе имени модели если не указан явно
		if llm_timeout is None:

			def _get_model_timeout(llm_model: BaseChatModel) -> int:
				"""Определение таймаута на основе имени модели"""
				model_name = getattr(llm_model, 'model', '').lower()
				if 'gemini' in model_name:
					if '3-pro' in model_name:
						return 90
					return 45
				elif 'groq' in model_name:
					return 30
				elif 'o3' in model_name or 'claude' in model_name or 'sonnet' in model_name or 'deepseek' in model_name:
					return 90
				else:
					return 60  # Таймаут по умолчанию

			llm_timeout = _get_model_timeout(llm)

		self.id = task_id or uuid7str()
		self.task_id: str = self.id
		self.session_id: str = uuid7str()

		base_profile = browser_profile or DEFAULT_BROWSER_PROFILE
		if base_profile is DEFAULT_BROWSER_PROFILE:
			base_profile = base_profile.model_copy()
		if demo_mode is not None and base_profile.demo_mode != demo_mode:
			base_profile = base_profile.model_copy(update={'demo_mode': demo_mode})
		browser_profile = base_profile

		# Обработка параметров browser vs browser_session (browser имеет приоритет)
		if browser and browser_session:
			raise ValueError('Cannot specify both "browser" and "browser_session" parameters. Use "browser" for the cleaner API.')
		browser_session = browser or browser_session

		if browser_session is not None and demo_mode is not None and browser_session.browser_profile.demo_mode != demo_mode:
			browser_session.browser_profile = browser_session.browser_profile.model_copy(update={'demo_mode': demo_mode})

		self.browser_session = browser_session or BrowserSession(
			browser_profile=browser_profile,
			id=uuid7str()[:-4] + self.id[-4:],  # Повторное использование последних 4 символов для группировки в логах
		)

		self._demo_mode_enabled: bool = bool(self.browser_profile.demo_mode) if self.browser_session else False
		if self._demo_mode_enabled and getattr(self.browser_profile, 'headless', False):
			self.logger.warning(
				'Demo mode is enabled but the browser is headless=True; set headless=False to view the in-browser panel.'
			)

		# Инициализация доступных путей к файлам как прямого атрибута
		self.available_file_paths = available_file_paths

		# Настройка инструментов сначала (нужно для определения output_model_schema)
		if tools is not None:
			self.tools = tools
		elif controller is not None:
			self.tools = controller
		else:
			# Исключить инструмент screenshot когда use_vision не auto
			exclude_actions = ['screenshot'] if use_vision != 'auto' else []
			self.tools = Tools(
				exclude_actions=exclude_actions,
				display_files_in_done_text=display_files_in_done_text,
				user_input_callback=user_input_callback
			)

		# Принудительное исключение screenshot когда use_vision != 'auto', даже если пользователь передал кастомные инструменты
		if use_vision != 'auto':
			self.tools.exclude_action('screenshot')

		# Структурированный вывод - использовать явный параметр или определить из инструментов
		tools_output_model = self.tools.get_output_model()
		if output_model_schema is not None and tools_output_model is not None:
			# Оба предоставлены - предупреждение если они различаются
			if output_model_schema is not tools_output_model:
				logger.warning(
					f'output_model_schema ({output_model_schema.__name__}) отличается от Tools output_model '
					f'({tools_output_model.__name__}). Используется Agent output_model_schema.'
				)
		elif output_model_schema is None and tools_output_model is not None:
			# Только tools имеет его - использовать его (приведение безопасно: оба являются подклассами BaseModel)
			output_model_schema = cast(type[AgentStructuredOutput], tools_output_model)
		self.output_model_schema = output_model_schema
		if self.output_model_schema is not None:
			self.tools.use_structured_output_action(self.output_model_schema)

		# Основные компоненты - улучшение задачи теперь имеет доступ к output_model_schema из инструментов
		self.task = self._enhance_task_with_schema(task, output_model_schema)
		self.llm = llm

		# Конфигурация резервного LLM
		self._fallback_llm: BaseChatModel | None = fallback_llm
		self._using_fallback_llm: bool = False
		self._original_llm: BaseChatModel = llm  # Сохранение оригинала для справки
		self.directly_open_url = directly_open_url
		self.include_recent_events = include_recent_events
		self._url_shortening_limit = _url_shortening_limit

		self.sensitive_data = sensitive_data

		self.sample_images = sample_images

		self.settings = AgentSettings(
			use_vision=use_vision,
			vision_detail_level=vision_detail_level,
			save_conversation_path=save_conversation_path,
			save_conversation_path_encoding=save_conversation_path_encoding,
			max_failures=max_failures,
			override_system_message=override_system_message,
			extend_system_message=extend_system_message,
			generate_gif=generate_gif,
			include_attributes=include_attributes,
			max_actions_per_step=max_actions_per_step,
			use_thinking=use_thinking,
			flash_mode=flash_mode,
			max_history_items=max_history_items,
			page_extraction_llm=page_extraction_llm,
			calculate_cost=calculate_cost,
			include_tool_call_examples=include_tool_call_examples,
			llm_timeout=llm_timeout,
			step_timeout=step_timeout,
			final_response_after_failure=final_response_after_failure,
			use_judge=False,
			ground_truth=None,
		)

		# Token cost service (учёт токенов, без judge_llm)
		self.token_cost_service = TokenCost(include_cost=calculate_cost)
		self.token_cost_service.register_llm(llm)
		self.token_cost_service.register_llm(page_extraction_llm)

		# Инициализация состояния
		self.state = injected_agent_state or AgentState()

		# Инициализация истории
		self.history = AgentHistoryList(history=[], usage=None)

		# Инициализация директории агента
		import time

		timestamp = int(time.time())
		base_tmp = Path(tempfile.gettempdir())
		self.agent_directory = base_tmp / f'agent_agent_{self.id}_{timestamp}'

		# Initialize file system and screenshot service (упрощённо, без сохранения состояния в AgentState)
		self._set_file_system(file_system_path)
		self._set_screenshot_service()

		# Инициализация sub-агентов для специализированных задач
		self.email_subagent = EmailSubAgent()

		# Настройка действий
		self._setup_action_models()
		self._set_agent_version_and_source(source)

		initial_url = None

		# Автоматическое извлечение URL из задачи (если включено)
		if self.directly_open_url and not self.state.follow_up_task and not initial_actions:
			initial_url = self._extract_start_url(self.task)
			if initial_url:
				self.logger.info(f'🔗 Найден URL в задаче: {initial_url}, добавляю как начальное действие...')
				initial_actions = [{'navigate': {'url': initial_url, 'new_tab': False}}]

		self.initial_url = initial_url

		self.initial_actions = self._convert_initial_actions(initial_actions) if initial_actions else None
		# Проверка возможности подключения к модели
		self._verify_and_setup_llm()

		# Обработка попыток использовать use_vision=True с моделями DeepSeek
		if 'deepseek' in self.llm.model.lower():
			self.logger.warning('⚠️ Модели DeepSeek пока не поддерживают use_vision=True. Устанавливаю use_vision=False...')
			self.settings.use_vision = False

		# Обработка попыток использовать use_vision=True с моделями XAI
		if 'grok' in self.llm.model.lower():
			self.logger.warning('⚠️ Модели XAI пока не поддерживают use_vision=True. Устанавливаю use_vision=False...')
			self.settings.use_vision = False

		logger.debug(
			f'{" +vision" if self.settings.use_vision else ""}'
			f' extraction_model={self.settings.page_extraction_llm.model if self.settings.page_extraction_llm else "Unknown"}'
			f'{" +file_system" if self.file_system else ""}'
		)

		# Сохраняем llm_screenshot_size в browser_session, чтобы инструменты могли к нему обращаться
		self.browser_session.llm_screenshot_size = llm_screenshot_size

		# Проверка является ли LLM экземпляром ChatAnthropic
		from agent.llm.anthropic.chat import ChatAnthropic

		is_anthropic = isinstance(self.llm, ChatAnthropic)

		# Инициализация менеджера сообщений с состоянием
		# Начальный системный промпт со всеми действиями - будет обновляться на каждом шаге
		self._message_manager = MessageManager(
			task=self.task,
			system_message=SystemPrompt(
				max_actions_per_step=self.settings.max_actions_per_step,
				override_system_message=override_system_message,
				extend_system_message=extend_system_message,
				use_thinking=self.settings.use_thinking,
				flash_mode=self.settings.flash_mode,
				is_anthropic=is_anthropic,
			).get_system_message(),
			file_system=self.file_system,
			state=self.state.message_manager_state,
			use_thinking=self.settings.use_thinking,
			# Настройки, которые ранее были в MessageManagerSettings
			include_attributes=self.settings.include_attributes,
			sensitive_data=sensitive_data,
			max_history_items=self.settings.max_history_items,
			vision_detail_level=self.settings.vision_detail_level,
			include_tool_call_examples=self.settings.include_tool_call_examples,
			include_recent_events=self.include_recent_events,
			sample_images=self.sample_images,
			llm_screenshot_size=llm_screenshot_size,
		)

		if self.sensitive_data:
			# Проверка наличия доменно-специфичных учетных данных в sensitive_data
			has_domain_specific_credentials = any(isinstance(v, dict) for v in self.sensitive_data.values())

			# Если не настроены allowed_domains, показать предупреждение безопасности
			if not self.browser_profile.allowed_domains:
				self.logger.warning(
					'⚠️ Agent(sensitive_data=••••••••) was provided but Browser(allowed_domains=[...]) is not locked down! ⚠️\n'
					'          ☠️ If the agent visits a malicious website and encounters a prompt-injection attack, your sensitive_data may be exposed!\n\n'
					'   \n'
				)

			# Если используем доменно-специфичные учетные данные, валидируем паттерны доменов
			elif has_domain_specific_credentials:
				# Для доменно-специфичного формата убеждаемся что все паттерны доменов включены в allowed_domains
				domain_patterns = [k for k, v in self.sensitive_data.items() if isinstance(v, dict)]

				# Валидация каждого паттерна домена против allowed_domains
				for domain_pattern in domain_patterns:
					is_allowed = False
					for allowed_domain in self.browser_profile.allowed_domains:
						# Специальные случаи, которые не требуют сопоставления URL
						if domain_pattern == allowed_domain or allowed_domain == '*':
							is_allowed = True
							break

						# Нужно создать примеры URL для сравнения паттернов
						# Извлечение частей домена, игнорируя схему
						pattern_domain = domain_pattern.split('://')[-1] if '://' in domain_pattern else domain_pattern
						allowed_domain_part = allowed_domain.split('://')[-1] if '://' in allowed_domain else allowed_domain

						# Проверка покрыт ли паттерн разрешенным доменом
						# Пример: "google.com" покрывается "*.google.com"
						if pattern_domain == allowed_domain_part or (
							allowed_domain_part.startswith('*.')
							and (
								pattern_domain == allowed_domain_part[2:]
								or pattern_domain.endswith('.' + allowed_domain_part[2:])
							)
						):
							is_allowed = True
							break

					if not is_allowed:
						self.logger.warning(
							f'⚠️ Domain pattern "{domain_pattern}" in sensitive_data is not covered by any pattern in allowed_domains={self.browser_profile.allowed_domains}\n'
							f'   This may be a security risk as credentials could be used on unintended domains.'
						)

		# Колбэки
		self.register_new_step_callback = register_new_step_callback
		self.register_done_callback = register_done_callback
		self.register_should_stop_callback = register_should_stop_callback
		self.register_external_agent_status_raise_error_callback = register_external_agent_status_raise_error_callback

		# Event bus для внутренних событий агента
		self.eventbus = EventBus(name=f'Agent_{str(self.id)[-4:]}')

		if self.settings.save_conversation_path:
			self.settings.save_conversation_path = Path(self.settings.save_conversation_path).expanduser().resolve()
			self.logger.info(f'💬 Сохраняю разговор в {_log_pretty_path(self.settings.save_conversation_path)}')

		# Инициализация отслеживания загрузок
		assert self.browser_session is not None, 'BrowserSession is not set up'
		self.has_downloads_path = self.browser_session.browser_profile.downloads_path is not None
		if self.has_downloads_path:
			self._last_known_downloads: list[str] = []
			self.logger.debug('📁 Инициализирован отслеживание загрузок для агента')

		# Управление паузой на основе событий (вынесено из AgentState для сериализации)
		self._external_pause_event = asyncio.Event()
		self._external_pause_event.set()

	def _enhance_task_with_schema(self, task: str, output_model_schema: type[AgentStructuredOutput] | None) -> str:
		"""Enhance task description with output schema information if provided."""
		if output_model_schema is None:
			return task

		try:
			schema = output_model_schema.model_json_schema()
			import json

			schema_json = json.dumps(schema, indent=2)

			enhancement = f'\nExpected output format: {output_model_schema.__name__}\n{schema_json}'
			return task + enhancement
		except Exception as e:
			self.logger.debug(f'Не удалось распарсить схему вывода: {e}')

		return task

	@property
	def logger(self) -> logging.Logger:
		"""Get instance-specific logger with task ID in the name"""
		# logger может быть вызван в __init__, поэтому не предполагаем что атрибуты self.* инициализированы
		_task_id = task_id[-4:] if (task_id := getattr(self, 'task_id', None)) else '----'
		_browser_session_id = browser_session.id[-4:] if (browser_session := getattr(self, 'browser_session', None)) else '----'
		_current_target_id = (
			browser_session.agent_focus_target_id[-2:]
			if (browser_session := getattr(self, 'browser_session', None)) and browser_session.agent_focus_target_id
			else '--'
		)
		return logging.getLogger(f'agent.Agent🅰 {_task_id} ⇢ 🅑 {_browser_session_id} 🅣 {_current_target_id}')

	@property
	def browser_profile(self) -> BrowserProfile:
		assert self.browser_session is not None, 'BrowserSession is not set up'
		return self.browser_session.browser_profile

	@property
	def is_using_fallback_llm(self) -> bool:
		"""Check if the agent is currently using the fallback LLM."""
		return self._using_fallback_llm

	@property
	def current_llm_model(self) -> str:
		"""Get the model name of the currently active LLM."""
		return self.llm.model if hasattr(self.llm, 'model') else 'unknown'

	async def _check_and_update_downloads(self, context: str = '') -> None:
		"""Check for new downloads and update available file paths."""
		if not self.has_downloads_path:
			return

		assert self.browser_session is not None, 'BrowserSession is not set up'

		try:
			current_downloads = self.browser_session.downloaded_files
			if current_downloads != self._last_known_downloads:
				self._update_available_file_paths(current_downloads)
				self._last_known_downloads = current_downloads
				if context:
					self.logger.debug(f'📁 {context}: Обновлены доступные файлы')
		except Exception as e:
			error_context = f' {context}' if context else ''
			self.logger.debug(f'📁 Не удалось проверить загрузки{error_context}: {type(e).__name__}: {e}')

	def _update_available_file_paths(self, downloads: list[str]) -> None:
		"""Update available_file_paths with downloaded files."""
		if not self.has_downloads_path:
			return

		current_files = set(self.available_file_paths or [])
		new_files = set(downloads) - current_files

		if new_files:
			self.available_file_paths = list(current_files | new_files)

			self.logger.info(
				f'📁 Добавлено {len(new_files)} загруженных файлов в available_file_paths (всего: {len(self.available_file_paths)} файлов)'
			)
			for file_path in new_files:
				self.logger.info(f'📄 Новый файл доступен: {file_path}')
		else:
			self.logger.debug(f'📁 Новые загрузки не обнаружены (отслеживается {len(current_files)} файлов)')

	def _set_file_system(self, file_system_path: str | None = None) -> None:
		"""Заглушка: файловая система отключена в упрощённой версии агента."""
		self.file_system = None
		self.file_system_path = None

	def _set_screenshot_service(self) -> None:
		"""Initialize screenshot service using agent directory"""
		try:
			from agent.screenshots.service import ScreenshotService

			self.screenshot_service = ScreenshotService(self.agent_directory)
			self.logger.debug(f'📸 Сервис скриншотов инициализирован в: {self.agent_directory}/screenshots')
		except Exception as e:
			self.logger.error(f'📸 Не удалось инициализировать сервис скриншотов: {e}.')
			raise e

	def save_file_system_state(self) -> None:
		"""Заглушка: сохранение состояния файловой системы отключено в упрощённой версии."""
		return

	def _set_agent_version_and_source(self, source_override: str | None = None) -> None:
		"""Получить версию и источник сборки агента из pyproject.toml (упрощённо)."""
		# Использование вспомогательной функции для определения версии
		version = get_agent_version()

		# Определение источника
		try:
			package_root = Path(__file__).parent.parent.parent
			repo_files = ['.git', 'README.md', 'docs', 'examples']
			if all(Path(package_root / file).exists() for file in repo_files):
				source = 'git'
			else:
				source = 'pip'
		except Exception as e:
			self.logger.debug(f'Error determining source: {e}')
			source = 'unknown'

		if source_override is not None:
			source = source_override
		self.version = version
		self.source = source

	def _setup_action_models(self) -> None:
		"""Setup dynamic action models from tools registry"""
		# Изначально включать только действия без фильтров
		self.ActionModel = self.tools.registry.create_action_model()
		# Создание выходной модели с динамическими действиями
		if self.settings.flash_mode:
			self.AgentOutput = AgentOutput.type_with_custom_actions_flash_mode(self.ActionModel)
		elif self.settings.use_thinking:
			self.AgentOutput = AgentOutput.type_with_custom_actions(self.ActionModel)
		else:
			self.AgentOutput = AgentOutput.type_with_custom_actions_no_thinking(self.ActionModel)

		# используется для принудительного выполнения действия done когда достигнут max_steps
		self.DoneActionModel = self.tools.registry.create_action_model(include_actions=['done'])
		if self.settings.flash_mode:
			self.DoneAgentOutput = AgentOutput.type_with_custom_actions_flash_mode(self.DoneActionModel)
		elif self.settings.use_thinking:
			self.DoneAgentOutput = AgentOutput.type_with_custom_actions(self.DoneActionModel)
		else:
			self.DoneAgentOutput = AgentOutput.type_with_custom_actions_no_thinking(self.DoneActionModel)

	async def _register_skills_as_actions(self) -> None:
		"""Заглушка: навыки и SkillService отключены в упрощённой версии агента."""
		return

	async def _get_unavailable_skills_info(self) -> str:
		"""Заглушка: навыки и связанные cookies не используются в этой версии агента."""
		return ''

	def add_new_task(self, new_task: str) -> None:
		"""Add a new task to the agent, keeping the same task_id as tasks are continuous"""
		# Просто делегировать менеджеру сообщений - не нужен новый task_id или события
		# Задача продолжается с новыми инструкциями, она не заканчивается и не начинается новая
		self.task = new_task
		self._message_manager.add_new_task(new_task)
		# Пометить как follow-up задачу и пересоздать eventbus (закрывается после каждого запуска)
		self.state.follow_up_task = True
		# Сброс флагов управления чтобы агент мог продолжить
		self.state.stopped = False
		self.state.paused = False
		agent_id_suffix = str(self.id)[-4:].replace('-', '_')
		if agent_id_suffix and agent_id_suffix[0].isdigit():
			agent_id_suffix = 'a' + agent_id_suffix
		self.eventbus = EventBus(name=f'Agent_{agent_id_suffix}')

	async def _check_stop_or_pause(self) -> None:
		"""Check if the agent should stop or pause, and handle accordingly."""

		# Проверка нового should_stop_callback - устанавливает остановленное состояние чисто без исключений
		if self.register_should_stop_callback:
			if await self.register_should_stop_callback():
				self.logger.info('Внешний callback запросил остановку')
				self.state.stopped = True
				raise InterruptedError

		if self.register_external_agent_status_raise_error_callback:
			if await self.register_external_agent_status_raise_error_callback():
				raise InterruptedError

		if self.state.stopped:
			raise InterruptedError

		if self.state.paused:
			raise InterruptedError

	@observe(name='agent.step', ignore_output=True, ignore_input=True)
	@time_execution_async('--step')
	async def step(self, step_info: AgentStepInfo | None = None) -> None:
		"""Execute one step of the task"""
		# Initialize timing first, before any exceptions can occur

		self.step_start_time = time.time()

		browser_state_summary = None

		try:
			# Phase 1: Prepare context and timing
			browser_state_summary = await self._prepare_context(step_info)

			# Phase 2: Get model output and execute actions
			await self._get_next_action(browser_state_summary)
			await self._execute_actions()

			# Phase 3: Post-processing
			await self._post_process()

		except Exception as e:
			# Handle ALL exceptions in one place
			await self._handle_step_error(e)

		finally:
			await self._finalize(browser_state_summary)

	async def _prepare_context(self, step_info: AgentStepInfo | None = None) -> BrowserStateSummary:
		"""Prepare the context for the step: browser state, action models, page actions"""
		# step_start_time is now set in step() method

		assert self.browser_session is not None, 'BrowserSession is not set up'

		self.logger.debug(f'🌐 Шаг {self.state.n_steps}: Получаю состояние браузера...')
		# Always take screenshots for all steps
		self.logger.debug('📸 Запрашиваю состояние браузера с include_screenshot=True')
		browser_state_summary = await self.browser_session.get_browser_state_summary(
			include_screenshot=True,  # always capture even if use_vision=False so that cloud sync is useful (it's fast now anyway)
			include_recent_events=self.include_recent_events,
		)
		if browser_state_summary.screenshot:
			self.logger.debug(f'📸 Получено состояние браузера СО скриншотом, длина: {len(browser_state_summary.screenshot)}')
		else:
			self.logger.debug('📸 Получено состояние браузера БЕЗ скриншота')

		# Check for new downloads after getting browser state (catches PDF auto-downloads and previous step downloads)
		await self._check_and_update_downloads(f'Step {self.state.n_steps}: after getting browser state')

		self.logger.info(f'🌐 URL страницы: {browser_state_summary.url}')
		self.logger.info(f'📄 Title страницы: {browser_state_summary.title}')
		
		if browser_state_summary.dom_state and browser_state_summary.dom_state.selector_map:
			selector_map = browser_state_summary.dom_state.selector_map
			self.logger.info(f'👁️ Агент видит {len(selector_map)} интерактивных элементов на странице')
			
			# Показываем первые 10 элементов чтобы понять что видит агент
			elements_preview = []
			for idx, (element_index, element) in enumerate(list(selector_map.items())[:10]):
				# Пробуем разные способы извлечения текста
				element_text = ''
				if hasattr(element, 'ax_node') and element.ax_node and element.ax_node.name:
					element_text = element.ax_node.name
				elif hasattr(element, 'get_all_children_text'):
					element_text = element.get_all_children_text()
				elif hasattr(element, 'get_meaningful_text_for_llm'):
					element_text = element.get_meaningful_text_for_llm()
				elif hasattr(element, 'node_value'):
					element_text = element.node_value or ''
				
				element_role = ''
				if hasattr(element, 'ax_node') and element.ax_node and element.ax_node.role:
					element_role = element.ax_node.role
				elif hasattr(element, 'tag_name'):
					element_role = element.tag_name or ''
				
				# Обрезаем текст до 50 символов
				text_preview = element_text[:50] + '...' if len(element_text) > 50 else element_text
				elements_preview.append(f'  [{element_index}] {element_role}: {text_preview}')
			
			if elements_preview:
				self.logger.info(f'👁️ Первые элементы которые видит агент:\n' + '\n'.join(elements_preview))
			
			# Используем субагента для почты для логирования контекста (без хардкода)
			if self.email_subagent.is_email_client(browser_state_summary):
				# Извлекаем метаданные письма для логирования
				email_metadata = self.email_subagent.extract_email_metadata(browser_state_summary)
				if email_metadata['is_opened']:
					self.logger.info('📧 Открыто письмо в почтовом клиенте')
					if email_metadata['subject']:
						self.logger.info(f'   Тема: {email_metadata["subject"]}')
					if email_metadata['sender']:
						self.logger.info(f'   Отправитель: {email_metadata["sender"]}')
					if email_metadata['body_preview']:
						body_preview = email_metadata['body_preview'][:200] + '...' if len(email_metadata['body_preview']) > 200 else email_metadata['body_preview']
						self.logger.info(f'   Содержание (первые 200 символов): {body_preview}')
				
				# Проверяем наличие диалогов через субагента
				if self.email_subagent.detect_dialog(browser_state_summary):
					self.logger.warning('⚠️ Обнаружен открытый диалог на странице почтового клиента - он будет автоматически закрыт при следующем действии')

		self._log_step_context(browser_state_summary)
		await self._check_stop_or_pause()

		# Update action models with page-specific actions
		self.logger.debug(f'📝 Шаг {self.state.n_steps}: Обновляю модели действий...')
		await self._update_action_models_for_page(browser_state_summary.url)

		# Get page-specific filtered actions
		page_filtered_actions = self.tools.registry.get_prompt_description(browser_state_summary.url)

		# Page-specific actions will be included directly in the browser_state message
		self.logger.debug(f'💬 Шаг {self.state.n_steps}: Создаю сообщения состояния для контекста...')

		# Skills service removed in simplified version
		unavailable_skills_info = None

		self._message_manager.create_state_messages(
			browser_state_summary=browser_state_summary,
			model_output=self.state.last_model_output,
			result=self.state.last_result,
			step_info=step_info,
			use_vision=self.settings.use_vision,
			page_filtered_actions=page_filtered_actions if page_filtered_actions else None,
			sensitive_data=self.sensitive_data,
			available_file_paths=self.available_file_paths,  # Всегда передавать текущие доступные пути к файлам
			unavailable_skills_info=unavailable_skills_info,
			email_subagent=self.email_subagent,  # Передаем субагента для добавления контекста о почтовых интерфейсах
		)

		await self._force_done_after_last_step(step_info)
		await self._force_done_after_failure()
		return browser_state_summary

	@observe_debug(ignore_input=True, name='get_next_action')
	async def _get_next_action(self, browser_state_summary: BrowserStateSummary) -> None:
		"""Execute LLM interaction with retry logic and handle callbacks"""
		input_messages = self._message_manager.get_messages()
		self.logger.debug(
			f'🤖 Step {self.state.n_steps}: Calling LLM with {len(input_messages)} messages (model: {self.llm.model})...'
		)

		try:
			model_output = await asyncio.wait_for(
				self._get_model_output_with_retry(input_messages), timeout=self.settings.llm_timeout
			)
		except TimeoutError:

			@observe(name='_llm_call_timed_out_with_input')
			async def _log_model_input_to_lmnr(input_messages: list[BaseMessage]) -> None:
				"""Log the model input"""
				pass

			await _log_model_input_to_lmnr(input_messages)

			raise TimeoutError(
				f'LLM call timed out after {self.settings.llm_timeout} seconds. Keep your thinking and output short.'
			)

		self.state.last_model_output = model_output

		# Check again for paused/stopped state after getting model output
		await self._check_stop_or_pause()

		# Handle callbacks and conversation saving
		await self._handle_post_llm_processing(browser_state_summary, input_messages)

		# check again if Ctrl+C was pressed before we commit the output to history
		await self._check_stop_or_pause()

	async def _execute_actions(self) -> None:
		"""Execute the actions from model output"""
		if self.state.last_model_output is None:
			raise ValueError('No model output to execute actions from')

		result = await self.multi_act(self.state.last_model_output.action)
		self.state.last_result = result

	async def _post_process(self) -> None:
		"""Handle post-action processing like download tracking and result logging"""
		assert self.browser_session is not None, 'BrowserSession is not set up'

		# Проверка новых загрузок после выполнения действий
		await self._check_and_update_downloads('after executing actions')

		# проверка ошибок действий и len больше 1
		if self.state.last_result and len(self.state.last_result) == 1 and self.state.last_result[-1].error:
			self.state.consecutive_failures += 1
			self.logger.debug(f'🔄 Шаг {self.state.n_steps}: Последовательные неудачи: {self.state.consecutive_failures}')
			return

		if self.state.consecutive_failures > 0:
			self.state.consecutive_failures = 0
			self.logger.debug(f'🔄 Шаг {self.state.n_steps}: Последовательные неудачи сброшены до: {self.state.consecutive_failures}')

		# Log completion results
		if self.state.last_result and len(self.state.last_result) > 0 and self.state.last_result[-1].is_done:
			success = self.state.last_result[-1].success
			if success:
				# Green color for success
				self.logger.info(f'\n📄 \033[32m Финальный результат:\033[0m \n{self.state.last_result[-1].extracted_content}\n\n')
			else:
				# Red color for failure
				self.logger.info(f'\n📄 \033[31m Финальный результат:\033[0m \n{self.state.last_result[-1].extracted_content}\n\n')
			if self.state.last_result[-1].attachments:
				total_attachments = len(self.state.last_result[-1].attachments)
				for i, file_path in enumerate(self.state.last_result[-1].attachments):
					self.logger.info(f'👉 Attachment {i + 1 if total_attachments > 1 else ""}: {file_path}')

	async def _handle_step_error(self, error: Exception) -> None:
		"""Handle all types of errors that can occur during a step"""

		# Специальная обработка InterruptedError
		if isinstance(error, InterruptedError):
			error_msg = 'The agent was interrupted mid-step' + (f' - {str(error)}' if str(error) else '')
			# NOTE: This is not an error, it's a normal part of the execution when the user interrupts the agent
			self.logger.warning(f'{error_msg}')
			return

		# Обработка всех остальных исключений
		include_trace = self.logger.isEnabledFor(logging.DEBUG)
		error_msg = AgentError.format_error(error, include_trace=include_trace)
		max_total_failures = self.settings.max_failures + int(self.settings.final_response_after_failure)
		prefix = f'❌ Result failed {self.state.consecutive_failures + 1}/{max_total_failures} times: '
		self.state.consecutive_failures += 1

		# Использование WARNING для частичных неудач, ERROR только когда достигнут максимум неудач
		is_final_failure = self.state.consecutive_failures >= max_total_failures
		log_level = logging.ERROR if is_final_failure else logging.WARNING

		if 'Could not parse response' in error_msg or 'tool_use_failed' in error_msg or 'Failed to parse JSON' in error_msg:
			# Логирование ошибок парсинга JSON на уровне debug
			# Обрезаем сообщение об ошибке до разумного размера
			short_error = error_msg[:300] + '...' if len(error_msg) > 300 else error_msg
			self.logger.debug(f'Model: {self.llm.model} failed to parse response: {short_error}')
			# Все равно показываем ошибку пользователю, но только если это финальная ошибка
			if is_final_failure:
				self.logger.log(log_level, f'{prefix}{short_error}')
		else:
			self.logger.log(log_level, f'{prefix}{error_msg}')

		await self._demo_mode_log(f'Step error: {error_msg}', 'error', {'step': self.state.n_steps})
		self.state.last_result = [ActionResult(error=error_msg)]
		return None

	async def _finalize(self, browser_state_summary: BrowserStateSummary | None) -> None:
		"""Finalize the step with history, logging, and events"""
		step_end_time = time.time()
		if not self.state.last_result:
			return

		if browser_state_summary:
			step_interval = None
			if len(self.history.history) > 0:
				last_history_item = self.history.history[-1]

				if last_history_item.metadata:
					previous_end_time = last_history_item.metadata.step_end_time
					previous_start_time = last_history_item.metadata.step_start_time
					step_interval = max(0, previous_end_time - previous_start_time)
			metadata = StepMetadata(
				step_number=self.state.n_steps,
				step_start_time=self.step_start_time,
				step_end_time=step_end_time,
				step_interval=step_interval,
			)

			# Использование _make_history_item как в main ветке
			await self._make_history_item(
				self.state.last_model_output,
				browser_state_summary,
				self.state.last_result,
				metadata,
				state_message=self._message_manager.last_state_message_text,
			)

		# Логирование сводки завершения шага
		summary_message = self._log_step_completion_summary(self.step_start_time, self.state.last_result)
		if summary_message:
			await self._demo_mode_log(summary_message, 'info', {'step': self.state.n_steps})

		# Сохранение состояния файловой системы после завершения шага
		self.save_file_system_state()

		# Эмиссия событий создания и выполнения шага
		if browser_state_summary and self.state.last_model_output:
			# Извлечение ключевых данных шага для события
			actions_data = []
			if self.state.last_model_output.action:
				for action in self.state.last_model_output.action:
					action_dict = action.model_dump() if hasattr(action, 'model_dump') else {}
					actions_data.append(action_dict)

			# Cloud события удалены в упрощенной версии
			# CreateAgentStepEvent был удален

		# Увеличение счетчика шагов после полного завершения шага
		self.state.n_steps += 1

	async def _force_done_after_last_step(self, step_info: AgentStepInfo | None = None) -> None:
		"""Handle special processing for the last step"""
		if step_info and step_info.is_last_step():
			# Добавление предупреждения о последнем шаге при необходимости
			msg = 'You reached max_steps - this is your last step. Your only tool available is the "done" tool. No other tool is available. All other tools which you see in history or examples are not available.'
			msg += '\nIf the task is not yet fully finished as requested by the user, set success in "done" to false! E.g. if not all steps are fully completed. Else success to true.'
			msg += '\nInclude everything you found out for the ultimate task in the done text.'
			self.logger.debug('Last step finishing up')
			self._message_manager._add_context_message(UserMessage(content=msg))
			self.AgentOutput = self.DoneAgentOutput

	async def _force_done_after_failure(self) -> None:
		"""Принудительное завершение после неудачи"""
		# Создание сообщения восстановления
		if self.state.consecutive_failures >= self.settings.max_failures and self.settings.final_response_after_failure:
			msg = f'You failed {self.settings.max_failures} times. Therefore we terminate the agent.'
			msg += '\nYour only tool available is the "done" tool. No other tool is available. All other tools which you see in history or examples are not available.'
			msg += '\nIf the task is not yet fully finished as requested by the user, set success in "done" to false! E.g. if not all steps are fully completed. Else success to true.'
			msg += '\nInclude everything you found out for the ultimate task in the done text.'

			self.logger.debug('Force done action, because we reached max_failures.')
			self._message_manager._add_context_message(UserMessage(content=msg))
			self.AgentOutput = self.DoneAgentOutput

	@observe(ignore_input=True, ignore_output=False)
	async def _judge_trace(self) -> JudgementResult | None:
		"""Заглушка: judge-оценка отключена в упрощённой версии агента."""
		return None

	async def _judge_and_log(self) -> None:
		"""Заглушка: judge-оценка отключена."""
		return

	async def _get_model_output_with_retry(self, input_messages: list[BaseMessage]) -> AgentOutput:
		"""Get model output with retry logic for empty actions"""
		model_output = await self.get_model_output(input_messages)
		self.logger.debug(
			f'✅ Step {self.state.n_steps}: Got LLM response with {len(model_output.action) if model_output.action else 0} actions'
		)

		if (
			not model_output.action
			or not isinstance(model_output.action, list)
			or all(action.model_dump() == {} for action in model_output.action)
		):
			self.logger.warning('Model returned empty action. Retrying...')

			clarification_message = UserMessage(
				content='You forgot to return an action. Please respond with a valid JSON action according to the expected schema with your assessment and next actions.'
			)

			retry_messages = input_messages + [clarification_message]
			model_output = await self.get_model_output(retry_messages)

			if not model_output.action or all(action.model_dump() == {} for action in model_output.action):
				self.logger.warning('Model still returned empty after retry. Inserting safe noop action.')
				action_instance = self.ActionModel()
				setattr(
					action_instance,
					'done',
					{
						'success': False,
						'text': 'No next action returned by LLM!',
					},
				)
				model_output.action = [action_instance]

		return model_output

	async def _handle_post_llm_processing(
		self,
		browser_state_summary: BrowserStateSummary,
		input_messages: list[BaseMessage],
	) -> None:
		"""Handle callbacks and conversation saving after LLM interaction"""
		if self.register_new_step_callback and self.state.last_model_output:
			if inspect.iscoroutinefunction(self.register_new_step_callback):
				await self.register_new_step_callback(
					browser_state_summary,
					self.state.last_model_output,
					self.state.n_steps,
				)
			else:
				self.register_new_step_callback(
					browser_state_summary,
					self.state.last_model_output,
					self.state.n_steps,
				)

		if self.settings.save_conversation_path and self.state.last_model_output:
			# Обработка save_conversation_path как директории (согласованно с другими путями записи)
			conversation_dir = Path(self.settings.save_conversation_path)
			conversation_filename = f'conversation_{self.id}_{self.state.n_steps}.txt'
			target = conversation_dir / conversation_filename
			await save_conversation(
				input_messages,
				self.state.last_model_output,
				target,
				self.settings.save_conversation_path_encoding,
			)

	async def _make_history_item(
		self,
		model_output: AgentOutput | None,
		browser_state_summary: BrowserStateSummary,
		result: list[ActionResult],
		metadata: StepMetadata | None = None,
		state_message: str | None = None,
	) -> None:
		"""Create and store history item"""

		if model_output:
			interacted_elements = AgentHistory.get_interacted_element(model_output, browser_state_summary.dom_state.selector_map)
		else:
			interacted_elements = [None]

		# Store screenshot and get path
		screenshot_path = None
		if browser_state_summary.screenshot:
			self.logger.debug(
				f'📸 Storing screenshot for step {self.state.n_steps}, screenshot length: {len(browser_state_summary.screenshot)}'
			)
			screenshot_path = await self.screenshot_service.store_screenshot(browser_state_summary.screenshot, self.state.n_steps)
			self.logger.debug(f'📸 Screenshot stored at: {screenshot_path}')
		else:
			self.logger.debug(f'📸 No screenshot in browser_state_summary for step {self.state.n_steps}')

		state_history = BrowserStateHistory(
			url=browser_state_summary.url,
			title=browser_state_summary.title,
			tabs=browser_state_summary.tabs,
			interacted_element=interacted_elements,
			screenshot_path=screenshot_path,
		)

		history_item = AgentHistory(
			model_output=model_output,
			result=result,
			state=state_history,
			metadata=metadata,
			state_message=state_message,
		)

		self.history.add_item(history_item)

	def _remove_think_tags(self, text: str) -> str:
		THINK_TAGS = re.compile(r'<think>.*?</think>', re.DOTALL)
		STRAY_CLOSE_TAG = re.compile(r'.*?</think>', re.DOTALL)
		# Step 1: Remove well-formed <think>...</think>
		text = re.sub(THINK_TAGS, '', text)
		# Step 2: If there's an unmatched closing tag </think>,
		#         remove everything up to and including that.
		text = re.sub(STRAY_CLOSE_TAG, '', text)
		return text.strip()

	# region - URL replacement
	def _replace_urls_in_text(self, text: str) -> tuple[str, dict[str, str]]:
		"""Replace URLs in a text string"""

		replaced_urls: dict[str, str] = {}

		def replace_url(match: re.Match) -> str:
			"""Url can only have 1 query and 1 fragment"""
			import hashlib

			original_url = match.group(0)

			# Find where the query/fragment starts
			query_start = original_url.find('?')
			fragment_start = original_url.find('#')

			# Find the earliest position of query or fragment
			after_path_start = len(original_url)  # Default: no query/fragment
			if query_start != -1:
				after_path_start = min(after_path_start, query_start)
			if fragment_start != -1:
				after_path_start = min(after_path_start, fragment_start)

			# Split URL into base (up to path) and after_path (query + fragment)
			base_url = original_url[:after_path_start]
			after_path = original_url[after_path_start:]

			# If after_path is within the limit, don't shorten
			if len(after_path) <= self._url_shortening_limit:
				return original_url

			# If after_path is too long, truncate and add hash
			if after_path:
				truncated_after_path = after_path[: self._url_shortening_limit]
				# Create a short hash of the full after_path content
				hash_obj = hashlib.md5(after_path.encode('utf-8'))
				short_hash = hash_obj.hexdigest()[:7]
				# Create shortened URL
				shortened = f'{base_url}{truncated_after_path}...{short_hash}'
				# Only use shortened URL if it's actually shorter than the original
				if len(shortened) < len(original_url):
					replaced_urls[shortened] = original_url
					return shortened

			return original_url

		return URL_PATTERN.sub(replace_url, text), replaced_urls

	def _process_messsages_and_replace_long_urls_shorter_ones(self, input_messages: list[BaseMessage]) -> dict[str, str]:
		"""Replace long URLs with shorter ones
		? @dev edits input_messages in place

		returns:
			tuple[filtered_input_messages, urls we replaced {shorter_url: original_url}]
		"""
		from agent.llm.messages import AssistantMessage, UserMessage

		urls_replaced: dict[str, str] = {}

		# Process each message, in place
		for message in input_messages:
			# no need to process SystemMessage, we have control over that anyway
			if isinstance(message, (UserMessage, AssistantMessage)):
				if isinstance(message.content, str):
					# Simple string content
					message.content, replaced_urls = self._replace_urls_in_text(message.content)
					urls_replaced.update(replaced_urls)

				elif isinstance(message.content, list):
					# List of content parts
					for part in message.content:
						if isinstance(part, ContentPartTextParam):
							part.text, replaced_urls = self._replace_urls_in_text(part.text)
							urls_replaced.update(replaced_urls)

		return urls_replaced

	@staticmethod
	def _recursive_process_all_strings_inside_pydantic_model(model: BaseModel, url_replacements: dict[str, str]) -> None:
		"""Recursively process all strings inside a Pydantic model, replacing shortened URLs with originals in place."""
		for field_name, field_value in model.__dict__.items():
			if isinstance(field_value, str):
				# Replace shortened URLs with original URLs in string
				processed_string = Agent._replace_shortened_urls_in_string(field_value, url_replacements)
				setattr(model, field_name, processed_string)
			elif isinstance(field_value, BaseModel):
				# Recursively process nested Pydantic models
				Agent._recursive_process_all_strings_inside_pydantic_model(field_value, url_replacements)
			elif isinstance(field_value, dict):
				# Process dictionary values in place
				Agent._recursive_process_dict(field_value, url_replacements)
			elif isinstance(field_value, (list, tuple)):
				processed_value = Agent._recursive_process_list_or_tuple(field_value, url_replacements)
				setattr(model, field_name, processed_value)

	@staticmethod
	def _recursive_process_dict(dictionary: dict, url_replacements: dict[str, str]) -> None:
		"""Helper method to process dictionaries."""
		for k, v in dictionary.items():
			if isinstance(v, str):
				dictionary[k] = Agent._replace_shortened_urls_in_string(v, url_replacements)
			elif isinstance(v, BaseModel):
				Agent._recursive_process_all_strings_inside_pydantic_model(v, url_replacements)
			elif isinstance(v, dict):
				Agent._recursive_process_dict(v, url_replacements)
			elif isinstance(v, (list, tuple)):
				dictionary[k] = Agent._recursive_process_list_or_tuple(v, url_replacements)

	@staticmethod
	def _recursive_process_list_or_tuple(container: list | tuple, url_replacements: dict[str, str]) -> list | tuple:
		"""Helper method to process lists and tuples."""
		if isinstance(container, tuple):
			# For tuples, create a new tuple with processed items
			processed_items = []
			for item in container:
				if isinstance(item, str):
					processed_items.append(Agent._replace_shortened_urls_in_string(item, url_replacements))
				elif isinstance(item, BaseModel):
					Agent._recursive_process_all_strings_inside_pydantic_model(item, url_replacements)
					processed_items.append(item)
				elif isinstance(item, dict):
					Agent._recursive_process_dict(item, url_replacements)
					processed_items.append(item)
				elif isinstance(item, (list, tuple)):
					processed_items.append(Agent._recursive_process_list_or_tuple(item, url_replacements))
				else:
					processed_items.append(item)
			return tuple(processed_items)
		else:
			# For lists, modify in place
			for i, item in enumerate(container):
				if isinstance(item, str):
					container[i] = Agent._replace_shortened_urls_in_string(item, url_replacements)
				elif isinstance(item, BaseModel):
					Agent._recursive_process_all_strings_inside_pydantic_model(item, url_replacements)
				elif isinstance(item, dict):
					Agent._recursive_process_dict(item, url_replacements)
				elif isinstance(item, (list, tuple)):
					container[i] = Agent._recursive_process_list_or_tuple(item, url_replacements)
			return container

	@staticmethod
	def _replace_shortened_urls_in_string(text: str, url_replacements: dict[str, str]) -> str:
		"""Replace all shortened URLs in a string with their original URLs."""
		result = text
		for shortened_url, original_url in url_replacements.items():
			result = result.replace(shortened_url, original_url)
		return result

	# endregion - URL replacement

	@time_execution_async('--get_next_action')
	@observe_debug(ignore_input=True, ignore_output=True, name='get_model_output')
	async def get_model_output(self, input_messages: list[BaseMessage]) -> AgentOutput:
		"""Get next action from LLM based on current state"""

		urls_replaced = self._process_messsages_and_replace_long_urls_shorter_ones(input_messages)

		# Build kwargs for ainvoke
		# Примечание: некоторые LLM-провайдеры умеют автоматически строить описания действий на основе output_format
		kwargs: dict = {'output_format': self.AgentOutput}

		try:
			response = await self.llm.ainvoke(input_messages, **kwargs)
			parsed: AgentOutput = response.completion  # type: ignore[assignment]

			# Replace any shortened URLs in the LLM response back to original URLs
			if urls_replaced:
				self._recursive_process_all_strings_inside_pydantic_model(parsed, urls_replaced)

			# cut the number of actions to max_actions_per_step if needed
			if len(parsed.action) > self.settings.max_actions_per_step:
				parsed.action = parsed.action[: self.settings.max_actions_per_step]

			if not (hasattr(self.state, 'paused') and (self.state.paused or self.state.stopped)):
				log_response(parsed, self.tools.registry.registry, self.logger)
				await self._broadcast_model_state(parsed)

			self._log_next_action_summary(parsed)
			return parsed
		except ValidationError:
			# Just re-raise - Pydantic's validation errors are already descriptive
			raise
		except (ModelRateLimitError, ModelProviderError) as e:
			# Check if we can switch to a fallback LLM
			if not self._try_switch_to_fallback_llm(e):
				# No fallback available, re-raise the original error
				raise
			# Retry with the fallback LLM
			return await self.get_model_output(input_messages)

	def _try_switch_to_fallback_llm(self, error: ModelRateLimitError | ModelProviderError) -> bool:
		"""
		Attempt to switch to a fallback LLM after a rate limit or provider error.

		Returns True if successfully switched to a fallback, False if no fallback available.
		Once switched, the agent will use the fallback LLM for the rest of the run.
		"""
		# Already using fallback - can't switch again
		if self._using_fallback_llm:
			# Обрезаем сообщение об ошибке, чтобы не выводить огромные пасты валидации
			error_msg_short = error.message[:200] + '...' if len(error.message) > 200 else error.message
			self.logger.warning(
				f'⚠️ Fallback LLM also failed ({type(error).__name__}: {error_msg_short}), no more fallbacks available'
			)
			return False

		# Check if error is retryable (rate limit, auth errors, or server errors)
		# 401: API key invalid/expired - fallback to different provider
		# 402: Insufficient credits/payment required - fallback to different provider
		# 429: Rate limit exceeded
		# 500, 502, 503, 504: Server errors
		retryable_status_codes = {401, 402, 429, 500, 502, 503, 504}
		is_retryable = isinstance(error, ModelRateLimitError) or (
			hasattr(error, 'status_code') and error.status_code in retryable_status_codes
		)

		if not is_retryable:
			return False

		# Check if we have a fallback LLM configured
		if self._fallback_llm is None:
			# Обрезаем сообщение об ошибке, чтобы не выводить огромные пасты валидации
			error_msg_short = error.message[:200] + '...' if len(error.message) > 200 else error.message
			self.logger.warning(f'⚠️ LLM error ({type(error).__name__}: {error_msg_short}) but no fallback_llm configured')
			return False

		self._log_fallback_switch(error, self._fallback_llm)

		# Switch to the fallback LLM
		self.llm = self._fallback_llm
		self._using_fallback_llm = True

		# Register the fallback LLM for token cost tracking
		self.token_cost_service.register_llm(self._fallback_llm)

		return True

	def _log_fallback_switch(self, error: ModelRateLimitError | ModelProviderError, fallback: BaseChatModel) -> None:
		"""Log when switching to a fallback LLM."""
		original_model = self._original_llm.model if hasattr(self._original_llm, 'model') else 'unknown'
		fallback_model = fallback.model if hasattr(fallback, 'model') else 'unknown'
		error_type = type(error).__name__
		status_code = getattr(error, 'status_code', 'N/A')

		self.logger.warning(
			f'⚠️ Primary LLM ({original_model}) failed with {error_type} (status={status_code}), '
			f'switching to fallback LLM ({fallback_model})'
		)

	async def _log_agent_run(self) -> None:
		"""Log the agent run"""
		# Blue color for task
		self.logger.info(f'\033[34m🎯 Task: {self.task}\033[0m')

		self.logger.debug(f'🤖 Версия библиотеки агента {self.version} ({self.source})')

		# Check for latest version and log upgrade message if needed
		if CONFIG.AGENT_VERSION_CHECK:
			latest_version = await check_latest_agent_version()
			if latest_version and latest_version != self.version:
				self.logger.info(
					f'📦 Доступна новая версия агента: {latest_version} (текущая: {self.version}).'
				)

	def _log_first_step_startup(self) -> None:
		"""Log startup message only on the first step"""
		if len(self.history.history) == 0:
			self.logger.info(
				f'Запуск агента версии {self.version} с провайдером={self.llm.provider} и моделью={self.llm.model}'
			)

	def _log_step_context(self, browser_state_summary: BrowserStateSummary) -> None:
		"""Log step context information"""
		url = browser_state_summary.url if browser_state_summary else ''
		url_short = url[:50] + '...' if len(url) > 50 else url
		interactive_count = len(browser_state_summary.dom_state.selector_map) if browser_state_summary else 0
		self.logger.info('\n')
		self.logger.info(f'📍 Step {self.state.n_steps}:')
		self.logger.debug(f'Evaluating page with {interactive_count} interactive elements on: {url_short}')

	def _log_next_action_summary(self, parsed: 'AgentOutput') -> None:
		"""Log a comprehensive summary of the next action(s)"""
		if not (self.logger.isEnabledFor(logging.DEBUG) and parsed.action):
			return

		action_count = len(parsed.action)

		# Collect action details
		action_details = []
		for i, action in enumerate(parsed.action):
			action_data = action.model_dump(exclude_unset=True)
			action_name = next(iter(action_data.keys())) if action_data else 'unknown'
			action_params = action_data.get(action_name, {}) if action_data else {}

			# Format key parameters concisely
			param_summary = []
			if isinstance(action_params, dict):
				for key, value in action_params.items():
					if key == 'index':
						param_summary.append(f'#{value}')
					elif key == 'text' and isinstance(value, str):
						text_preview = value[:30] + '...' if len(value) > 30 else value
						param_summary.append(f'text="{text_preview}"')
					elif key == 'url':
						param_summary.append(f'url="{value}"')
					elif key == 'success':
						param_summary.append(f'success={value}')
					elif isinstance(value, (str, int, bool)):
						val_str = str(value)[:30] + '...' if len(str(value)) > 30 else str(value)
						param_summary.append(f'{key}={val_str}')

			param_str = f'({", ".join(param_summary)})' if param_summary else ''
			action_details.append(f'{action_name}{param_str}')

	def _prepare_demo_message(self, message: str, limit: int = 600) -> str:
		# Previously truncated long entries; keep full text for better context in demo panel
		return message.strip()

	async def _demo_mode_log(self, message: str, level: str = 'info', metadata: dict[str, Any] | None = None) -> None:
		if not self._demo_mode_enabled or not message or self.browser_session is None:
			return
		try:
			await self.browser_session.send_demo_mode_log(
				message=self._prepare_demo_message(message),
				level=level,
				metadata=metadata or {},
			)
		except Exception as exc:
			self.logger.debug(f'[DemoMode] Failed to send overlay log: {exc}')

	async def _broadcast_model_state(self, parsed: 'AgentOutput') -> None:
		if not self._demo_mode_enabled:
			return

		state = parsed.current_state
		step_meta = {'step': self.state.n_steps}

		if state.thinking:
			await self._demo_mode_log(state.thinking, 'thought', step_meta)

		if state.evaluation_previous_goal:
			eval_text = state.evaluation_previous_goal
			level = 'success' if 'success' in eval_text.lower() else 'warning' if 'failure' in eval_text.lower() else 'info'
			await self._demo_mode_log(eval_text, level, step_meta)

		if state.memory:
			await self._demo_mode_log(f'Memory: {state.memory}', 'info', step_meta)

		if state.next_goal:
			await self._demo_mode_log(f'Next goal: {state.next_goal}', 'info', step_meta)

	def _log_step_completion_summary(self, step_start_time: float, result: list[ActionResult]) -> str | None:
		"""Log step completion summary with action count, timing, and success/failure stats"""
		if not result:
			return None

		step_duration = time.time() - step_start_time
		action_count = len(result)

		# Count success and failures
		success_count = sum(1 for r in result if not r.error)
		failure_count = action_count - success_count

		# Format success/failure indicators
		success_indicator = f'✅ {success_count}' if success_count > 0 else ''
		failure_indicator = f'❌ {failure_count}' if failure_count > 0 else ''
		status_parts = [part for part in [success_indicator, failure_indicator] if part]
		status_str = ' | '.join(status_parts) if status_parts else '✅ 0'

		message = (
			f'📍 Step {self.state.n_steps}: Ran {action_count} action{"" if action_count == 1 else "s"} '
			f'in {step_duration:.2f}s: {status_str}'
		)
		self.logger.debug(message)
		return message

	def _log_final_outcome_messages(self) -> None:
		"""Log helpful messages to user based on agent run outcome"""
		# Check if agent failed
		is_successful = self.history.is_successful()

		if is_successful is False or is_successful is None:
			# Get final result to check for specific failure reasons
			final_result = self.history.final_result()
			final_result_str = str(final_result).lower() if final_result else ''

			# Check for captcha/cloudflare related failures
			captcha_keywords = ['captcha', 'cloudflare', 'recaptcha', 'challenge', 'bot detection', 'access denied']
			has_captcha_issue = any(keyword in final_result_str for keyword in captcha_keywords)

			if has_captcha_issue:
				# Suggest use_cloud=True for captcha/cloudflare issues
				task_preview = self.task[:10] if len(self.task) > 10 else self.task
				self.logger.info('')
				self.logger.info('Failed because of CAPTCHA? For better browser stealth, try:')
				self.logger.info(f'   agent = Agent(task="{task_preview}...", browser=Browser())')

			# General failure message
			self.logger.info('')
			self.logger.info('Did the Agent not work as expected? Let us fix this!')
			self.logger.info('   Если ошибка повторяется, сохраните лог и пример задачи для дальнейшей отладки.')

	def _log_agent_event(self, max_steps: int, agent_run_error: str | None = None) -> None:
		"""Заглушка: отправка телеметрии отключена в упрощённой версии."""
		return

	async def take_step(self, step_info: AgentStepInfo | None = None) -> tuple[bool, bool]:
		"""Take a step

		Returns:
		        Tuple[bool, bool]: (is_done, is_valid)
		"""
		if step_info is not None and step_info.step_number == 0:
			# First step
			self._log_first_step_startup()
			# Normally there was no try catch here but the callback can raise an InterruptedError which we skip
			try:
				await self._execute_initial_actions()
			except InterruptedError:
				pass
			except Exception as e:
				raise e

		await self.step(step_info)

		if self.history.is_done():
			await self.log_completion()

			# Run judge before done callback if enabled
			# Judge-оценка отключена в упрощённой версии

			if self.register_done_callback:
				if inspect.iscoroutinefunction(self.register_done_callback):
					await self.register_done_callback(self.history)
				else:
					self.register_done_callback(self.history)
			return True, True

		return False, False

	def _extract_start_url(self, task: str) -> str | None:
		"""Extract URL from task string using naive pattern matching."""

		import re

		# Remove email addresses from task before looking for URLs
		task_without_emails = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', task)

		# Look for common URL patterns
		patterns = [
			r'https?://[^\s<>"\']+',  # Full URLs with http/https
			r'(?:www\.)?[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}(?:/[^\s<>"\']*)?',  # Domain names with subdomains and optional paths
		]

		# File extensions that should be excluded from URL detection
		# These are likely files rather than web pages to navigate to
		excluded_extensions = {
			# Documents
			'pdf',
			'doc',
			'docx',
			'xls',
			'xlsx',
			'ppt',
			'pptx',
			'odt',
			'ods',
			'odp',
			# Text files
			'txt',
			'md',
			'csv',
			'json',
			'xml',
			'yaml',
			'yml',
			# Archives
			'zip',
			'rar',
			'7z',
			'tar',
			'gz',
			'bz2',
			'xz',
			# Images
			'jpg',
			'jpeg',
			'png',
			'gif',
			'bmp',
			'svg',
			'webp',
			'ico',
			# Audio/Video
			'mp3',
			'mp4',
			'avi',
			'mkv',
			'mov',
			'wav',
			'flac',
			'ogg',
			# Code/Data
			'py',
			'js',
			'css',
			'java',
			'cpp',
			# Academic/Research
			'bib',
			'bibtex',
			'tex',
			'latex',
			'cls',
			'sty',
			# Other common file types
			'exe',
			'msi',
			'dmg',
			'pkg',
			'deb',
			'rpm',
			'iso',
			# GitHub/Project paths
			'polynomial',
		}

		excluded_words = {
			'never',
			'dont',
			'not',
			"don't",
		}

		found_urls = []
		for pattern in patterns:
			matches = re.finditer(pattern, task_without_emails)
			for match in matches:
				url = match.group(0)
				original_position = match.start()  # Store original position before URL modification

				# Remove trailing punctuation that's not part of URLs
				url = re.sub(r'[.,;:!?()\[\]]+$', '', url)

				# Check if URL ends with a file extension that should be excluded
				url_lower = url.lower()
				should_exclude = False
				for ext in excluded_extensions:
					if f'.{ext}' in url_lower:
						should_exclude = True
						break

				if should_exclude:
					self.logger.debug(f'Excluding URL with file extension from auto-navigation: {url}')
					continue

				# If in the 20 characters before the url position is a word in excluded_words skip to avoid "Never go to this url"
				context_start = max(0, original_position - 20)
				context_text = task_without_emails[context_start:original_position]
				if any(word.lower() in context_text.lower() for word in excluded_words):
					self.logger.debug(
						f'Excluding URL with word in excluded words from auto-navigation: {url} (context: "{context_text.strip()}")'
					)
					continue

				# Add https:// if missing (after excluded words check to avoid position calculation issues)
				if not url.startswith(('http://', 'https://')):
					url = 'https://' + url

				found_urls.append(url)

		unique_urls = list(set(found_urls))
		# If multiple URLs found, skip directly_open_urling
		if len(unique_urls) > 1:
			self.logger.debug(f'Multiple URLs found ({len(found_urls)}), skipping directly_open_url to avoid ambiguity')
			return None

		# If exactly one URL found, return it
		if len(unique_urls) == 1:
			return unique_urls[0]

		return None

	async def _execute_step(
		self,
		step: int,
		max_steps: int,
		step_info: AgentStepInfo,
		on_step_start: AgentHookFunc | None = None,
		on_step_end: AgentHookFunc | None = None,
	) -> bool:
		"""
		Execute a single step with timeout.

		Returns:
			bool: True if task is done, False otherwise
		"""
		if on_step_start is not None:
			await on_step_start(self)

		await self._demo_mode_log(
			f'Starting step {step + 1}/{max_steps}',
			'info',
			{'step': step + 1, 'total_steps': max_steps},
		)

		self.logger.debug(f'🚶 Starting step {step + 1}/{max_steps}...')

		try:
			await asyncio.wait_for(
				self.step(step_info),
				timeout=self.settings.step_timeout,
			)
			self.logger.debug(f'✅ Completed step {step + 1}/{max_steps}')
		except TimeoutError:
			# Handle step timeout gracefully
			error_msg = f'Step {step + 1} timed out after {self.settings.step_timeout} seconds'
			self.logger.error(f'⏰ {error_msg}')
			await self._demo_mode_log(error_msg, 'error', {'step': step + 1})
			self.state.consecutive_failures += 1
			self.state.last_result = [ActionResult(error=error_msg)]

		if on_step_end is not None:
			await on_step_end(self)

		if self.history.is_done():
			await self.log_completion()

			# Run judge before done callback if enabled
			# Judge-оценка отключена в упрощённой версии

			if self.register_done_callback:
				if inspect.iscoroutinefunction(self.register_done_callback):
					await self.register_done_callback(self.history)
				else:
					self.register_done_callback(self.history)

			return True

		return False

	@observe(name='agent.run', ignore_input=True, ignore_output=True)
	@time_execution_async('--run')
	async def run(
		self,
		max_steps: int = 100,
		on_step_start: AgentHookFunc | None = None,
		on_step_end: AgentHookFunc | None = None,
	) -> AgentHistoryList[AgentStructuredOutput]:
		"""Execute the task with maximum number of steps"""

		loop = asyncio.get_event_loop()
		agent_run_error: str | None = None  # Initialize error tracking variable
		should_delay_close = False

		# Set up the  signal handler with callbacks specific to this agent
		from agent.utils import SignalHandler

		signal_handler = SignalHandler(
			loop=loop,
			pause_callback=self.pause,
			resume_callback=self.resume,
			custom_exit_callback=None,
			exit_on_second_int=True,
		)
		signal_handler.register()

		try:
			await self._log_agent_run()

			self.logger.debug(
				f'🔧 Agent setup: Agent Session ID {self.session_id[-4:]}, Task ID {self.task_id[-4:]}, Browser Session ID {self.browser_session.id[-4:] if self.browser_session else "None"} {"(connecting via CDP)" if (self.browser_session and self.browser_session.cdp_url) else "(launching local browser)"}'
			)

			# Initialize timing for session and task
			self._session_start_time = time.time()
			self._task_start_time = self._session_start_time  # Initialize task start time

			# Only dispatch session events if this is the first run
			if not self.state.session_initialized:
				# Cloud events removed in simplified version
				self.state.session_initialized = True

			# Log startup message on first step (only if we haven't already done steps)
			self._log_first_step_startup()
			# Start browser session and attach watchdogs
			await self.browser_session.start()
			if self._demo_mode_enabled:
				await self._demo_mode_log(f'Started task: {self.task}', 'info', {'tag': 'task'})
				await self._demo_mode_log(
					'Demo mode active - follow the side panel for live thoughts and actions.',
					'info',
					{'tag': 'status'},
				)

			# Register skills as actions if SkillService is configured
			await self._register_skills_as_actions()

			# Пересчитываем initial_actions если задача была изменена после инициализации
			# (например, в console_interface агент создаётся с пустой задачей, а потом задача задаётся)
			if self.directly_open_url and not self.state.follow_up_task and not self.initial_actions:
				initial_url = self._extract_start_url(self.task)
				if initial_url:
					self.logger.info(f'🔗 Найден URL в задаче: {initial_url}, добавляю как начальное действие...')
					self.initial_url = initial_url
					self.initial_actions = self._convert_initial_actions([{'navigate': {'url': initial_url, 'new_tab': False}}])

			# Normally there was no try catch here but the callback can raise an InterruptedError
			try:
				await self._execute_initial_actions()
			except InterruptedError:
				pass
			except Exception as e:
				raise e

			self.logger.debug(
				f'🔄 Starting main execution loop with max {max_steps} steps (currently at step {self.state.n_steps})...'
			)
			while self.state.n_steps <= max_steps:
				current_step = self.state.n_steps - 1  # Convert to 0-indexed for step_info

				# Use the consolidated pause state management
				if self.state.paused:
					self.logger.debug(f'⏸️ Step {self.state.n_steps}: Agent paused, waiting to resume...')
					await self._external_pause_event.wait()
					signal_handler.reset()

				# Check if we should stop due to too many failures, if final_response_after_failure is True, we try one last time
				if (self.state.consecutive_failures) >= self.settings.max_failures + int(
					self.settings.final_response_after_failure
				):
					self.logger.error(f'❌ Stopping due to {self.settings.max_failures} consecutive failures')
					agent_run_error = f'Stopped due to {self.settings.max_failures} consecutive failures'
					break

				# Check control flags before each step
				if self.state.stopped:
					self.logger.info('🛑 Agent stopped')
					agent_run_error = 'Agent stopped programmatically'
					break

				step_info = AgentStepInfo(step_number=current_step, max_steps=max_steps)
				is_done = await self._execute_step(current_step, max_steps, step_info, on_step_start, on_step_end)

				if is_done:
					# Agent has marked the task as done
					if self._demo_mode_enabled and self.history.history:
						final_result_text = self.history.final_result() or 'Task completed'
						await self._demo_mode_log(f'Final Result: {final_result_text}', 'success', {'tag': 'task'})

					should_delay_close = True
					break
			else:
				agent_run_error = 'Failed to complete task in maximum steps'

				self.history.add_item(
					AgentHistory(
						model_output=None,
						result=[ActionResult(error=agent_run_error, include_in_memory=True)],
						state=BrowserStateHistory(
							url='',
							title='',
							tabs=[],
							interacted_element=[],
							screenshot_path=None,
						),
						metadata=None,
					)
				)

				self.logger.info(f'❌ {agent_run_error}')

			self.history.usage = await self.token_cost_service.get_usage_summary()

			# set the model output schema and call it on the fly
			if self.history._output_model_schema is None and self.output_model_schema is not None:
				self.history._output_model_schema = self.output_model_schema

			return self.history

		except KeyboardInterrupt:
			# Already handled by our signal handler, but catch any direct KeyboardInterrupt as well
			self.logger.debug('Got KeyboardInterrupt during execution, returning current history')
			agent_run_error = 'KeyboardInterrupt'

			self.history.usage = await self.token_cost_service.get_usage_summary()

			return self.history

		except Exception as e:
			self.logger.error(f'Agent run failed with exception: {e}', exc_info=True)
			agent_run_error = str(e)
			raise e

		finally:
			if should_delay_close and self._demo_mode_enabled and agent_run_error is None:
				await asyncio.sleep(30)
			if agent_run_error:
				await self._demo_mode_log(f'Agent stopped: {agent_run_error}', 'error', {'tag': 'run'})
			# Log token usage summary
			await self.token_cost_service.log_usage_summary()

			# Unregister signal handlers before cleanup
			signal_handler.unregister()

			# Telemetry and cloud events removed in simplified version

			# Generate GIF if needed before stopping event bus
			if self.settings.generate_gif:
				output_path: str = 'agent_history.gif'
				if isinstance(self.settings.generate_gif, str):
					output_path = self.settings.generate_gif

				# Lazy import gif module to avoid heavy startup cost
				from agent.agent.gif import create_history_gif

				create_history_gif(task=self.task, history=self.history, output_path=output_path)

				# Cloud events removed in simplified version
				# CreateAgentOutputFileEvent was removed

			# Log final messages to user based on outcome
			self._log_final_outcome_messages()

			# Stop the event bus gracefully, waiting for all events to be processed
			# Использование более длинного таймаута для избежания блокировок при тестировании
			await self.eventbus.stop(timeout=3.0)

			await self.close()

	@observe_debug(ignore_input=True, ignore_output=True)
	@time_execution_async('--multi_act')
	async def multi_act(self, actions: list[ActionModel]) -> list[ActionResult]:
		"""Execute multiple actions"""
		results: list[ActionResult] = []
		time_elapsed = 0
		total_actions = len(actions)

		assert self.browser_session is not None, 'BrowserSession is not set up'
		try:
			if (
				self.browser_session._cached_browser_state_summary is not None
				and self.browser_session._cached_browser_state_summary.dom_state is not None
			):
				cached_selector_map = dict(self.browser_session._cached_browser_state_summary.dom_state.selector_map)
				cached_element_hashes = {e.parent_branch_hash() for e in cached_selector_map.values()}
			else:
				cached_selector_map = {}
				cached_element_hashes = set()
		except Exception as e:
			self.logger.error(f'Error getting cached selector map: {e}')
			cached_selector_map = {}
			cached_element_hashes = set()

		for i, action in enumerate(actions):
			if i > 0:
				# ONLY ALLOW TO CALL `done` IF IT IS A SINGLE ACTION
				if action.model_dump(exclude_unset=True).get('done') is not None:
					msg = f'Done action is allowed only as a single action - stopped after action {i} / {total_actions}.'
					self.logger.debug(msg)
					break

			# wait between actions (only after first action)
			if i > 0:
				self.logger.debug(f'Waiting {self.browser_profile.wait_between_actions} seconds between actions')
				await asyncio.sleep(self.browser_profile.wait_between_actions)

			try:
				await self._check_stop_or_pause()
				# Get action name from the action model
				action_data = action.model_dump(exclude_unset=True)
				action_name = next(iter(action_data.keys())) if action_data else 'unknown'

				# Проверка на капчу и форму входа перед выполнением действий
				if action_name in ['click', 'navigate', 'input'] and self.browser_session is not None:
					browser_state = self.browser_session._cached_browser_state_summary
					if browser_state:
						url = browser_state.url or ''
						title = browser_state.title or ''
						url_lower = url.lower()
						title_lower = title.lower()
						
						# Проверяем URL и заголовок на наличие капчи
						is_captcha_page = (
							'captcha' in url_lower or 'showcaptcha' in url_lower or
							'робот' in title_lower or 'robot' in title_lower
						)
						
						# Проверяем наличие формы входа (поля для логина/пароля)
						is_login_form = False
						has_password_field = False
						has_login_field = False
						has_submit_button = False
						
						if browser_state.dom_state and browser_state.dom_state.selector_map:
							selector_map = browser_state.dom_state.selector_map
							for element in selector_map.values():
								element_type = getattr(element, 'type', '') or ''
								element_role = getattr(element, 'role', '') or ''
								element_text = getattr(element, 'text', '') or ''
								element_placeholder = getattr(element, 'placeholder', '') or ''
								
								text_lower = element_text.lower()
								placeholder_lower = element_placeholder.lower()
								type_lower = element_type.lower()
								role_lower = element_role.lower()
								
								# Проверяем на поле пароля
								if (
									type_lower == 'password' or
									'password' in text_lower or
									'password' in placeholder_lower or
									'пароль' in text_lower or
									'пароль' in placeholder_lower
								):
									has_password_field = True
								
								# Проверяем на поле логина/email (но не пароль!)
								if type_lower != 'password':  # Исключаем поле пароля
									if (
										(type_lower in ['email', 'text'] and role_lower == 'textbox') or
										'login' in text_lower or 'логин' in text_lower or
										'email' in text_lower or 'почта' in text_lower or
										'username' in text_lower or 'имя пользователя' in text_lower or
										'login' in placeholder_lower or 'логин' in placeholder_lower or
										'email' in placeholder_lower or 'почта' in placeholder_lower or
										'телефон' in text_lower or 'phone' in text_lower
									):
										has_login_field = True
								
								# Проверяем на кнопку отправки формы
								if (
									role_lower == 'button' and (
										'войти' in text_lower or 'login' in text_lower or
										'войти' in placeholder_lower or 'login' in placeholder_lower or
										'отправить' in text_lower or 'submit' in text_lower or
										'вход' in text_lower or 'sign in' in text_lower
									)
								):
									has_submit_button = True
						
						# Форма входа определяется наличием поля пароля И (поля логина ИЛИ кнопки отправки)
						is_login_form = has_password_field and (has_login_field or has_submit_button)
						
						# Проверяем текст элементов на наличие капчи и деструктивных действий
						is_captcha_element = False
						is_destructive_action = False
						destructive_action_type = None
						if action_name == 'click' and 'index' in action_data.get('click', {}):
							click_params = action_data.get('click', {})
							index = click_params.get('index')
							if index is not None and browser_state.dom_state:
								selector_map = browser_state.dom_state.selector_map
								# Ищем элемент по index (который является backend_node_id в selector_map)
								clicked_element = selector_map.get(index) if index in selector_map else None
								if clicked_element:
									# Получаем текст элемента разными способами
									element_text = ''
									if hasattr(clicked_element, 'ax_node') and clicked_element.ax_node and clicked_element.ax_node.name:
										element_text = clicked_element.ax_node.name
									elif hasattr(clicked_element, 'get_all_children_text'):
										element_text = clicked_element.get_all_children_text() or ''
									elif hasattr(clicked_element, 'get_meaningful_text_for_llm'):
										element_text = clicked_element.get_meaningful_text_for_llm() or ''
									elif hasattr(clicked_element, 'text'):
										element_text = getattr(clicked_element, 'text', '') or ''
									
									text_lower = element_text.lower()
									
									# Проверка на капчу
									is_captcha_element = (
										'робот' in text_lower or 'robot' in text_lower or
										'не робот' in text_lower or 'not a robot' in text_lower
									)
									
									# Проверка на деструктивные действия (оплата, удаление)
									# ВАЖНО: проверяем только если элемент найден и текст получен
									if not is_captcha_element and element_text:
										# Ключевые слова для оплаты/подтверждения заказа (только финальные действия оплаты)
										payment_keywords = [
											'оплат', 'pay now', 'checkout', 'place order', 'оформить заказ',
											'подтвердить заказ', 'оплатить заказ', 'купить сейчас', 'buy now',
											'подтвердить и оплатить', 'confirm and pay', 'proceed to payment'
										]
										# Ключевые слова для удаления
										delete_keywords = [
											'удалить письмо', 'delete email', 'удалить навсегда',
											'delete permanently', 'удалить безвозвратно'
										]
										
										is_payment_action = any(kw in text_lower for kw in payment_keywords)
										is_delete_action = any(kw in text_lower for kw in delete_keywords)
										
										if is_payment_action:
											is_destructive_action = True
											destructive_action_type = 'payment'
										elif is_delete_action:
											is_destructive_action = True
											destructive_action_type = 'delete'
						
						# Если обнаружена форма входа, блокируем действия и заменяем на wait_for_user_input
						if is_login_form:
							# Проверяем историю агента, чтобы не запрашивать вход повторно, если уже был вход
							already_waited_for_login = False
							if hasattr(self, 'history') and self.history and hasattr(self.history, 'history') and self.history.history:
								# Получаем URL предыдущего шага для сравнения
								previous_url = None
								for history_item in reversed(self.history.history[-5:]):  # Проверяем последние 5 шагов
									if history_item.state and history_item.state.url:
										previous_url = history_item.state.url
										break
								
								# Проверяем, вызывали ли мы уже wait_for_user_input или request_user_input
								for history_item in reversed(self.history.history[-5:]):  # Проверяем последние 5 шагов
									if history_item.model_output and history_item.model_output.action:
										for act in history_item.model_output.action:
											act_data = act.model_dump(exclude_unset=True)
											if 'wait_for_user_input' in act_data or 'request_user_input' in act_data:
												# Если URL изменился после ожидания, значит вход выполнен
												if previous_url and browser_state.url != previous_url:
													already_waited_for_login = True
													break
									if already_waited_for_login:
										break
							
							if not already_waited_for_login:
								self.logger.warning(
									f'⚠️ Обнаружена форма входа - блокирую действие {action_name} и запрашиваю wait_for_user_input'
								)
								# Заменяем действие на wait_for_user_input
								from agent.tools.views import WaitForUserInputAction
								from agent.tools.registry.views import ActionModel
								from pydantic import create_model, Field
								
								# Создаем новое действие wait_for_user_input
								WaitForUserInputActionModel = create_model(
									'WaitForUserInputActionModel',
									__base__=ActionModel,
									wait_for_user_input=(WaitForUserInputAction, Field(...))
								)
								
								login_action = WaitForUserInputActionModel(
									wait_for_user_input=WaitForUserInputAction(
										message='Пожалуйста, заполните форму входа в браузере (логин, пароль и т.д.)'
									)
								)
								action = login_action
								action_name = 'wait_for_user_input'
								action_data = {'wait_for_user_input': {'message': 'Пожалуйста, заполните форму входа в браузере (логин, пароль и т.д.)'}}
						
						# Если обнаружена капча, блокируем действия и заменяем на request_user_input
						elif is_captcha_page or is_captcha_element:
							self.logger.warning(
								f'⚠️ Блокирую действие {action_name} на странице с капчей - агент должен использовать request_user_input'
							)
							# Заменяем действие на request_user_input
							from agent.tools.views import RequestUserInputAction
							from agent.tools.registry.views import ActionModel
							from pydantic import create_model
							
							# Создаем новое действие request_user_input
							from pydantic import Field
							RequestUserInputActionModel = create_model(
								'RequestUserInputActionModel',
								__base__=ActionModel,
								request_user_input=(RequestUserInputAction, Field(...))
							)
							
							captcha_action = RequestUserInputActionModel(
								request_user_input=RequestUserInputAction(
									prompt='Пожалуйста, решите капчу в браузере и введите "готово" (или "done") когда закончите'
								)
							)
							action = captcha_action
							action_name = 'request_user_input'
							action_data = {'request_user_input': {'prompt': 'Пожалуйста, решите капчу в браузере и введите "готово" (или "done") когда закончите'}}
						
						# Security layer: проверка на деструктивные действия (оплата, удаление)
						elif is_destructive_action:
							action_description = 'оплату/подтверждение заказа' if destructive_action_type == 'payment' else 'удаление'
							self.logger.warning(
								f'🛡️ Security layer: блокирую деструктивное действие {action_name} ({action_description}) - запрашиваю подтверждение пользователя'
							)
							# Заменяем действие на request_user_input с запросом подтверждения
							from agent.tools.views import RequestUserInputAction
							from agent.tools.registry.views import ActionModel
							from pydantic import create_model, Field
							
							RequestUserInputActionModel = create_model(
								'RequestUserInputActionModel',
								__base__=ActionModel,
								request_user_input=(RequestUserInputAction, Field(...))
							)
							
							if destructive_action_type == 'payment':
								prompt_text = 'Обнаружена кнопка оплаты/подтверждения заказа. Вы хотите оплатить/подтвердить заказ? Ответьте только "да"/"yes" для подтверждения или "нет"/"no" для отмены.'
							else:  # delete
								prompt_text = 'Обнаружена кнопка удаления. Вы хотите удалить этот элемент? Ответьте только "да"/"yes" для подтверждения или "нет"/"no" для отмены.'
							
							destructive_action = RequestUserInputActionModel(
								request_user_input=RequestUserInputAction(
									prompt=prompt_text
								)
							)
							action = destructive_action
							action_name = 'request_user_input'
							action_data = {'request_user_input': {'prompt': prompt_text}}
						
						# Если достигнут лимит неудачных попыток клика в модальном окне, блокируем действие и запрашиваем помощь
						elif self.state.modal_click_failures >= 3 and action_name == 'click':
							# Проверяем, что модальное окно действительно открыто
							if browser_state and self.email_subagent.detect_dialog(browser_state):
								self.logger.warning(
									f'🛑 Блокирую действие {action_name} - достигнут лимит неудачных попыток клика в модальном окне (3). '
									'Запрашиваю помощь пользователя.'
								)
								# Заменяем действие на request_user_input
								from agent.tools.views import RequestUserInputAction
								from agent.tools.registry.views import ActionModel
								from pydantic import create_model, Field
								
								RequestUserInputActionModel = create_model(
									'RequestUserInputActionModel',
									__base__=ActionModel,
									request_user_input=(RequestUserInputAction, Field(...))
								)
								
								modal_action = RequestUserInputActionModel(
									request_user_input=RequestUserInputAction(
										prompt='Не удалось найти кнопку отправки в модальном окне после 3 попыток. Пожалуйста, нажмите кнопку отправки формы в модальном окне вручную, затем введите "готово" (или "done") когда форма будет отправлена.'
									)
								)
								action = modal_action
								action_name = 'request_user_input'
								action_data = {'request_user_input': {'prompt': 'Не удалось найти кнопку отправки в модальном окне после 3 попыток. Пожалуйста, нажмите кнопку отправки формы в модальном окне вручную, затем введите "готово" (или "done") когда форма будет отправлена.'}}
								# Сбрасываем счетчик после запроса помощи
								self.state.modal_click_failures = 0

				# Используем субагента для почты для обработки почтовых специфичных ситуаций
				if self.browser_session is not None:
					browser_state = self.browser_session._cached_browser_state_summary
					if browser_state:
						# Информируем агента о наличии диалога, но НЕ закрываем автоматически
						# Агент сам решает - работать с диалогом (например, форма отклика) или закрыть его
						if self.email_subagent.detect_dialog(browser_state):
							self.logger.info('ℹ️ Обнаружен открытый диалог - агент должен решить: работать с ним или закрыть через Escape')
						
						# Логируем метаданные письма только для почтовых клиентов
						if self.email_subagent.is_email_client(browser_state):
							email_metadata = self.email_subagent.extract_email_metadata(browser_state)
							
							# Логируем метаданные письма перед действиями
							if email_metadata['is_opened'] and action_name == 'click':
								click_params = action_data.get('click', {})
								index = click_params.get('index')
								if index is not None and browser_state.dom_state:
									selector_map = browser_state.dom_state.selector_map
									clicked_element = selector_map.get(index)
									if clicked_element:
										# Логируем метаданные письма (без хардкода - агент сам решает что делать)
										self.logger.info(f'📧 Действие на странице почтового клиента:')
										if email_metadata['subject']:
											self.logger.info(f'   Тема письма: {email_metadata["subject"]}')
										if email_metadata['sender']:
											self.logger.info(f'   Отправитель: {email_metadata["sender"]}')
										if email_metadata['body_preview']:
											body_preview = email_metadata['body_preview'][:300] + '...' if len(email_metadata['body_preview']) > 300 else email_metadata['body_preview']
											self.logger.info(f'   Содержание (первые 300 символов): {body_preview}')

				# Log action before execution
				await self._log_action(action, action_name, i + 1, total_actions)

				time_start = time.time()

				result = await self.tools.act(
					action=action,
					browser_session=self.browser_session,
					file_system=self.file_system,
					page_extraction_llm=self.settings.page_extraction_llm,
					sensitive_data=self.sensitive_data,
					available_file_paths=self.available_file_paths,
				)

				time_end = time.time()
				time_elapsed = time_end - time_start

				# После действий, которые могут изменить DOM (особенно в SPA), ждем обновления страницы
				# Это критично для SPA-приложений (например, почтовые клиенты), где клик может менять содержимое без изменения URL
				if action_name in ['click', 'navigate']:
					# Увеличиваем задержку для SPA приложений, чтобы дать время DOM обновиться
					# SPA страницы требуют времени для загрузки контента
					wait_time = 2.0  # Одинаковое время для click и navigate
					self.logger.info(f'⏳ Ожидание {wait_time}s после {action_name} для обновления DOM (SPA)')
					await asyncio.sleep(wait_time)
					
					# Инвалидируем кэш DOM watchdog, чтобы принудительно перестроить DOM при следующем запросе
					# Это гарантирует, что следующее получение browser_state будет свежим
					if self.browser_session and self.browser_session._dom_watchdog:
						self.browser_session._dom_watchdog.clear_cache()
						self.logger.info(f'🔄 Кэш DOM очищен после {action_name} - следующее получение browser_state будет свежим')
					
					# Проверяем, осталось ли модальное окно открытым после клика (для отслеживания неудачных попыток)
					if action_name == 'click' and self.browser_session:
						# Получаем свежий browser_state после очистки кэша
						await asyncio.sleep(0.5)  # Небольшая задержка для обновления DOM
						fresh_browser_state = self.browser_session._cached_browser_state_summary
						if fresh_browser_state and self.email_subagent.detect_dialog(fresh_browser_state):
							# Модальное окно все еще открыто после клика - увеличиваем счетчик
							self.state.modal_click_failures += 1
							self.logger.warning(
								f'⚠️ Модальное окно все еще открыто после клика. Счетчик неудачных попыток: {self.state.modal_click_failures}/3'
							)
							if self.state.modal_click_failures >= 3:
								self.logger.warning(
									'🛑 Достигнут лимит неудачных попыток клика в модальном окне (3). '
									'В следующем шаге будет запрошена помощь пользователя.'
								)
						else:
							# Модальное окно закрыто - сбрасываем счетчик
							if self.state.modal_click_failures > 0:
								self.logger.info(f'✅ Модальное окно закрыто. Счетчик неудачных попыток сброшен с {self.state.modal_click_failures} до 0')
								self.state.modal_click_failures = 0
					
					# После request_user_input проверяем, закрыто ли модальное окно (пользователь мог нажать кнопку)
					if action_name == 'request_user_input' and self.browser_session:
						await asyncio.sleep(0.5)  # Небольшая задержка для обновления DOM
						fresh_browser_state = self.browser_session._cached_browser_state_summary
						if fresh_browser_state:
							if not self.email_subagent.detect_dialog(fresh_browser_state):
								# Модальное окно закрыто после request_user_input - сбрасываем счетчик
								if self.state.modal_click_failures > 0:
									self.logger.info(f'✅ Модальное окно закрыто после request_user_input. Счетчик неудачных попыток сброшен с {self.state.modal_click_failures} до 0')
									self.state.modal_click_failures = 0
								
								# КРИТИЧЕСКИ ВАЖНО: Если модальное окно закрыто после request_user_input,
								# это означает, что пользователь выполнил действие (например, нажал кнопку отправки).
								# В этом случае задача уже выполнена - модифицируем result, чтобы агент завершил задачу.
								# Проверяем, что пользователь ответил "done" или "готово"
								if result.extracted_content and ('подтвердил' in result.extracted_content.lower() or 'выполнено' in result.extracted_content.lower()):
									self.logger.info('✅ Модальное окно закрыто после request_user_input с подтверждением пользователя. Задача выполнена пользователем - завершаем выполнение.')
									# Модифицируем result, чтобы агент понял, что задача выполнена
									result.is_done = True
									result.success = True
									result.long_term_memory = 'Пользователь успешно выполнил действие (например, нажал кнопку отправки отклика). Задача выполнена.'
									result.extracted_content = 'Задача выполнена пользователем. Модальное окно закрыто, действие успешно завершено.'
									# Прерываем выполнение дальнейших действий в этом шаге
									break

				if result.error:
					await self._demo_mode_log(
						f'Action "{action_name}" failed: {result.error}',
						'error',
						{'action': action_name, 'step': self.state.n_steps},
					)
				elif result.is_done:
					completion_text = result.long_term_memory or result.extracted_content or 'Task marked as done.'
					level = 'success' if result.success is not False else 'warning'
					await self._demo_mode_log(
						completion_text,
						level,
						{'action': action_name, 'step': self.state.n_steps},
					)

				results.append(result)

				if results[-1].is_done or results[-1].error or i == total_actions - 1:
					break

			except Exception as e:
				# Handle any exceptions during action execution
				self.logger.error(f'❌ Executing action {i + 1} failed -> {type(e).__name__}: {e}')
				await self._demo_mode_log(
					f'Action "{action_name}" raised {type(e).__name__}: {e}',
					'error',
					{'action': action_name, 'step': self.state.n_steps},
				)
				raise e

		return results

	async def _log_action(self, action, action_name: str, action_num: int, total_actions: int) -> None:
		"""Log the action before execution with colored formatting"""
		# Color definitions
		blue = '\033[34m'  # Action name
		magenta = '\033[35m'  # Parameter names
		reset = '\033[0m'

		# Format action number and name
		if total_actions > 1:
			action_header = f'▶️  [{action_num}/{total_actions}] {blue}{action_name}{reset}:'
			plain_header = f'▶️  [{action_num}/{total_actions}] {action_name}:'
		else:
			action_header = f'▶️   {blue}{action_name}{reset}:'
			plain_header = f'▶️  {action_name}:'

		# Get action parameters
		action_data = action.model_dump(exclude_unset=True)
		params = action_data.get(action_name, {})

		# Build parameter parts with colored formatting
		param_parts = []
		plain_param_parts = []

		if params and isinstance(params, dict):
			for param_name, value in params.items():
				# Truncate long values for readability
				if isinstance(value, str) and len(value) > 150:
					display_value = value[:150] + '...'
				elif isinstance(value, list) and len(str(value)) > 200:
					display_value = str(value)[:200] + '...'
				else:
					display_value = value

				param_parts.append(f'{magenta}{param_name}{reset}: {display_value}')
				plain_param_parts.append(f'{param_name}: {display_value}')

		# Join all parts
		if param_parts:
			params_string = ', '.join(param_parts)
			self.logger.info(f'  {action_header} {params_string}')
		else:
			self.logger.info(f'  {action_header}')

		if self._demo_mode_enabled:
			panel_message = plain_header
			if plain_param_parts:
				panel_message = f'{panel_message} {", ".join(plain_param_parts)}'
			await self._demo_mode_log(panel_message.strip(), 'action', {'action': action_name, 'step': self.state.n_steps})

	async def log_completion(self) -> None:
		"""Log the completion of the task"""
		# self._task_end_time = time.time()
		if self.history.is_successful():
			self.logger.info('✅ Task completed successfully')
			await self._demo_mode_log('Task completed successfully', 'success', {'tag': 'task'})

	async def _generate_rerun_summary(
		self, original_task: str, results: list[ActionResult], summary_llm: BaseChatModel | None = None
	) -> ActionResult:
		"""Generate AI summary of rerun completion using screenshot and last step info"""
		from agent.agent.views import RerunSummaryAction

		# Get current screenshot
		screenshot_b64 = None
		try:
			screenshot = await self.browser_session.take_screenshot(full_page=False)
			if screenshot:
				import base64

				screenshot_b64 = base64.b64encode(screenshot).decode('utf-8')
		except Exception as e:
			self.logger.warning(f'Failed to capture screenshot for rerun summary: {e}')

		# Build summary prompt and message
		error_count = sum(1 for r in results if r.error)
		success_count = len(results) - error_count

		from agent.agent.prompts import get_rerun_summary_message, get_rerun_summary_prompt

		prompt = get_rerun_summary_prompt(
			original_task=original_task,
			total_steps=len(results),
			success_count=success_count,
			error_count=error_count,
		)

		# Use provided LLM, agent's LLM, or fall back to OpenAI with structured output
		try:
			# Determine which LLM to use
			if summary_llm is None:
				# Try to use the agent's LLM first
				summary_llm = self.llm
				self.logger.debug('Using agent LLM for rerun summary')
			else:
				self.logger.debug(f'Using provided LLM for rerun summary: {summary_llm.model}')

			# Build message with prompt and optional screenshot
			from agent.llm.messages import BaseMessage

			message = get_rerun_summary_message(prompt, screenshot_b64)
			messages: list[BaseMessage] = [message]  # type: ignore[list-item]

			# Try calling with structured output first
			self.logger.debug(f'Calling LLM for rerun summary with {len(messages)} message(s)')
			try:
				kwargs: dict = {'output_format': RerunSummaryAction}
				response = await summary_llm.ainvoke(messages, **kwargs)
				summary: RerunSummaryAction = response.completion  # type: ignore[assignment]
				self.logger.debug(f'LLM response type: {type(summary)}')
				self.logger.debug(f'LLM response: {summary}')
			except Exception as structured_error:
				# Если структурированный вывод не поддерживается LLM для этого типа,
				# fall back to text response without parsing
				self.logger.debug(f'Structured output failed: {structured_error}, falling back to text response')

				response = await summary_llm.ainvoke(messages, None)
				response_text = response.completion
				self.logger.debug(f'LLM text response: {response_text}')

				# Use the text response directly as the summary
				summary = RerunSummaryAction(
					summary=response_text if isinstance(response_text, str) else str(response_text),
					success=error_count == 0,
					completion_status='complete' if error_count == 0 else ('partial' if success_count > 0 else 'failed'),
				)

			self.logger.info(f'📊 Rerun Summary: {summary.summary}')
			self.logger.info(f'📊 Status: {summary.completion_status} (success={summary.success})')

			return ActionResult(
				is_done=True,
				success=summary.success,
				extracted_content=summary.summary,
				long_term_memory=f'Rerun completed with status: {summary.completion_status}. {summary.summary[:100]}',
			)

		except Exception as e:
			self.logger.warning(f'Failed to generate AI summary: {e.__class__.__name__}: {e}')
			self.logger.debug('Full error traceback:', exc_info=True)
			# Fallback to simple summary
			return ActionResult(
				is_done=True,
				success=error_count == 0,
				extracted_content=f'Rerun completed: {success_count}/{len(results)} steps succeeded',
				long_term_memory=f'Rerun completed: {success_count} steps succeeded, {error_count} errors',
			)

	async def _execute_ai_step(
		self,
		query: str,
		include_screenshot: bool = False,
		extract_links: bool = False,
		ai_step_llm: BaseChatModel | None = None,
	) -> ActionResult:
		"""
		Execute an AI step during rerun to re-evaluate extract actions.
		Analyzes full page DOM/markdown + optional screenshot.

		Args:
			query: What to analyze or extract from the current page
			include_screenshot: Whether to include screenshot in analysis
			extract_links: Whether to include links in markdown extraction
			ai_step_llm: Optional LLM to use. If not provided, uses agent's LLM

		Returns:
			ActionResult with extracted content
		"""
		from agent.agent.prompts import get_ai_step_system_prompt, get_ai_step_user_prompt, get_rerun_summary_message
		from agent.llm.messages import SystemMessage, UserMessage
		from agent.utils import sanitize_surrogates

		# Use provided LLM or agent's LLM
		llm = ai_step_llm or self.llm
		self.logger.debug(f'Using LLM for AI step: {llm.model}')

		# Extract clean markdown
		try:
			from agent.dom.markdown_extractor import extract_clean_markdown

			content, content_stats = await extract_clean_markdown(
				browser_session=self.browser_session, extract_links=extract_links
			)
		except Exception as e:
			return ActionResult(error=f'Could not extract clean markdown: {type(e).__name__}: {e}')

		# Get screenshot if requested
		screenshot_b64 = None
		if include_screenshot:
			try:
				screenshot = await self.browser_session.take_screenshot(full_page=False)
				if screenshot:
					import base64

					screenshot_b64 = base64.b64encode(screenshot).decode('utf-8')
			except Exception as e:
				self.logger.warning(f'Failed to capture screenshot for ai_step: {e}')

		# Build prompt with content stats
		original_html_length = content_stats['original_html_chars']
		initial_markdown_length = content_stats['initial_markdown_chars']
		final_filtered_length = content_stats['final_filtered_chars']
		chars_filtered = content_stats['filtered_chars_removed']

		stats_summary = f"""Content processed: {original_html_length:,} HTML chars → {initial_markdown_length:,} initial markdown → {final_filtered_length:,} filtered markdown"""
		if chars_filtered > 0:
			stats_summary += f' (filtered {chars_filtered:,} chars of noise)'

		# Sanitize content
		content = sanitize_surrogates(content)
		query = sanitize_surrogates(query)

		# Get prompts from prompts.py
		system_prompt = get_ai_step_system_prompt()
		prompt_text = get_ai_step_user_prompt(query, stats_summary, content)

		# Build user message with optional screenshot
		if screenshot_b64:
			user_message = get_rerun_summary_message(prompt_text, screenshot_b64)
		else:
			user_message = UserMessage(content=prompt_text)

		try:
			import asyncio

			response = await asyncio.wait_for(llm.ainvoke([SystemMessage(content=system_prompt), user_message]), timeout=120.0)

			current_url = await self.browser_session.get_current_page_url()
			extracted_content = (
				f'<url>\n{current_url}\n</url>\n<query>\n{query}\n</query>\n<result>\n{response.completion}\n</result>'
			)

			# Simple memory handling
			MAX_MEMORY_LENGTH = 1000
			if len(extracted_content) < MAX_MEMORY_LENGTH:
				memory = extracted_content
				include_extracted_content_only_once = False
			else:
				file_name = await self.file_system.save_extracted_content(extracted_content)
				memory = f'Query: {query}\nContent in {file_name} and once in <read_state>.'
				include_extracted_content_only_once = True

			self.logger.info(f'🤖 AI Step: {memory}')
			return ActionResult(
				extracted_content=extracted_content,
				include_extracted_content_only_once=include_extracted_content_only_once,
				long_term_memory=memory,
			)
		except Exception as e:
			self.logger.warning(f'Failed to execute AI step: {e.__class__.__name__}: {e}')
			self.logger.debug('Full error traceback:', exc_info=True)
			return ActionResult(error=f'AI step failed: {e}')

	async def rerun_history(
		self,
		history: AgentHistoryList,
		max_retries: int = 3,
		skip_failures: bool = True,
		delay_between_actions: float = 2.0,
		summary_llm: BaseChatModel | None = None,
		ai_step_llm: BaseChatModel | None = None,
	) -> list[ActionResult]:
		"""
		Rerun a saved history of actions with error handling and retry logic.

		Args:
		                history: The history to replay
		                max_retries: Maximum number of retries per action
		                skip_failures: Whether to skip failed actions or stop execution
		                delay_between_actions: Delay between actions in seconds
		                summary_llm: Optional LLM to use for generating the final summary. If not provided, uses the agent's LLM
		                ai_step_llm: Optional LLM to use for AI steps (extract actions). If not provided, uses the agent's LLM

		Returns:
		                List of action results (including AI summary as the final result)
		"""
		# Skip cloud sync session events for rerunning (we're replaying, not starting new)
		self.state.session_initialized = True

		# Initialize browser session
		await self.browser_session.start()

		results = []

		for i, history_item in enumerate(history.history):
			goal = history_item.model_output.current_state.next_goal if history_item.model_output else ''
			step_num = history_item.metadata.step_number if history_item.metadata else i
			step_name = 'Initial actions' if step_num == 0 else f'Step {step_num}'

			# Determine step delay
			if history_item.metadata and history_item.metadata.step_interval is not None:
				step_delay = history_item.metadata.step_interval
				# Format delay nicely - show ms for values < 1s, otherwise show seconds
				if step_delay < 1.0:
					delay_str = f'{step_delay * 1000:.0f}ms'
				else:
					delay_str = f'{step_delay:.1f}s'
				delay_source = f'using saved step_interval={delay_str}'
			else:
				step_delay = delay_between_actions
				if step_delay < 1.0:
					delay_str = f'{step_delay * 1000:.0f}ms'
				else:
					delay_str = f'{step_delay:.1f}s'
				delay_source = f'using default delay={delay_str}'

			self.logger.info(f'Replaying {step_name} ({i + 1}/{len(history.history)}) [{delay_source}]: {goal}')

			if (
				not history_item.model_output
				or not history_item.model_output.action
				or history_item.model_output.action == [None]
			):
				self.logger.warning(f'{step_name}: No action to replay, skipping')
				results.append(ActionResult(error='No action to replay'))
				continue

			retry_count = 0
			while retry_count < max_retries:
				try:
					result = await self._execute_history_step(history_item, step_delay, ai_step_llm)
					results.extend(result)
					break

				except Exception as e:
					retry_count += 1
					if retry_count == max_retries:
						error_msg = f'{step_name} failed after {max_retries} attempts: {str(e)}'
						self.logger.error(error_msg)
						if not skip_failures:
							results.append(ActionResult(error=error_msg))
							raise RuntimeError(error_msg)
					else:
						self.logger.warning(f'{step_name} failed (attempt {retry_count}/{max_retries}), retrying...')
						await asyncio.sleep(delay_between_actions)

		# Generate AI summary of rerun completion
		self.logger.info('🤖 Generating AI summary of rerun completion...')
		summary_result = await self._generate_rerun_summary(self.task, results, summary_llm)
		results.append(summary_result)

		await self.close()
		return results

	async def _execute_initial_actions(self) -> None:
		# Execute initial actions if provided
		if self.initial_actions and not self.state.follow_up_task:
			self.logger.debug(f'⚡ Executing {len(self.initial_actions)} initial actions...')
			result = await self.multi_act(self.initial_actions)
			# update result 1 to mention that its was automatically loaded
			if result and self.initial_url and result[0].long_term_memory:
				result[0].long_term_memory = f'Found initial url and automatically loaded it. {result[0].long_term_memory}'
			self.state.last_result = result

			# Save initial actions to history as step 0 for rerun capability
			# Skip browser state capture for initial actions (usually just URL navigation)
			if self.settings.flash_mode:
				model_output = self.AgentOutput(
					evaluation_previous_goal=None,
					memory='Initial navigation',
					next_goal=None,
					action=self.initial_actions,
				)
			else:
				model_output = self.AgentOutput(
					evaluation_previous_goal='Start',
					memory=None,
					next_goal='Initial navigation',
					action=self.initial_actions,
				)

			metadata = StepMetadata(step_number=0, step_start_time=time.time(), step_end_time=time.time(), step_interval=None)

			# Create minimal browser state history for initial actions
			state_history = BrowserStateHistory(
				url=self.initial_url or '',
				title='Initial Actions',
				tabs=[],
				interacted_element=[None] * len(self.initial_actions),  # No DOM elements needed
				screenshot_path=None,
			)

			history_item = AgentHistory(
				model_output=model_output,
				result=result,
				state=state_history,
				metadata=metadata,
			)

			self.history.add_item(history_item)
			self.logger.debug('📝 Saved initial actions to history as step 0')
			self.logger.debug('Initial actions completed')

	async def _execute_history_step(
		self, history_item: AgentHistory, delay: float, ai_step_llm: BaseChatModel | None = None
	) -> list[ActionResult]:
		"""Execute a single step from history with element validation.

		For extract actions, uses AI to re-evaluate the content since page content may have changed.
		"""
		assert self.browser_session is not None, 'BrowserSession is not set up'

		await asyncio.sleep(delay)
		state = await self.browser_session.get_browser_state_summary(include_screenshot=False)
		if not state or not history_item.model_output:
			raise ValueError('Invalid state or model output')

		results = []
		pending_actions = []

		for i, action in enumerate(history_item.model_output.action):
			# Check if this is an extract action - use AI step instead
			action_data = action.model_dump(exclude_unset=True)
			action_name = next(iter(action_data.keys()), None)

			if action_name == 'extract':
				# Execute any pending actions first to maintain correct order
				# (e.g., if step is [click, extract], click must happen before extract)
				if pending_actions:
					batch_results = await self.multi_act(pending_actions)
					results.extend(batch_results)
					pending_actions = []

				# Now execute AI step for extract action
				extract_params = action_data['extract']
				query = extract_params.get('query', '')
				extract_links = extract_params.get('extract_links', False)

				self.logger.info(f'🤖 Using AI step for extract action: {query[:50]}...')
				ai_result = await self._execute_ai_step(
					query=query,
					include_screenshot=False,  # Match original extract behavior
					extract_links=extract_links,
					ai_step_llm=ai_step_llm,
				)
				results.append(ai_result)
			else:
				# For non-extract actions, update indices and collect for batch execution
				updated_action = await self._update_action_indices(
					history_item.state.interacted_element[i],
					action,
					state,
				)
				if updated_action is None:
					raise ValueError(f'Could not find matching element {i} in current page')
				pending_actions.append(updated_action)

		# Execute any remaining pending actions
		if pending_actions:
			batch_results = await self.multi_act(pending_actions)
			results.extend(batch_results)

		return results

	async def _update_action_indices(
		self,
		historical_element: DOMInteractedElement | None,
		action: ActionModel,  # Type this properly based on your action model
		browser_state_summary: BrowserStateSummary,
	) -> ActionModel | None:
		"""
		Update action indices based on current page state.
		Returns updated action or None if element cannot be found.
		"""
		if not historical_element or not browser_state_summary.dom_state.selector_map:
			return action

		# selector_hash_map = {hash(e): e for e in browser_state_summary.dom_state.selector_map.values()}

		highlight_index, current_element = next(
			(
				(highlight_index, element)
				for highlight_index, element in browser_state_summary.dom_state.selector_map.items()
				if element.element_hash == historical_element.element_hash
			),
			(None, None),
		)

		if not current_element or highlight_index is None:
			return None

		old_index = action.get_index()
		if old_index != highlight_index:
			action.set_index(highlight_index)
			self.logger.info(f'Element moved in DOM, updated index from {old_index} to {highlight_index}')

		return action

	async def load_and_rerun(
		self,
		history_file: str | Path | None = None,
		variables: dict[str, str] | None = None,
		**kwargs,
	) -> list[ActionResult]:
		"""
		Load history from file and rerun it, optionally substituting variables.

		Args:
			history_file: Path to the history file
			variables: Optional dict mapping variable names to new values (e.g. {'email': 'new@example.com'})
			**kwargs: Additional arguments passed to rerun_history
		"""
		if not history_file:
			history_file = 'AgentHistory.json'
		history = AgentHistoryList.load_from_file(history_file, self.AgentOutput)

		# Substitute variables if provided
		if variables:
			history = self._substitute_variables_in_history(history, variables)

		return await self.rerun_history(history, **kwargs)

	def save_history(self, file_path: str | Path | None = None) -> None:
		"""Save the history to a file with sensitive data filtering"""
		if not file_path:
			file_path = 'AgentHistory.json'
		self.history.save_to_file(file_path, sensitive_data=self.sensitive_data)

	def pause(self) -> None:
		"""Pause the agent before the next step"""
		print('\n\n⏸️ Paused the agent and left the browser open.\n\tPress [Enter] to resume or [Ctrl+C] again to quit.')
		self.state.paused = True
		self._external_pause_event.clear()

	def resume(self) -> None:
		"""Resume the agent"""
		print('----------------------------------------------------------------------')
		print('▶️  Resuming agent execution where it left off...\n')
		self.state.paused = False
		self._external_pause_event.set()

	def stop(self) -> None:
		"""Stop the agent"""
		self.logger.info('⏹️ Agent stopping')
		self.state.stopped = True

		# Signal pause event to unblock any waiting code so it can check the stopped state
		self._external_pause_event.set()

		# Task stopped

	def _convert_initial_actions(self, actions: list[dict[str, dict[str, Any]]]) -> list[ActionModel]:
		"""Convert dictionary-based actions to ActionModel instances"""
		converted_actions = []
		action_model = self.ActionModel
		for action_dict in actions:
			# Each action_dict should have a single key-value pair
			action_name = next(iter(action_dict))
			params = action_dict[action_name]

			# Get the parameter model for this action from registry
			action_info = self.tools.registry.registry.actions[action_name]
			param_model = action_info.param_model

			# Create validated parameters using the appropriate param model
			validated_params = param_model(**params)

			# Create ActionModel instance with the validated parameters
			action_model = self.ActionModel(**{action_name: validated_params})
			converted_actions.append(action_model)

		return converted_actions

	def _verify_and_setup_llm(self):
		"""
		Verify that the LLM API keys are setup and the LLM API is responding properly.
		Also handles tool calling method detection if in auto mode.
		"""

		# Skip verification if already done
		if getattr(self.llm, '_verified_api_keys', None) is True or CONFIG.SKIP_LLM_API_KEY_VERIFICATION:
			setattr(self.llm, '_verified_api_keys', True)
			return True

	@property
	def message_manager(self) -> MessageManager:
		return self._message_manager

	async def close(self):
		"""Close all resources"""
		try:
			# Only close browser if keep_alive is False (or not set)
			if self.browser_session is not None:
				if not self.browser_session.browser_profile.keep_alive:
					# Kill the browser session - this dispatches BrowserStopEvent,
					# stops the EventBus with clear=True, and recreates a fresh EventBus
					await self.browser_session.kill()

			# Skills service removed in simplified version

			# Force garbage collection
			gc.collect()

			# Логирование оставшихся потоков и asyncio задач
			import threading

			threads = threading.enumerate()
			self.logger.debug(f'🧵 Remaining threads ({len(threads)}): {[t.name for t in threads]}')

			# Get all asyncio tasks
			tasks = asyncio.all_tasks(asyncio.get_event_loop())
			# Filter out the current task (this close() coroutine)
			other_tasks = [t for t in tasks if t != asyncio.current_task()]
			if other_tasks:
				self.logger.debug(f'⚡ Remaining asyncio tasks ({len(other_tasks)}):')
				for task in other_tasks[:10]:  # Limit to first 10 to avoid spam
					self.logger.debug(f'  - {task.get_name()}: {task}')

		except Exception as e:
			self.logger.error(f'Error during cleanup: {e}')

	async def _update_action_models_for_page(self, page_url: str) -> None:
		"""Update action models with page-specific actions"""
		# Create new action model with current page's filtered actions
		self.ActionModel = self.tools.registry.create_action_model(page_url=page_url)
		# Update output model with the new actions
		if self.settings.flash_mode:
			self.AgentOutput = AgentOutput.type_with_custom_actions_flash_mode(self.ActionModel)
		elif self.settings.use_thinking:
			self.AgentOutput = AgentOutput.type_with_custom_actions(self.ActionModel)
		else:
			self.AgentOutput = AgentOutput.type_with_custom_actions_no_thinking(self.ActionModel)

		# Update done action model too
		self.DoneActionModel = self.tools.registry.create_action_model(include_actions=['done'], page_url=page_url)
		if self.settings.flash_mode:
			self.DoneAgentOutput = AgentOutput.type_with_custom_actions_flash_mode(self.DoneActionModel)
		elif self.settings.use_thinking:
			self.DoneAgentOutput = AgentOutput.type_with_custom_actions(self.DoneActionModel)
		else:
			self.DoneAgentOutput = AgentOutput.type_with_custom_actions_no_thinking(self.DoneActionModel)

	async def authenticate_cloud_sync(self, show_instructions: bool = True) -> bool:
		"""
		Authenticate with cloud service for future runs.

		This is useful when users want to authenticate after a task has completed
		so that future runs will sync to the cloud.

		Args:
			show_instructions: Whether to show authentication instructions to user

		Returns:
			bool: True if authentication was successful
		"""
		self.logger.warning('Cloud sync has been removed and is no longer available')
		return False

	def run_sync(
		self,
		max_steps: int = 100,
		on_step_start: AgentHookFunc | None = None,
		on_step_end: AgentHookFunc | None = None,
	) -> AgentHistoryList[AgentStructuredOutput]:
		"""Synchronous wrapper around the async run method for easier usage without asyncio."""
		import asyncio

		return asyncio.run(self.run(max_steps=max_steps, on_step_start=on_step_start, on_step_end=on_step_end))

	def detect_variables(self) -> dict[str, DetectedVariable]:
		"""Detect reusable variables in agent history"""
		from agent.agent.variable_detector import detect_variables_in_history

		return detect_variables_in_history(self.history)

	def _substitute_variables_in_history(self, history: AgentHistoryList, variables: dict[str, str]) -> AgentHistoryList:
		"""Substitute variables in history with new values for rerunning with different data"""
		from agent.agent.variable_detector import detect_variables_in_history

		# Detect variables in the history
		detected_vars = detect_variables_in_history(history)

		# Build a mapping of original values to new values
		value_replacements: dict[str, str] = {}
		for var_name, new_value in variables.items():
			if var_name in detected_vars:
				old_value = detected_vars[var_name].original_value
				value_replacements[old_value] = new_value
			else:
				self.logger.warning(f'Variable "{var_name}" not found in history, skipping substitution')

		if not value_replacements:
			self.logger.info('No variables to substitute')
			return history

		# Create a deep copy of history to avoid modifying the original
		import copy

		modified_history = copy.deepcopy(history)

		# Substitute values in all actions
		substitution_count = 0
		for history_item in modified_history.history:
			if not history_item.model_output or not history_item.model_output.action:
				continue

			for action in history_item.model_output.action:
				# Handle both Pydantic models and dicts
				if hasattr(action, 'model_dump'):
					action_dict = action.model_dump()
				elif isinstance(action, dict):
					action_dict = action
				else:
					action_dict = vars(action) if hasattr(action, '__dict__') else {}

				# Substitute in all string fields
				substitution_count += self._substitute_in_dict(action_dict, value_replacements)

				# Update the action with modified values
				if hasattr(action, 'model_dump'):
					# For Pydantic RootModel, we need to recreate from the modified dict
					if hasattr(action, 'root'):
						# This is a RootModel - recreate it from the modified dict
						new_action = type(action).model_validate(action_dict)
						# Replace the root field in-place using object.__setattr__ to bypass Pydantic's immutability
						object.__setattr__(action, 'root', getattr(new_action, 'root'))
					else:
						# Regular Pydantic model - update fields in-place
						for key, val in action_dict.items():
							if hasattr(action, key):
								setattr(action, key, val)
				elif isinstance(action, dict):
					action.update(action_dict)

		self.logger.info(f'Substituted {substitution_count} value(s) in {len(value_replacements)} variable type(s) in history')
		return modified_history

	def _substitute_in_dict(self, data: dict, replacements: dict[str, str]) -> int:
		"""Recursively substitute values in a dictionary, returns count of substitutions made"""
		count = 0
		for key, value in data.items():
			if isinstance(value, str):
				# Replace if exact match
				if value in replacements:
					data[key] = replacements[value]
					count += 1
			elif isinstance(value, dict):
				# Recurse into nested dicts
				count += self._substitute_in_dict(value, replacements)
			elif isinstance(value, list):
				# Handle lists
				for i, item in enumerate(value):
					if isinstance(item, str) and item in replacements:
						value[i] = replacements[item]
						count += 1
					elif isinstance(item, dict):
						count += self._substitute_in_dict(item, replacements)
		return count
