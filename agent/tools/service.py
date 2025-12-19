import asyncio
import json
import logging
import os
from collections.abc import Callable
from typing import Any, Generic, TypeVar

try:
	from lmnr import Laminar  # type: ignore
except ImportError:
	Laminar = None  # type: ignore
from pydantic import BaseModel

from agent.agent.views import ActionModel, ActionResult
from agent.browser import BrowserSession
from agent.browser.events import (
	ClickCoordinateEvent,
	ClickElementEvent,
	GetDropdownOptionsEvent,
	GoBackEvent,
	NavigateToUrlEvent,
	ScrollEvent,
	ScrollToTextEvent,
	SendKeysEvent,
	TypeTextEvent,
	UploadFileEvent,
)
from agent.browser.views import BrowserError
from agent.dom.service import EnhancedDOMTreeNode
from agent.llm.base import BaseChatModel
from agent.llm.messages import SystemMessage, UserMessage
from agent.observability import observe_debug
from agent.tools.registry.service import Registry
from agent.tools.utils import get_click_description
from agent.tools.views import (
	ClickElementAction,
	ClickRoleAction,
	ClickTextAction,
	DoneAction,
	ExtractAction,
	GetDropdownOptionsAction,
	InputTextAction,
	NavigateAction,
	NoParamsAction,
	RequestUserInputAction,
	ScrollAction,
	SelectDropdownOptionAction,
	SendKeysAction,
	StructuredOutputAction,
	WaitForUserInputAction,
)
from agent.utils import create_task_with_error_handling, sanitize_surrogates, time_execution_sync

logger = logging.getLogger(__name__)

# Импортируем EnhancedDOMTreeNode и пересобираем модели событий с прямыми ссылками на него
# Это должно быть сделано после завершения всех импортов
ClickElementEvent.model_rebuild()
TypeTextEvent.model_rebuild()
ScrollEvent.model_rebuild()
UploadFileEvent.model_rebuild()

Context = TypeVar('Context')

T = TypeVar('T', bound=BaseModel)


def _detect_sensitive_key_name(text: str, sensitive_data: dict[str, str | dict[str, str]] | None) -> str | None:
	"""Определяет, какому ключу чувствительных данных соответствует данное текстовое значение."""
	if not sensitive_data or not text:
		return None

	# Собираем все чувствительные значения и их ключи
	for domain_or_key, content in sensitive_data.items():
		if isinstance(content, dict):
			# Новый формат: {domain: {key: value}}
			for key, value in content.items():
				if value and value == text:
					return key
		elif content:  # Формат: {key: value}
			if content == text:
				return domain_or_key

	return None


def handle_browser_error(e: BrowserError) -> ActionResult:
	if e.long_term_memory is not None:
		if e.short_term_memory is not None:
			return ActionResult(
				extracted_content=e.short_term_memory, error=e.long_term_memory, include_extracted_content_only_once=True
			)
		else:
			return ActionResult(error=e.long_term_memory)
	# Возвращаемся к исходной обработке ошибок, если long_term_memory равен None
	logger.warning(
		'⚠️ A BrowserError was raised without long_term_memory - always set long_term_memory when raising BrowserError to propagate right messages to LLM.'
	)
	raise e


class Tools(Generic[Context]):
	def __init__(
		self,
		exclude_actions: list[str] | None = None,
		output_model: type[T] | None = None,
		display_files_in_done_text: bool = True,
		user_input_callback: Callable[[str], str] | None = None,
	):
		self.registry = Registry[Context](exclude_actions if exclude_actions is not None else [])
		self.display_files_in_done_text = display_files_in_done_text
		self._output_model: type[BaseModel] | None = output_model
		self.user_input_callback = user_input_callback

		"""Регистрирует все стандартные действия браузера"""

		self._register_done_action(output_model)

		# Базовые действия навигации
		@self.registry.action(
			'',
			param_model=NavigateAction,
		)
		async def navigate(params: NavigateAction, browser_session: BrowserSession):
			try:
				# ВАЖНО: Принудительно отключаем открытие новых вкладок
				# LLM иногда решает открыть new_tab=True, что ломает контекст работы
				# Все навигации должны происходить в текущей вкладке
				event = browser_session.event_bus.dispatch(NavigateToUrlEvent(url=params.url, new_tab=False))
				await event
				await event.event_result(raise_if_any=True, raise_if_none=False)

				memory = f'Переход на {params.url}'
				msg = f'🔗 {memory}'

				logger.info(msg)
				return ActionResult(extracted_content=msg, long_term_memory=memory)
			except Exception as e:
				error_msg = str(e)
				# Всегда логируем реальную ошибку сначала для отладки
				browser_session.logger.error(f'❌ Навигация не удалась: {error_msg}')

				# Проверка на RuntimeError о CDP клиенте
				if isinstance(e, RuntimeError) and 'CDP client not initialized' in error_msg:
					browser_session.logger.error('❌ Ошибка подключения браузера - CDP клиент не инициализирован')
					return ActionResult(error=f'Ошибка подключения браузера: {error_msg}')
				# Проверка на сетевые ошибки
				elif any(
					err in error_msg
					for err in [
						'ERR_NAME_NOT_RESOLVED',
						'ERR_INTERNET_DISCONNECTED',
						'ERR_CONNECTION_REFUSED',
						'ERR_TIMED_OUT',
						'net::',
					]
				):
					site_unavailable_msg = f'Навигация не удалась - сайт недоступен: {params.url}'
					browser_session.logger.warning(f'⚠️ {site_unavailable_msg} - {error_msg}')
					return ActionResult(error=site_unavailable_msg)
				else:
					# Возвращаем ошибку в ActionResult вместо повторного выброса
					return ActionResult(error=f'Навигация не удалась: {str(e)}')

		@self.registry.action('Назад', param_model=NoParamsAction)
		async def go_back(_: NoParamsAction, browser_session: BrowserSession):
			try:
				event = browser_session.event_bus.dispatch(GoBackEvent())
				await event
				memory = 'Вернулся назад'
				msg = f'🔙  {memory}'
				logger.info(msg)
				return ActionResult(extracted_content=memory)
			except Exception as e:
				logger.error(f'Не удалось отправить GoBackEvent: {type(e).__name__}: {e}')
				error_msg = f'Не удалось вернуться назад: {str(e)}'
				return ActionResult(error=error_msg)

		@self.registry.action('Wait for x seconds.')
		async def wait(seconds: int = 3):
			# Ограничиваем время ожидания максимумом в 30 секунд
			# Уменьшаем время ожидания на 3 секунды, чтобы учесть вызов LLM, который занимает минимум 3 секунды
			# Так что если модель решает ждать 5 секунд, вызов LLM занял минимум 3 секунды, поэтому нужно ждать только 2 секунды
			# Примечание от Mert: вышесказанное не имеет смысла, потому что мы делаем вызов LLM сразу после этого, или это может быть после другого действия, после которого мы хотим подождать
			# поэтому я откатываю это.
			actual_seconds = min(max(seconds - 1, 0), 30)
			sec_text = 'секунду' if seconds == 1 else ('секунды' if seconds < 5 else 'секунд')
			memory = f'Ожидание {seconds} {sec_text}'
			logger.info(f'🕒 ожидание {seconds} {sec_text}')
			await asyncio.sleep(actual_seconds)
			return ActionResult(extracted_content=memory, long_term_memory=memory)

		# Вспомогательная функция для преобразования координат
		def _convert_llm_coordinates_to_viewport(llm_x: int, llm_y: int, browser_session: BrowserSession) -> tuple[int, int]:
			"""Преобразует координаты из размера скриншота LLM в исходный размер viewport."""
			if browser_session.llm_screenshot_size and browser_session._original_viewport_size:
				original_width, original_height = browser_session._original_viewport_size
				llm_width, llm_height = browser_session.llm_screenshot_size

				# Преобразуем координаты используя дроби
				actual_x = int((llm_x / llm_width) * original_width)
				actual_y = int((llm_y / llm_height) * original_height)

				logger.info(
					f'🔄 Converting coordinates: LLM ({llm_x}, {llm_y}) @ {llm_width}x{llm_height} '
					f'→ Viewport ({actual_x}, {actual_y}) @ {original_width}x{original_height}'
				)
				return actual_x, actual_y
			return llm_x, llm_y

		# Действия взаимодействия с элементами
		async def _click_by_coordinate(params: ClickElementAction, browser_session: BrowserSession) -> ActionResult:
			# Убеждаемся, что координаты предоставлены (проверка типов)
			if params.coordinate_x is None or params.coordinate_y is None:
				return ActionResult(error='Both coordinate_x and coordinate_y must be provided')

			try:
				# Преобразуем координаты из размера LLM в исходный размер viewport, если использовалось изменение размера
				actual_x, actual_y = _convert_llm_coordinates_to_viewport(
					params.coordinate_x, params.coordinate_y, browser_session
				)

				# Подсвечиваем координату, по которой кликаем (действительно неблокирующая операция)
				asyncio.create_task(browser_session.highlight_coordinate_click(actual_x, actual_y))

				# Отправляем ClickCoordinateEvent - обработчик проверит безопасность и кликнет
				event = browser_session.event_bus.dispatch(
					ClickCoordinateEvent(coordinate_x=actual_x, coordinate_y=actual_y, force=True)
				)
				await event
				# Ждём завершения обработчика и получаем любое исключение или метаданные
				click_metadata = await event.event_result(raise_if_any=True, raise_if_none=False)

				# Проверяем ошибки валидации (происходит только когда force=False)
				if isinstance(click_metadata, dict) and 'validation_error' in click_metadata:
					error_msg = click_metadata['validation_error']
					return ActionResult(error=error_msg)

				memory = f'Клик по координатам {params.coordinate_x}, {params.coordinate_y}'
				msg = f'🖱️ {memory}'
				logger.info(msg)

				return ActionResult(
					extracted_content=memory,
					metadata={'click_x': actual_x, 'click_y': actual_y},
				)
			except BrowserError as e:
				return handle_browser_error(e)
			except Exception as e:
				error_msg = f'Не удалось кликнуть по координатам ({params.coordinate_x}, {params.coordinate_y}).'
				return ActionResult(error=error_msg)

		async def _click_by_index(params: ClickElementAction, browser_session: BrowserSession) -> ActionResult:
			assert params.index is not None
			try:
				# Индексы могут начинаться с 0, но должны быть валидными
				if params.index < 0:
					msg = f'Индекс {params.index} невалиден. Индексы должны быть >= 0.'
					logger.warning(f'⚠️ {msg}')
					return ActionResult(extracted_content=msg)

				# Поиск узла в карте селекторов
				node = await browser_session.get_element_by_index(params.index)
				if node is None:
					msg = f'Элемент с индексом {params.index} недоступен - страница могла измениться. Попробуйте обновить состояние браузера.'
					logger.warning(f'⚠️ {msg}')
					return ActionResult(extracted_content=msg)

				# Получение описания кликнутого элемента
				element_desc = get_click_description(node)

				# Подсветка элемента, на который кликают (неблокирующая)
				create_task_with_error_handling(
					browser_session.highlight_interaction_element(node), name='highlight_click_element', suppress_exceptions=True
				)

				event = browser_session.event_bus.dispatch(ClickElementEvent(node=node))
				await event
				# Ждём завершения обработчика и получаем любое исключение или метаданные
				click_metadata = await event.event_result(raise_if_any=True, raise_if_none=False)

				# Проверка на ошибку валидации (например, попытка кликнуть на <select> или file input)
				if isinstance(click_metadata, dict) and 'validation_error' in click_metadata:
					error_msg = click_metadata['validation_error']
					# Если это select элемент, попробуем получить опции выпадающего списка как полезное сокращение
					if 'Cannot click on <select> elements.' in error_msg:
						try:
							return await dropdown_options(
								params=GetDropdownOptionsAction(index=params.index), browser_session=browser_session
							)
						except Exception as dropdown_error:
							logger.debug(
								f'Failed to get dropdown options as shortcut during click on dropdown: {type(dropdown_error).__name__}: {dropdown_error}'
							)
					return ActionResult(error=error_msg)

				# Формирование памяти с информацией об элементе
				memory = f'Клик по {element_desc}'
				logger.info(f'🖱️ {memory}')

				# Включаем координаты клика в метаданные, если доступны
				return ActionResult(
					extracted_content=memory,
					metadata=click_metadata if isinstance(click_metadata, dict) else None,
				)
			except BrowserError as e:
				return handle_browser_error(e)
			except Exception as e:
				error_msg = f'Не удалось кликнуть на элемент {params.index}: {str(e)}'
				return ActionResult(error=error_msg)

		@self.registry.action(
			'Клик по элементу по индексу или координатам. Предпочитайте индекс координатам, когда возможно. Укажите либо координаты, либо индекс.',
			param_model=ClickElementAction,
		)
		async def click(params: ClickElementAction, browser_session: BrowserSession):
			# Проверяем, что предоставлен либо индекс, либо координаты
			if params.index is None and (params.coordinate_x is None or params.coordinate_y is None):
				return ActionResult(error='Must provide either index or both coordinate_x and coordinate_y')

			# Пробуем клик по индексу сначала, если индекс предоставлен
			if params.index is not None:
				return await _click_by_index(params, browser_session)
			# Клик по координатам, когда индекс не предоставлен
			else:
				return await _click_by_coordinate(params, browser_session)

		@self.registry.action(
			'Ввод текста в элемент по индексу. Работает только с индексом, НИКОГДА не используйте координаты для ввода текста.',
			param_model=InputTextAction,
		)
		async def input(
			params: InputTextAction,
			browser_session: BrowserSession,
			has_sensitive_data: bool = False,
			sensitive_data: dict[str, str | dict[str, str]] | None = None,
		):
			# Поиск узла в карте селекторов
			node = await browser_session.get_element_by_index(params.index)
			if node is None:
				msg = f'Элемент с индексом {params.index} недоступен - страница могла измениться. Попробуйте обновить состояние браузера.'
				logger.warning(f'⚠️ {msg}')
				return ActionResult(extracted_content=msg)

			# Подсветка элемента, в который вводят (неблокирующая)
			create_task_with_error_handling(
				browser_session.highlight_interaction_element(node), name='highlight_type_element', suppress_exceptions=True
			)

			# Отправляем событие ввода текста с узлом
			try:
				# Определяем, какой ключ чувствительных данных используется
				sensitive_key_name = None
				if has_sensitive_data and sensitive_data:
					sensitive_key_name = _detect_sensitive_key_name(params.text, sensitive_data)

				event = browser_session.event_bus.dispatch(
					TypeTextEvent(
						node=node,
						text=params.text,
						clear=params.clear,
						is_sensitive=has_sensitive_data,
						sensitive_key_name=sensitive_key_name,
					)
				)
				await event
				input_metadata = await event.event_result(raise_if_any=True, raise_if_none=False)

				# Создание сообщения с обработкой чувствительных данных
				if has_sensitive_data:
					if sensitive_key_name:
						msg = f'Введено {sensitive_key_name}'
						log_msg = f'Введено <{sensitive_key_name}>'
					else:
						msg = 'Введены чувствительные данные'
						log_msg = 'Введено <чувствительные>'
				else:
					msg = f"Введено '{params.text}'"
					log_msg = f"Введено '{params.text}'"

				logger.debug(log_msg)

				# Если указан press_enter=True, нажимаем Enter после ввода текста
				# Это особенно полезно для полей поиска, где кнопка поиска может быть неточно определена
				if params.press_enter:
					try:
						enter_event = browser_session.event_bus.dispatch(SendKeysEvent(keys='Enter'))
						await enter_event
						await enter_event.event_result(raise_if_any=True, raise_if_none=False)
						msg += ' и нажат Enter'
						logger.info('⏎ Enter нажат после ввода текста')
					except Exception as e:
						logger.warning(f'Не удалось нажать Enter: {e}')

				# Включаем координаты ввода в метаданные, если доступны
				return ActionResult(
					extracted_content=msg,
					long_term_memory=msg,
					metadata=input_metadata if isinstance(input_metadata, dict) else None,
				)
			except BrowserError as e:
				return handle_browser_error(e)
			except Exception as e:
				# Логирование полной ошибки для отладки
				logger.error(f'Не удалось отправить TypeTextEvent: {type(e).__name__}: {e}')
				error_msg = f'Не удалось ввести текст в элемент {params.index}: {e}'
				return ActionResult(error=error_msg)


		@self.registry.action(
			"""LLM извлекает структурированные данные из markdown страницы. Используйте когда: на правильной странице, знаете что извлекать, не вызывали ранее на той же странице+запросе. Не может получить интерактивные элементы. Установите extract_links=True для адресов. Используйте start_from_char если предыдущее извлечение было обрезано для извлечения данных дальше по странице.""",
			param_model=ExtractAction,
		)
		async def extract(
			params: ExtractAction,
			browser_session: BrowserSession,
			page_extraction_llm: BaseChatModel,
		):
			# Константы
			MAX_CHAR_LIMIT = 30000
			query = params['query'] if isinstance(params, dict) else params.query
			extract_links = params['extract_links'] if isinstance(params, dict) else params.extract_links
			start_from_char = params['start_from_char'] if isinstance(params, dict) else params.start_from_char

			# Извлекаем чистый markdown используя унифицированный метод
			try:
				from agent.dom.markdown_extractor import extract_clean_markdown

				content, content_stats = await extract_clean_markdown(
					browser_session=browser_session, extract_links=extract_links
				)
			except Exception as e:
				raise RuntimeError(f'Не удалось извлечь чистый markdown: {type(e).__name__}')

			# Исходная длина контента для обработки
			final_filtered_length = content_stats['final_filtered_chars']

			if start_from_char > 0:
				if start_from_char >= len(content):
					return ActionResult(
						error=f'start_from_char ({start_from_char}) превышает длину контента {final_filtered_length} символов.'
					)
				content = content[start_from_char:]
				content_stats['started_from_char'] = start_from_char

			# Умное обрезание с сохранением контекста
			truncated = False
			if len(content) > MAX_CHAR_LIMIT:
				# Пробуем обрезать на естественной точке разрыва (абзац, предложение)
				truncate_at = MAX_CHAR_LIMIT

				# Ищем разрыв абзаца в последних 500 символах от лимита
				paragraph_break = content.rfind('\n\n', MAX_CHAR_LIMIT - 500, MAX_CHAR_LIMIT)
				if paragraph_break > 0:
					truncate_at = paragraph_break
				else:
					# Ищем разрыв предложения в последних 200 символах от лимита
					sentence_break = content.rfind('.', MAX_CHAR_LIMIT - 200, MAX_CHAR_LIMIT)
					if sentence_break > 0:
						truncate_at = sentence_break + 1

				content = content[:truncate_at]
				truncated = True
				next_start = (start_from_char or 0) + truncate_at
				content_stats['truncated_at_char'] = truncate_at
				content_stats['next_start_char'] = next_start

			# Добавляем статистику контента в результат
			original_html_length = content_stats['original_html_chars']
			initial_markdown_length = content_stats['initial_markdown_chars']
			chars_filtered = content_stats['filtered_chars_removed']

			stats_summary = f"""Content processed: {original_html_length:,} HTML chars → {initial_markdown_length:,} initial markdown → {final_filtered_length:,} filtered markdown"""
			if start_from_char > 0:
				stats_summary += f' (started from char {start_from_char:,})'
			if truncated:
				stats_summary += f' → {len(content):,} final chars (truncated, use start_from_char={content_stats["next_start_char"]} to continue)'
			elif chars_filtered > 0:
				stats_summary += f' (filtered {chars_filtered:,} chars of noise)'

			system_prompt = """
You are an expert at extracting data from the markdown of a webpage.

<input>
You will be given a query and the markdown of a webpage that has been filtered to remove noise and advertising content.
</input>

<instructions>
- You are tasked to extract information from the webpage that is relevant to the query.
- You should ONLY use the information available in the webpage to answer the query. Do not make up information or provide guess from your own knowledge.
- If the information relevant to the query is not available in the page, your response should mention that.
- If the query asks for all items, products, etc., make sure to directly list all of them.
- If the content was truncated and you need more information, note that the user can use start_from_char parameter to continue from where truncation occurred.
</instructions>

<output>
- Your output should present ALL the information relevant to the query in a concise way.
- Do not answer in conversational format - directly output the relevant information or that the information is unavailable.
</output>
""".strip()

			# Очищаем суррогаты из контента, чтобы предотвратить ошибки кодировки UTF-8
			content = sanitize_surrogates(content)
			query = sanitize_surrogates(query)

			prompt = f'<query>\n{query}\n</query>\n\n<content_stats>\n{stats_summary}\n</content_stats>\n\n<webpage_content>\n{content}\n</webpage_content>'

			try:
				response = await asyncio.wait_for(
					page_extraction_llm.ainvoke([SystemMessage(content=system_prompt), UserMessage(content=prompt)]),
					timeout=120.0,
				)

				current_url = await browser_session.get_current_page_url()
				extracted_content = (
					f'<url>\n{current_url}\n</url>\n<query>\n{query}\n</query>\n<result>\n{response.completion}\n</result>'
				)

				# Simple memory handling (без сохранения в файлы для простых задач)
				MAX_MEMORY_LENGTH = 1000
				if len(extracted_content) < MAX_MEMORY_LENGTH:
					memory = extracted_content
					include_extracted_content_only_once = False
				else:
					# Обрезаем память, но не сохраняем в файл
					memory = f'Запрос: {query}\nРезультат: {extracted_content[:MAX_MEMORY_LENGTH]}... (обрезано, полный контент в состоянии_чтения)'
					include_extracted_content_only_once = True

				logger.info(f'📄 {memory}')
				return ActionResult(
					extracted_content=extracted_content,
					include_extracted_content_only_once=include_extracted_content_only_once,
					long_term_memory=memory,
				)
			except Exception as e:
				logger.debug(f'Ошибка при извлечении контента: {e}')
				raise RuntimeError(str(e))

		@self.registry.action(
			"""Прокрутка по страницам. ОБЯЗАТЕЛЬНО: down=True/False (True=вниз, False=вверх, по умолчанию=True). Опционально: pages=0.5-10.0 (по умолчанию 1.0). Используйте index для контейнеров прокрутки (выпадающие списки/кастомный UI). Большое количество страниц (10) достигает низа. Многостраничная прокрутка последовательно. Высота на основе viewport, резерв 1000px/страница.""",
			param_model=ScrollAction,
		)
		async def scroll(params: ScrollAction, browser_session: BrowserSession):
			try:
				# Look up the node from the selector map if index is provided
				# Special case: index 0 means scroll the whole page (root/body element)
				node = None
				if params.index is not None and params.index != 0:
					node = await browser_session.get_element_by_index(params.index)
					if node is None:
						# Элемент не существует
						msg = f'Элемент с индексом {params.index} не найден в состоянии браузера'
						return ActionResult(error=msg)

				direction = 'down' if params.down else 'up'
				target = f'element {params.index}' if params.index is not None and params.index != 0 else ''

				# Get actual viewport height for more accurate scrolling
				try:
					cdp_session = await browser_session.get_or_create_cdp_session()
					metrics = await cdp_session.cdp_client.send.Page.getLayoutMetrics(session_id=cdp_session.session_id)

					# Use cssVisualViewport for the most accurate representation
					css_viewport = metrics.get('cssVisualViewport', {})
					css_layout_viewport = metrics.get('cssLayoutViewport', {})

					# Get viewport height, prioritizing cssVisualViewport
					viewport_height = int(css_viewport.get('clientHeight') or css_layout_viewport.get('clientHeight', 1000))

					logger.debug(f'Detected viewport height: {viewport_height}px')
				except Exception as e:
					viewport_height = 1000  # Fallback to 1000px
					logger.debug(f'Failed to get viewport height, using fallback 1000px: {e}')

				# For multiple pages (>=1.0), scroll one page at a time to ensure each scroll completes
				if params.pages >= 1.0:
					import asyncio

					num_full_pages = int(params.pages)
					remaining_fraction = params.pages - num_full_pages

					completed_scrolls = 0

					# Scroll one page at a time
					for i in range(num_full_pages):
						try:
							pixels = viewport_height  # Use actual viewport height
							if not params.down:
								pixels = -pixels

							event = browser_session.event_bus.dispatch(
								ScrollEvent(direction=direction, amount=abs(pixels), node=node)
							)
							await event
							await event.event_result(raise_if_any=True, raise_if_none=False)
							completed_scrolls += 1

							# Small delay to ensure scroll completes before next one
							await asyncio.sleep(0.15)

						except Exception as e:
							logger.warning(f'Scroll {i + 1}/{num_full_pages} failed: {e}')
							# Continue with remaining scrolls even if one fails

					# Handle fractional page if present
					if remaining_fraction > 0:
						try:
							pixels = int(remaining_fraction * viewport_height)
							if not params.down:
								pixels = -pixels

							event = browser_session.event_bus.dispatch(
								ScrollEvent(direction=direction, amount=abs(pixels), node=node)
							)
							await event
							await event.event_result(raise_if_any=True, raise_if_none=False)
							completed_scrolls += remaining_fraction

						except Exception as e:
							logger.warning(f'Fractional scroll failed: {e}')

					if params.pages == 1.0:
						direction_ru = 'вниз' if direction == 'down' else 'вверх'
						long_term_memory = f'Прокручено {direction_ru} {target} {viewport_height}px'.replace('  ', ' ')
					else:
						direction_ru = 'вниз' if direction == 'down' else 'вверх'
						long_term_memory = f'Прокручено {direction_ru} {target} {completed_scrolls:.1f} страниц'.replace('  ', ' ')
				else:
					# For fractional pages <1.0, do single scroll
					pixels = int(params.pages * viewport_height)
					event = browser_session.event_bus.dispatch(
						ScrollEvent(direction='down' if params.down else 'up', amount=pixels, node=node)
					)
					await event
					await event.event_result(raise_if_any=True, raise_if_none=False)
					direction_ru = 'вниз' if direction == 'down' else 'вверх'
					long_term_memory = f'Прокручено {direction_ru} {target} {params.pages} страниц'.replace('  ', ' ')

				msg = f'🔍 {long_term_memory}'
				logger.info(msg)
				return ActionResult(extracted_content=msg, long_term_memory=long_term_memory)
			except Exception as e:
				logger.error(f'Не удалось отправить ScrollEvent: {type(e).__name__}: {e}')
				error_msg = 'Не удалось выполнить действие прокрутки.'
				return ActionResult(error=error_msg)

		@self.registry.action(
			'Отправка клавиш.',
			param_model=SendKeysAction,
		)
		async def send_keys(params: SendKeysAction, browser_session: BrowserSession):
			# Отправка события клавиш
			try:
				event = browser_session.event_bus.dispatch(SendKeysEvent(keys=params.keys))
				await event
				await event.event_result(raise_if_any=True, raise_if_none=False)
				memory = f'Отправлены клавиши: {params.keys}'
				msg = f'⌨️  {memory}'
				logger.info(msg)
				return ActionResult(extracted_content=memory, long_term_memory=memory)
			except Exception as e:
				logger.error(f'Не удалось отправить SendKeysEvent: {type(e).__name__}: {e}')
				error_msg = f'Не удалось отправить клавиши: {str(e)}'
				return ActionResult(error=error_msg)

		@self.registry.action('Прокрутка к тексту.')
		async def find_text(text: str, browser_session: BrowserSession):  # type: ignore
			# Отправка события прокрутки к тексту
			event = browser_session.event_bus.dispatch(ScrollToTextEvent(text=text))

			try:
				# Обработчик возвращает None при успехе или выбрасывает исключение если текст не найден
				await event.event_result(raise_if_any=True, raise_if_none=False)
				memory = f'Прокручено к тексту: {text}'
				msg = f'🔍  {memory}'
				logger.info(msg)
				return ActionResult(extracted_content=memory, long_term_memory=memory)
			except Exception as e:
				# Текст не найден
				msg = f"Текст '{text}' не найден или не виден на странице"
				logger.info(msg)
				return ActionResult(
					extracted_content=msg,
					long_term_memory=f"Попытка прокрутки к тексту '{text}' не удалась - текст не найден",
				)

		@self.registry.action(
			'Клик по видимому тексту на странице. Используйте когда элемент не имеет индекса в DOM, но текст виден на скриншоте (например, кнопка "Откликнуться", "Submit").',
			param_model=ClickTextAction,
		)
		async def click_text(params: ClickTextAction, browser_session: BrowserSession):
			"""Click element by visible text using JavaScript with full mouse event simulation"""
			try:
				# Use JavaScript to find and click element by text content
				# Includes full mouse event simulation for React/Vue compatibility
				script = """
				(text, exact) => {
					function simulateClick(el) {
						el.scrollIntoView({behavior: 'instant', block: 'center'});
						const rect = el.getBoundingClientRect();
						const x = rect.left + rect.width / 2;
						const y = rect.top + rect.height / 2;
						const opts = {bubbles: true, cancelable: true, view: window, clientX: x, clientY: y};
						el.dispatchEvent(new MouseEvent('mouseenter', opts));
						el.dispatchEvent(new MouseEvent('mouseover', opts));
						el.dispatchEvent(new MouseEvent('mousedown', {...opts, button: 0}));
						el.dispatchEvent(new MouseEvent('mouseup', {...opts, button: 0}));
						el.dispatchEvent(new MouseEvent('click', {...opts, button: 0}));
						if (el.click) el.click();
					}
					
					const elements = document.querySelectorAll('a, button, [role="button"], input[type="submit"], input[type="button"]');
					for (const el of elements) {
						const elText = el.textContent || el.innerText || el.value || '';
						if (exact ? elText.trim() === text : elText.toLowerCase().includes(text.toLowerCase())) {
							simulateClick(el);
							return 'clicked: ' + elText.trim().substring(0, 50);
						}
					}
					// Fallback: try any element with matching text
					const allElements = document.querySelectorAll('*');
					for (const el of allElements) {
						const elText = el.textContent || el.innerText || '';
						if (exact ? elText.trim() === text : elText.toLowerCase().includes(text.toLowerCase())) {
							simulateClick(el);
							return 'clicked (fallback): ' + elText.trim().substring(0, 50);
						}
					}
					return 'not_found';
				}
				"""
				cdp_session = await browser_session.get_or_create_cdp_session()
				result = await cdp_session.cdp_client.send.Runtime.evaluate(
					params={
						'expression': f'({script})("{params.text}", {str(params.exact).lower()})',
						'returnByValue': True,
					}
				)
				
				value = result.get('result', {}).get('value', 'error')
				if value == 'not_found':
					msg = f"Текст '{params.text}' не найден на странице"
					logger.warning(msg)
					return ActionResult(extracted_content=msg)
				
				msg = f"🖱️ click_text: {value}"
				logger.info(msg)
				return ActionResult(extracted_content=msg)
			except Exception as e:
				msg = f"Ошибка click_text: {e}"
				logger.error(msg)
				return ActionResult(error=msg)

		@self.registry.action(
			'Клик по элементу с ARIA ролью (button, link, menuitem). Используйте когда элемент не имеет индекса, но известна его роль и имя.',
			param_model=ClickRoleAction,
		)
		async def click_role(params: ClickRoleAction, browser_session: BrowserSession):
			"""Click element by ARIA role using JavaScript with full mouse event simulation"""
			try:
				role = params.role.lower()
				name = params.name
				
				script = """
				(role, name, exact) => {
					function simulateClick(el) {
						el.scrollIntoView({behavior: 'instant', block: 'center'});
						const rect = el.getBoundingClientRect();
						const x = rect.left + rect.width / 2;
						const y = rect.top + rect.height / 2;
						const opts = {bubbles: true, cancelable: true, view: window, clientX: x, clientY: y};
						el.dispatchEvent(new MouseEvent('mouseenter', opts));
						el.dispatchEvent(new MouseEvent('mouseover', opts));
						el.dispatchEvent(new MouseEvent('mousedown', {...opts, button: 0}));
						el.dispatchEvent(new MouseEvent('mouseup', {...opts, button: 0}));
						el.dispatchEvent(new MouseEvent('click', {...opts, button: 0}));
						if (el.click) el.click();
					}
					
					const roleSelectors = {
						'button': 'button, [role="button"], input[type="button"], input[type="submit"]',
						'link': 'a, [role="link"]',
						'menuitem': '[role="menuitem"]',
						'checkbox': 'input[type="checkbox"], [role="checkbox"]',
						'radio': 'input[type="radio"], [role="radio"]'
					};
					const selector = roleSelectors[role] || '[role="' + role + '"]';
					const elements = document.querySelectorAll(selector);
					
					for (const el of elements) {
						const elText = el.textContent || el.innerText || el.getAttribute('aria-label') || el.value || '';
						const nameMatch = !name || (exact ? elText.trim() === name : elText.toLowerCase().includes(name.toLowerCase()));
						if (nameMatch) {
							simulateClick(el);
							return 'clicked: ' + elText.trim().substring(0, 50);
						}
					}
					return 'not_found';
				}
				"""
				cdp_session = await browser_session.get_or_create_cdp_session()
				result = await cdp_session.cdp_client.send.Runtime.evaluate(
					params={
						'expression': f'({script})("{role}", "{name}", {str(params.exact).lower()})',
						'returnByValue': True,
					}
				)
				
				value = result.get('result', {}).get('value', 'error')
				if value == 'not_found':
					msg = f"Элемент с ролью '{role}' и именем '{name}' не найден"
					logger.warning(msg)
					return ActionResult(extracted_content=msg)
				
				msg = f"🖱️ click_role: {value}"
				logger.info(msg)
				return ActionResult(extracted_content=msg)
			except Exception as e:
				msg = f"Ошибка click_role: {e}"
				logger.error(msg)
				return ActionResult(error=msg)

		@self.registry.action(
			'Получить скриншот текущего viewport. Используйте когда: нужна визуальная проверка, неясная компоновка, неопределенные позиции элементов, отладка проблем UI, или проверка состояния страницы. Скриншот включен в следующее состояние_браузера. Параметры не нужны.',
			param_model=NoParamsAction,
		)
		async def screenshot(_: NoParamsAction):
			"""Запрос включения скриншота в следующее наблюдение"""
			memory = 'Запрошен скриншот для следующего наблюдения'
			msg = f'📸 {memory}'
			logger.info(msg)

			# Return flag in metadata to signal that screenshot should be included
			return ActionResult(
				extracted_content=memory,
				metadata={'include_screenshot': True},
			)

		# Dropdown Actions

		@self.registry.action(
			'',
			param_model=GetDropdownOptionsAction,
		)
		async def dropdown_options(params: GetDropdownOptionsAction, browser_session: BrowserSession):
			"""Получить все опции из нативного выпадающего списка или ARIA меню"""
			# Поиск узла в карте селекторов
			node = await browser_session.get_element_by_index(params.index)
			if node is None:
				msg = f'Элемент с индексом {params.index} недоступен - страница могла измениться. Попробуйте обновить состояние браузера.'
				logger.warning(f'⚠️ {msg}')
				return ActionResult(extracted_content=msg)

			# Dispatch GetDropdownOptionsEvent to the event handler

			event = browser_session.event_bus.dispatch(GetDropdownOptionsEvent(node=node))
			dropdown_data = await event.event_result(timeout=3.0, raise_if_none=True, raise_if_any=True)

			if not dropdown_data:
				raise ValueError('Не удалось получить опции выпадающего списка - данные не возвращены')

			# Use structured memory from the handler
			return ActionResult(
				extracted_content=dropdown_data['short_term_memory'],
				long_term_memory=dropdown_data['long_term_memory'],
				include_extracted_content_only_once=True,
			)

		@self.registry.action(
			'Установить опцию элемента <select>.',
			param_model=SelectDropdownOptionAction,
		)
		async def select_dropdown(params: SelectDropdownOptionAction, browser_session: BrowserSession):
			"""Выбрать опцию выпадающего списка по тексту опции, которую хотите выбрать"""
			# Поиск узла в карте селекторов
			node = await browser_session.get_element_by_index(params.index)
			if node is None:
				msg = f'Элемент с индексом {params.index} недоступен - страница могла измениться. Попробуйте обновить состояние браузера.'
				logger.warning(f'⚠️ {msg}')
				return ActionResult(extracted_content=msg)

			# Dispatch SelectDropdownOptionEvent to the event handler
			from agent.browser.events import SelectDropdownOptionEvent

			event = browser_session.event_bus.dispatch(SelectDropdownOptionEvent(node=node, text=params.text))
			selection_data = await event.event_result()

			if not selection_data:
				raise ValueError('Не удалось выбрать опцию выпадающего списка - данные не возвращены')

			# Проверка успешности выбора
			if selection_data.get('success') == 'true':
				# Извлечение сообщения из возвращенных данных
				msg = selection_data.get('message', f'Выбрана опция: {params.text}')
				return ActionResult(
					extracted_content=msg,
					include_in_memory=True,
					long_term_memory=f"Выбрана опция выпадающего списка '{params.text}' с индексом {params.index}",
				)
			else:
				# Обработка структурированного ответа об ошибке
				if 'short_term_memory' in selection_data and 'long_term_memory' in selection_data:
					return ActionResult(
						extracted_content=selection_data['short_term_memory'],
						long_term_memory=selection_data['long_term_memory'],
						include_extracted_content_only_once=True,
					)
				else:
					# Резервный вариант для обычной ошибки
					error_msg = selection_data.get('error', f'Не удалось выбрать опцию: {params.text}')
					return ActionResult(error=error_msg)

		@self.registry.action(
			'Запросить ввод от пользователя. Используется для решения капчи или других действий, требующих вмешательства пользователя.',
			param_model=RequestUserInputAction,
		)
		async def request_user_input(params: RequestUserInputAction, browser_session: BrowserSession):
			"""Запросить ввод от пользователя (например, для решения капчи)"""
			# Проверяем, является ли это запросом да/нет (security layer)
			# Если промпт содержит "да/yes" или "нет/no", это запрос подтверждения, не нужно просить "готово"
			prompt_lower = params.prompt.lower()
			is_yes_no_prompt = ('да' in prompt_lower or 'yes' in prompt_lower) and ('нет' in prompt_lower or 'no' in prompt_lower)
			
			if self.user_input_callback is None:
				# Если callback не установлен, используем стандартный input()
				import sys
				print(f'\n🔒 {params.prompt}', file=sys.stderr)
				if not is_yes_no_prompt:
					# Для обычных запросов (капча и т.д.) просим "готово"
					print('Введите "готово" (или "done") когда закончите:', file=sys.stderr, end=' ')
				answer = input()
			else:
				# Используем callback функцию
				# Для security layer не добавляем "готово" в промпт
				if is_yes_no_prompt:
					answer = self.user_input_callback(params.prompt)
				else:
					# Для обычных запросов добавляем "готово"
					answer = self.user_input_callback(f'{params.prompt}\nВведите "готово" (или "done") когда закончите:')
			
			# Если ответ "done", "готово" или "yes" (без учета регистра), это подтверждение
			answer_lower = answer.strip().lower()
			if answer_lower in ['done', 'готово', 'yes', 'да']:
				return ActionResult(
					extracted_content='Пользователь подтвердил: действие выполнено (например, капча решена). Продолжаем выполнение задачи.',
					long_term_memory='Пользователь решил капчу или выполнил требуемое действие',
				)
			
			# Иначе возвращаем значение для использования в следующем действии
			return ActionResult(extracted_content=answer)

		@self.registry.action(
			'Ожидание ввода данных от пользователя в браузере. Используется для форм входа/регистрации, чтобы чувствительные данные (пароли, личная информация) не проходили через LLM чат. Пользователь заполнит форму вручную в браузере и введет "готово" когда закончит.',
			param_model=WaitForUserInputAction,
		)
		async def wait_for_user_input(params: WaitForUserInputAction, browser_session: BrowserSession):
			"""Ожидание ввода данных от пользователя в браузере (для форм входа/регистрации)"""
			# Используем сообщение по умолчанию, если не указано
			msg = params.message or "Пожалуйста, заполните форму в браузере (логин, пароль и т.д.)"
			
			if self.user_input_callback is None:
				# Если callback не установлен, используем стандартный input()
				import sys
				prompt_msg = f'\n🔒 SECURITY: {msg}\nВведите "готово" (или "done") когда закончите ввод данных в браузере.\n> '
				print(prompt_msg, file=sys.stderr, end='')
				answer = input()
			else:
				# Используем callback функцию
				prompt_msg = f'\n🔒 SECURITY: {msg}\nВведите "готово" (или "done") когда закончите ввод данных в браузере.\n> '
				answer = self.user_input_callback(prompt_msg)
			
			# Проверяем, что ответ - это подтверждение (только "готово", "done", "yes", "да")
			answer_lower = answer.strip().lower()
			if answer_lower not in ['готово', 'done', 'yes', 'да']:
				return ActionResult(
					error=f'Неверный ответ: ожидалось "готово" или "done", получено: {answer}'
				)
			
			return ActionResult(
				extracted_content='Пользователь подтвердил, что закончил ввод данных. Продолжаем выполнение задачи.',
				long_term_memory='Пользователь заполнил форму входа/регистрации в браузере',
			)


	def _register_done_action(self, output_model: type[T] | None, display_files_in_done_text: bool = True):
		if output_model is not None:
			self.display_files_in_done_text = display_files_in_done_text

			@self.registry.action(
				'Complete task with structured output.',
				param_model=StructuredOutputAction[output_model],
			)
			async def done(params: StructuredOutputAction):
				# Exclude success from the output JSON since it's an internal parameter
				# Use mode='json' to properly serialize enums at all nesting levels
				output_dict = params.data.model_dump(mode='json')

				return ActionResult(
					is_done=True,
					success=params.success,
					extracted_content=json.dumps(output_dict, ensure_ascii=False),
					long_term_memory=f'Task completed. Success Status: {params.success}',
				)

		else:

			@self.registry.action(
				'Complete task.',
				param_model=DoneAction,
			)
			async def done(params: DoneAction):
				user_message = params.text

				len_text = len(params.text)
				len_max_memory = 100
				memory = f'Task completed: {params.success} - {params.text[:len_max_memory]}'
				if len_text > len_max_memory:
					memory += f' - {len_text - len_max_memory} more characters'

				attachments = []

				return ActionResult(
					is_done=True,
					success=params.success,
					extracted_content=user_message,
					long_term_memory=memory,
					attachments=attachments,
				)

	def use_structured_output_action(self, output_model: type[T]):
		self._output_model = output_model
		self._register_done_action(output_model)

	def get_output_model(self) -> type[BaseModel] | None:
		"""Get the output model if structured output is configured."""
		return self._output_model

	# Register ---------------------------------------------------------------

	def action(self, description: str, **kwargs):
		"""Decorator for registering custom actions

		@param description: Describe the LLM what the function does (better description == better function calling)
		"""
		return self.registry.action(description, **kwargs)

	def exclude_action(self, action_name: str) -> None:
		"""Exclude an action from the tools registry.

		This method can be used to remove actions after initialization,
		useful for enforcing constraints like disabling screenshot when use_vision != 'auto'.

		Args:
			action_name: Name of the action to exclude (e.g., 'screenshot')
		"""
		self.registry.exclude_action(action_name)

	# Act --------------------------------------------------------------------
	@observe_debug(ignore_input=True, ignore_output=True, name='act')
	@time_execution_sync('--act')
	async def act(
		self,
		action: ActionModel,
		browser_session: BrowserSession,
		page_extraction_llm: BaseChatModel | None = None,
		sensitive_data: dict[str, str | dict[str, str]] | None = None,
		available_file_paths: list[str] | None = None,
		file_system: Any | None = None,
	) -> ActionResult:
		"""Execute an action"""

		for action_name, params in action.model_dump(exclude_unset=True).items():
			if params is not None:
				# Use Laminar span if available, otherwise use no-op context manager
				if Laminar is not None:
					span_context = Laminar.start_as_current_span(
						name=action_name,
						input={
							'action': action_name,
							'params': params,
						},
						span_type='TOOL',
					)
				else:
					# No-op context manager when lmnr is not available
					from contextlib import nullcontext

					span_context = nullcontext()

				with span_context:
					try:
						result = await self.registry.execute_action(
							action_name=action_name,
							params=params,
							browser_session=browser_session,
							page_extraction_llm=page_extraction_llm,
							file_system=file_system,
							sensitive_data=sensitive_data,
							available_file_paths=available_file_paths,
						)
					except BrowserError as e:
						logger.error(f'❌ Action {action_name} failed with BrowserError: {str(e)}')
						result = handle_browser_error(e)
					except TimeoutError as e:
						logger.error(f'❌ Action {action_name} failed with TimeoutError: {str(e)}')
						result = ActionResult(error=f'{action_name} was not executed due to timeout.')
					except Exception as e:
						# Log the original exception with traceback for observability
						logger.error(f"Action '{action_name}' failed with error: {str(e)}")
						result = ActionResult(error=str(e))

					if Laminar is not None:
						Laminar.set_span_output(result)

				if isinstance(result, str):
					return ActionResult(extracted_content=result)
				elif isinstance(result, ActionResult):
					return result
				elif result is None:
					return ActionResult()
				else:
					raise ValueError(f'Invalid action result type: {type(result)} of {result}')
		return ActionResult()

	def __getattr__(self, name: str):
		"""
		Enable direct action calls like tools.navigate(url=..., browser_session=...).
		Предоставляет упрощенный API для тестов и прямого использования с сохранением обратной совместимости.
		"""
		# Check if this is a registered action
		if name in self.registry.registry.actions:
			from typing import Union

			from pydantic import create_model

			action = self.registry.registry.actions[name]

			# Create a wrapper that calls act() to ensure consistent error handling and result normalization
			async def action_wrapper(**kwargs):
				# Extract browser_session (required positional argument for act())
				browser_session = kwargs.get('browser_session')

				# Separate action params from special params (injected dependencies)
				special_param_names = {
					'browser_session',
					'page_extraction_llm',
					'file_system',
					'available_file_paths',
					'sensitive_data',
				}

				# Extract action params (params for the action itself)
				action_params = {k: v for k, v in kwargs.items() if k not in special_param_names}

				# Extract special params (injected dependencies) - exclude browser_session as it's positional
				special_kwargs = {k: v for k, v in kwargs.items() if k in special_param_names and k != 'browser_session'}

				# Create the param instance
				params_instance = action.param_model(**action_params)

				# Dynamically create an ActionModel with this action
				# Use Union for type compatibility with create_model
				DynamicActionModel = create_model(
					'DynamicActionModel',
					__base__=ActionModel,
					**{name: (Union[action.param_model, None], None)},  # type: ignore
				)

				# Create the action model instance
				action_model = DynamicActionModel(**{name: params_instance})

				# Call act() which has all the error handling, result normalization, and observability
				# browser_session is passed as positional argument (required by act())
				return await self.act(action=action_model, browser_session=browser_session, **special_kwargs)  # type: ignore

			return action_wrapper

		# If not an action, raise AttributeError for normal Python behavior
		raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


Controller = Tools
