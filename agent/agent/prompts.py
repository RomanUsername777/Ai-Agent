import importlib.resources
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Optional

from agent.dom.views import NodeType, SimplifiedNode
from agent.llm.messages import ContentPartImageParam, ContentPartTextParam, ImageURL, SystemMessage, UserMessage
from agent.observability import observe_debug
from agent.utils import is_new_tab_page, sanitize_surrogates

if TYPE_CHECKING:
	from agent.agent.views import AgentStepInfo
	from agent.browser.views import BrowserStateSummary


class SystemPrompt:
	def __init__(
		self,
		max_actions_per_step: int = 3,
		override_system_message: str | None = None,
		extend_system_message: str | None = None,
		use_thinking: bool = True,
		flash_mode: bool = False,
		is_anthropic: bool = False,
		file_system: Any | None = None,
	):
		self.max_actions_per_step = max_actions_per_step
		self.use_thinking = use_thinking
		self.flash_mode = flash_mode
		self.is_anthropic = is_anthropic
		self.file_system: Any | None = file_system
		prompt = ''
		if override_system_message is not None:
			prompt = override_system_message
		else:
			self._load_prompt_template()
			prompt = self.prompt_template.format(max_actions=self.max_actions_per_step)

		if extend_system_message:
			prompt += f'\n{extend_system_message}'

		self.system_message = SystemMessage(content=prompt, cache=True)

	def _load_prompt_template(self) -> None:
		"""Загрузить шаблон системного промпта из markdown-файла."""
		try:
			# В упрощённой версии всегда используем один основной промпт
			template_filename = 'system_prompt.md'

			# Такой способ работает и при разработке, и при установке как пакет
			with importlib.resources.files('agent.agent').joinpath(template_filename).open('r', encoding='utf-8') as f:
				self.prompt_template = f.read()
		except Exception as e:
			raise RuntimeError(f'Failed to load system prompt template: {e}')

	def get_system_message(self) -> SystemMessage:
		"""
		Вернуть готовое системное сообщение для агента.
		"""
		return self.system_message


class AgentMessagePrompt:
	vision_detail_level: Literal['auto', 'low', 'high']

	def __init__(
		self,
		browser_state_summary: 'BrowserStateSummary',
		file_system: Any,
		agent_history_description: str | None = None,
		read_state_description: str | None = None,
		task: str | None = None,
		include_attributes: list[str] | None = None,
		step_info: Optional['AgentStepInfo'] = None,
		page_filtered_actions: str | None = None,
		max_clickable_elements_length: int = 40000,
		sensitive_data: str | None = None,
		available_file_paths: list[str] | None = None,
		screenshots: list[str] | None = None,
		vision_detail_level: Literal['auto', 'low', 'high'] = 'auto',
		include_recent_events: bool = False,
		sample_images: list[ContentPartTextParam | ContentPartImageParam] | None = None,
		read_state_images: list[dict] | None = None,
		llm_screenshot_size: tuple[int, int] | None = None,
		unavailable_skills_info: str | None = None,
		email_subagent: Any | None = None,  # EmailSubAgent для добавления контекста о почтовых интерфейсах
	):
		self.browser_state: 'BrowserStateSummary' = browser_state_summary
		self.file_system: Any | None = file_system
		self.agent_history_description: str | None = agent_history_description
		self.read_state_description: str | None = read_state_description
		self.task: str | None = task
		self.include_attributes = include_attributes
		self.step_info = step_info
		self.page_filtered_actions: str | None = page_filtered_actions
		self.max_clickable_elements_length: int = max_clickable_elements_length
		self.sensitive_data: str | None = sensitive_data
		self.available_file_paths: list[str] | None = available_file_paths
		self.screenshots = screenshots or []
		self.vision_detail_level = vision_detail_level
		self.include_recent_events = include_recent_events
		self.sample_images = sample_images or []
		self.read_state_images = read_state_images or []
		self.unavailable_skills_info: str | None = unavailable_skills_info
		self.llm_screenshot_size = llm_screenshot_size
		self.email_subagent = email_subagent
		assert self.browser_state

	def _extract_page_statistics(self) -> dict[str, int]:
		"""Извлечь агрегированную статистику по странице из DOM-снимка для контекста LLM."""
		stats = {
			'links': 0,
			'iframes': 0,
			'shadow_open': 0,
			'shadow_closed': 0,
			'scroll_containers': 0,
			'images': 0,
			'interactive_elements': 0,
			'total_elements': 0,
		}

		if not self.browser_state.dom_state or not self.browser_state.dom_state._root:
			return stats

		def traverse_node(node: SimplifiedNode) -> None:
			"""Рекурсивно обойти упрощённое дерево DOM и посчитать элементы."""
			if not node or not node.original_node:
				return

			original = node.original_node
			stats['total_elements'] += 1

			# Считаем элементы по типу узла и тегу
			if original.node_type == NodeType.ELEMENT_NODE:
				tag = original.tag_name.lower() if original.tag_name else ''

				if tag == 'a':
					stats['links'] += 1
				elif tag in ('iframe', 'frame'):
					stats['iframes'] += 1
				elif tag == 'img':
					stats['images'] += 1

				# Проверяем, является ли контейнер прокручиваемым
				if original.is_actually_scrollable:
					stats['scroll_containers'] += 1

				# Проверяем, является ли элемент интерактивным
				if node.is_interactive:
					stats['interactive_elements'] += 1

				# Проверяем, является ли элемент хостом shadow DOM
				if node.is_shadow_host:
					# Проверяем, есть ли закрытые shadow-потомки
					has_closed_shadow = any(
						child.original_node.node_type == NodeType.DOCUMENT_FRAGMENT_NODE
						and child.original_node.shadow_root_type
						and child.original_node.shadow_root_type.lower() == 'closed'
						for child in node.children
					)
					if has_closed_shadow:
						stats['shadow_closed'] += 1
					else:
						stats['shadow_open'] += 1

			elif original.node_type == NodeType.DOCUMENT_FRAGMENT_NODE:
				# Фрагмент Shadow DOM - это реальные shadow-корни
				# Не считаем дважды, так как уже считаем их на уровне хоста выше
				pass

			# Обходим потомков
			for child in node.children:
				traverse_node(child)

		traverse_node(self.browser_state.dom_state._root)
		return stats

	@observe_debug(ignore_input=True, ignore_output=True, name='_get_browser_state_description')
	def _get_browser_state_description(self) -> str:
		# Сначала извлекаем статистику страницы
		page_stats = self._extract_page_statistics()

		# Форматируем статистику для LLM
		stats_text = '<page_stats>'
		if page_stats['total_elements'] < 10:
			stats_text += 'Page appears empty (SPA not loaded?) - '
		stats_text += f'{page_stats["links"]} links, {page_stats["interactive_elements"]} interactive, '
		stats_text += f'{page_stats["iframes"]} iframes, {page_stats["scroll_containers"]} scroll containers'
		if page_stats['shadow_open'] > 0 or page_stats['shadow_closed'] > 0:
			stats_text += f', {page_stats["shadow_open"]} shadow(open), {page_stats["shadow_closed"]} shadow(closed)'
		if page_stats['images'] > 0:
			stats_text += f', {page_stats["images"]} images'
		stats_text += f', {page_stats["total_elements"]} total elements'
		stats_text += '</page_stats>\n'

		elements_text = self.browser_state.dom_state.llm_representation(include_attributes=self.include_attributes)

		if len(elements_text) > self.max_clickable_elements_length:
			elements_text = elements_text[: self.max_clickable_elements_length]
			truncated_text = f' (truncated to {self.max_clickable_elements_length} characters)'
		else:
			truncated_text = ''

		has_content_above = False
		has_content_below = False
		# Расширенная информация о странице для модели
		page_info_text = ''
		if self.browser_state.page_info:
			pi = self.browser_state.page_info
			# Вычисляем статистику страницы динамически
			pages_above = pi.pixels_above / pi.viewport_height if pi.viewport_height > 0 else 0
			pages_below = pi.pixels_below / pi.viewport_height if pi.viewport_height > 0 else 0
			has_content_above = pages_above > 0
			has_content_below = pages_below > 0
			total_pages = pi.page_height / pi.viewport_height if pi.viewport_height > 0 else 0
			current_page_position = pi.scroll_y / max(pi.page_height - pi.viewport_height, 1)
			page_info_text = '<page_info>'
			page_info_text += f'{pages_above:.1f} pages above, '
			page_info_text += f'{pages_below:.1f} pages below, '
			page_info_text += f'{total_pages:.1f} total pages'
			page_info_text += '</page_info>\n'
			# , at {current_page_position:.0%} of page
		if elements_text != '':
			if has_content_above:
				if self.browser_state.page_info:
					pi = self.browser_state.page_info
					pages_above = pi.pixels_above / pi.viewport_height if pi.viewport_height > 0 else 0
					elements_text = f'... {pages_above:.1f} pages above ...\n{elements_text}'
			else:
				elements_text = f'[Start of page]\n{elements_text}'
			if not has_content_below:
				elements_text = f'{elements_text}\n[End of page]'
		else:
			elements_text = 'empty page'

		tabs_text = ''
		current_tab_candidates = []

		# Находим вкладки, совпадающие и по URL, и по заголовку, чтобы надёжнее определить текущую вкладку
		for tab in self.browser_state.tabs:
			if tab.url == self.browser_state.url and tab.title == self.browser_state.title:
				current_tab_candidates.append(tab.target_id)

		# Если есть ровно одно совпадение, помечаем его как текущее
		# Иначе не помечаем никакую вкладку как текущую, чтобы избежать путаницы
		current_target_id = current_tab_candidates[0] if len(current_tab_candidates) == 1 else None

		for tab in self.browser_state.tabs:
			tabs_text += f'Tab {tab.target_id[-4:]}: {tab.url} - {tab.title[:30]}\n'

		current_tab_text = f'Current tab: {current_target_id[-4:]}' if current_target_id is not None else ''

		# Проверяем, является ли текущая страница просмотрщиком PDF, и добавляем соответствующее сообщение
		pdf_message = ''
		if self.browser_state.is_pdf_viewer:
			pdf_message = (
				'PDF viewer cannot be rendered. In this page, DO NOT use the extract action as PDF content cannot be rendered. '
			)
			pdf_message += (
				'Use the read_file action on the downloaded PDF in available_file_paths to read the full text content.\n\n'
			)

		# Добавляем недавние события, если они доступны и запрошены
		recent_events_text = ''
		if self.include_recent_events and self.browser_state.recent_events:
			recent_events_text = f'Recent browser events: {self.browser_state.recent_events}\n'

		# Добавляем сообщения о закрытых всплывающих окнах, если есть
		closed_popups_text = ''
		if self.browser_state.closed_popup_messages:
			closed_popups_text = 'Auto-closed JavaScript dialogs:\n'
			for popup_msg in self.browser_state.closed_popup_messages:
				closed_popups_text += f'  - {popup_msg}\n'
			closed_popups_text += '\n'

		# Добавляем контекст от субагента для почты, если доступен
		email_context_text = ''
		if self.email_subagent and self.email_subagent.is_email_client(self.browser_state):
			email_context_text = self.email_subagent.suggest_email_context(self.browser_state)
		
		browser_state = f"""{stats_text}{current_tab_text}
Available tabs:
{tabs_text}
{page_info_text}
{recent_events_text}{closed_popups_text}{pdf_message}{email_context_text}Interactive elements{truncated_text}:
{elements_text}
"""
		return browser_state

	def _get_agent_state_description(self) -> str:
		if self.step_info:
			step_info_description = f'Step{self.step_info.step_number + 1} maximum:{self.step_info.max_steps}\n'
		else:
			step_info_description = ''

		time_str = datetime.now().strftime('%Y-%m-%d')
		step_info_description += f'Today:{time_str}'

		_todo_contents = self.file_system.get_todo_contents() if self.file_system else ''
		if not len(_todo_contents):
			_todo_contents = '[empty todo.md, fill it when applicable]'

		agent_state = f"""
<user_request>
{self.task}
</user_request>
<file_system>
{self.file_system.describe() if self.file_system else 'No file system available'}
</file_system>
<todo_contents>
{_todo_contents}
</todo_contents>
"""
		if self.sensitive_data:
			agent_state += f'<sensitive_data>{self.sensitive_data}</sensitive_data>\n'

		agent_state += f'<step_info>{step_info_description}</step_info>\n'
		if self.available_file_paths:
			available_file_paths_text = '\n'.join(self.available_file_paths)
			agent_state += f'<available_file_paths>{available_file_paths_text}\nUse with absolute paths</available_file_paths>\n'
		return agent_state

	def _resize_screenshot(self, screenshot_b64: str) -> str:
		"""Изменяет размер скриншота до llm_screenshot_size, если настроено."""
		if not self.llm_screenshot_size:
			return screenshot_b64

		try:
			import base64
			import logging
			from io import BytesIO

			from PIL import Image

			img = Image.open(BytesIO(base64.b64decode(screenshot_b64)))
			if img.size == self.llm_screenshot_size:
				return screenshot_b64

			logging.getLogger(__name__).info(
				f'🔄 Resizing screenshot from {img.size[0]}x{img.size[1]} to {self.llm_screenshot_size[0]}x{self.llm_screenshot_size[1]} for LLM'
			)

			img_resized = img.resize(self.llm_screenshot_size, Image.Resampling.LANCZOS)
			buffer = BytesIO()
			img_resized.save(buffer, format='PNG')
			return base64.b64encode(buffer.getvalue()).decode('utf-8')
		except Exception as e:
			logging.getLogger(__name__).warning(f'Failed to resize screenshot: {e}, using original')
			return screenshot_b64

	@observe_debug(ignore_input=True, ignore_output=True, name='get_user_message')
	def get_user_message(self, use_vision: bool = True) -> UserMessage:
		"""Получает полное состояние как одно кешированное сообщение"""
		# Не передаём скриншот модели, если страница - новая вкладка, шаг 0, и вкладка только одна
		if (
			is_new_tab_page(self.browser_state.url)
			and self.step_info is not None
			and self.step_info.step_number == 0
			and len(self.browser_state.tabs) == 1
		):
			use_vision = False

		# Собираем полное описание состояния
		state_description = (
			'<agent_history>\n'
			+ (self.agent_history_description.strip('\n') if self.agent_history_description else '')
			+ '\n</agent_history>\n\n'
		)
		state_description += '<agent_state>\n' + self._get_agent_state_description().strip('\n') + '\n</agent_state>\n'
		state_description += '<browser_state>\n' + self._get_browser_state_description().strip('\n') + '\n</browser_state>\n'
		# Добавляем read_state только если есть содержимое
		read_state_description = self.read_state_description.strip('\n').strip() if self.read_state_description else ''
		if read_state_description:
			state_description += '<read_state>\n' + read_state_description + '\n</read_state>\n'

		if self.page_filtered_actions:
			state_description += '<page_specific_actions>\n'
			state_description += self.page_filtered_actions + '\n'
			state_description += '</page_specific_actions>\n'

		# Добавляем информацию о недоступных навыках, если есть
		if self.unavailable_skills_info:
			state_description += '\n' + self.unavailable_skills_info + '\n'

		# Очищаем суррогаты из всего текстового содержимого
		state_description = sanitize_surrogates(state_description)

		# Проверяем, есть ли изображения для включения (из действия read_file)
		has_images = bool(self.read_state_images)

		if (use_vision is True and self.screenshots) or has_images:
			# Начинаем с текстового описания
			content_parts: list[ContentPartTextParam | ContentPartImageParam] = [ContentPartTextParam(text=state_description)]

			# Добавляем примеры изображений
			content_parts.extend(self.sample_images)

			# Добавляем скриншоты с метками
			for i, screenshot in enumerate(self.screenshots):
				if i == len(self.screenshots) - 1:
					label = 'Current screenshot:'
				else:
					# Используем простую, точную метку, так как у нас нет реальной информации о времени шага
					label = 'Previous screenshot:'

				# Добавляем метку как текстовое содержимое
				content_parts.append(ContentPartTextParam(text=label))

				# Изменяем размер скриншота, если настроен llm_screenshot_size
				processed_screenshot = self._resize_screenshot(screenshot)

				# Добавляем скриншот
				content_parts.append(
					ContentPartImageParam(
						image_url=ImageURL(
							url=f'data:image/png;base64,{processed_screenshot}',
							media_type='image/png',
							detail=self.vision_detail_level,
						),
					)
				)

			# Добавляем изображения из read_state (из действия read_file) перед скриншотами
			for img_data in self.read_state_images:
				img_name = img_data.get('name', 'unknown')
				img_base64 = img_data.get('data', '')

				if not img_base64:
					continue

				# Определяем формат изображения по имени
				if img_name.lower().endswith('.png'):
					media_type = 'image/png'
				else:
					media_type = 'image/jpeg'

				# Добавляем метку
				content_parts.append(ContentPartTextParam(text=f'Image from file: {img_name}'))

				# Добавляем изображение
				content_parts.append(
					ContentPartImageParam(
						image_url=ImageURL(
							url=f'data:{media_type};base64,{img_base64}',
							media_type=media_type,
							detail=self.vision_detail_level,
						),
					)
				)

			return UserMessage(content=content_parts, cache=True)

		return UserMessage(content=state_description, cache=True)


def get_rerun_summary_prompt(original_task: str, total_steps: int, success_count: int, error_count: int) -> str:
	return f'''You are analyzing the completion of a rerun task. Based on the screenshot and execution info, provide a summary.

Original task: {original_task}

Execution statistics:
- Total steps: {total_steps}
- Successful steps: {success_count}
- Failed steps: {error_count}

Analyze the screenshot to determine:
1. Whether the task completed successfully
2. What the final state shows
3. Overall completion status (complete/partial/failed)

Respond with:
- summary: A clear, concise summary of what happened during the rerun
- success: Whether the task completed successfully (true/false)
- completion_status: One of "complete", "partial", or "failed"'''


def get_rerun_summary_message(prompt: str, screenshot_b64: str | None = None) -> UserMessage:
	"""
	Build a UserMessage for rerun summary generation.

	Args:
		prompt: The prompt text
		screenshot_b64: Optional base64-encoded screenshot

	Returns:
		UserMessage with prompt and optional screenshot
	"""
	if screenshot_b64:
		# Со скриншотом: используем многочастное содержимое
		content_parts: list[ContentPartTextParam | ContentPartImageParam] = [
			ContentPartTextParam(type='text', text=prompt),
			ContentPartImageParam(
				type='image_url',
				image_url=ImageURL(url=f'data:image/png;base64,{screenshot_b64}'),
			),
		]
		return UserMessage(content=content_parts)
	else:
		# Без скриншота: используем простое строковое содержимое
		return UserMessage(content=prompt)


def get_ai_step_system_prompt() -> str:
	"""
	Получает системный промпт для действия AI step, используемого при повторном запуске.

	Returns:
		Строка системного промпта для AI step
	"""
	return """
You are an expert at extracting data from webpages.

<input>
You will be given:
1. A query describing what to extract
2. The markdown of the webpage (filtered to remove noise)
3. Optionally, a screenshot of the current page state
</input>

<instructions>
- Extract information from the webpage that is relevant to the query
- ONLY use the information available in the webpage - do not make up information
- If the information is not available, mention that clearly
- If the query asks for all items, list all of them
</instructions>

<output>
- Present ALL relevant information in a concise way
- Do not use conversational format - directly output the relevant information
- If information is unavailable, state that clearly
</output>
""".strip()


def get_ai_step_user_prompt(query: str, stats_summary: str, content: str) -> str:
	"""
	Build user prompt for AI step action.

	Args:
		query: What to extract or analyze
		stats_summary: Content statistics summary
		content: Page markdown content

	Returns:
		Formatted prompt string
	"""
	return f'<query>\n{query}\n</query>\n\n<content_stats>\n{stats_summary}\n</content_stats>\n\n<webpage_content>\n{content}\n</webpage_content>'
