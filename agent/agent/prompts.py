import importlib.resources
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Optional

from agent.dom.views import NodeType, SimplifiedNode
from agent.llm.messages import ContentPartImageParam, ContentPartTextParam, ImageURL, SystemMessage, UserMessage
from agent.observability import observe_debug
from agent.utils import is_new_tab_page, sanitize_surrogates

logger = logging.getLogger(__name__)

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
		
		# УНИВЕРСАЛЬНАЯ ЛОГИКА: сбор информации о кнопках отправки для внутренней фильтрации
		# Агент видит все элементы в browser_state с их текстом, атрибутами и индексами
		# Внутренняя логика помогает корректно идентифицировать кнопки отправки
		recommendations_text = ''
		if self.browser_state.dom_state and hasattr(self.browser_state.dom_state, 'selector_map'):
			selector_map = self.browser_state.dom_state.selector_map
			
			# УНИВЕРСАЛЬНАЯ ЛОГИКА: сбор информации о кнопках отправки для внутренней фильтрации
			# Используется для корректной идентификации кнопок отправки форм на любых сайтах
			clickable_submit_buttons = []
			for element_index, element in selector_map.items():
				element_text = ''
				if hasattr(element, 'ax_node') and element.ax_node and element.ax_node.name:
					element_text = element.ax_node.name
				elif hasattr(element, 'get_all_children_text'):
					element_text = element.get_all_children_text()
				elif hasattr(element, 'get_meaningful_text_for_llm'):
					element_text = element.get_meaningful_text_for_llm()
				elif hasattr(element, 'node_value'):
					element_text = element.node_value or ''
				
				# Универсальные ключевые слова для кнопок отправки
				submit_keywords = ['отправить', 'откликнуться', 'отклик', 'подтвердить', 'сохранить', 'готово', 'далее', 'send', 'submit', 'apply']
				has_submit_text = any(keyword in element_text.lower() for keyword in submit_keywords)
				has_data_qa = False
				if hasattr(element, 'attributes') and element.attributes:
					data_qa_value = element.attributes.get('data-qa')
					has_data_qa = data_qa_value is not None and data_qa_value != False
				has_role_button = hasattr(element, 'attributes') and element.attributes and element.attributes.get('role') == 'button'
				
				if has_submit_text and (has_data_qa or has_role_button):
					is_visible = getattr(element, 'is_visible', False) if hasattr(element, 'is_visible') else False
					is_clickable = False
					if hasattr(element, 'snapshot_node') and element.snapshot_node:
						is_clickable = getattr(element.snapshot_node, 'is_clickable', False) if element.snapshot_node else False
					
					if is_clickable:
						visibility_note = '|HIDDEN' if not is_visible else ''
						priority = 'ВЫСОКИЙ ПРИОРИТЕТ' if has_data_qa else 'ПРИОРИТЕТ'
						clickable_submit_buttons.append((element_index, visibility_note, priority, has_data_qa))
			
			# Проверяем, есть ли открытое модальное окно
			# УНИВЕРСАЛЬНАЯ ЛОГИКА: используем только стандартные ARIA атрибуты (role="dialog" или aria-modal="true")
			# Это работает на любых сайтах
			has_open_dialog = False
			dialog_element_index = None
			for element_index, element in selector_map.items():
				# Пропускаем body и другие некорректные элементы
				tag_name = getattr(element, 'tag_name', '').lower() if hasattr(element, 'tag_name') else ''
				if tag_name == 'body':
					continue
				
				# Проверяем стандартные ARIA атрибуты для модальных окон
				has_role_dialog = hasattr(element, 'attributes') and element.attributes and element.attributes.get('role') in ('dialog', 'alertdialog')
				has_aria_modal = hasattr(element, 'attributes') and element.attributes and element.attributes.get('aria-modal') == 'true'
				is_visible = getattr(element, 'is_visible', False) if hasattr(element, 'is_visible') else False
				
				# Если элемент имеет стандартные признаки модального окна и видим - это модальное окно
				if (has_role_dialog or has_aria_modal) and is_visible:
					has_open_dialog = True
					dialog_element_index = element_index
					break
				
			
			# Также ищем кнопки отправки (не только "Откликнуться", но и "Отправить", "Подтвердить" и т.д.)
			submit_buttons = []
			submit_buttons_in_dialog = []  # Кнопки внутри модального окна
			all_buttons_in_dialog = []  # ВСЕ кнопки внутри модального окна (для случая, когда нет текста "отправить")
			
			# Находим textarea для формы (внутри модального окна или на странице)
			textarea_in_dialog_index = None
			all_textareas_in_range = []
			all_textareas_on_page = []  # Все textarea на странице (для случая, когда модальное окно не найдено)
			
			import logging
			logger = logging.getLogger(__name__)
			
			# Ищем textarea даже если модальное окно не найдено - на некоторых страницах формы находятся прямо на странице
			
			for idx, elem in selector_map.items():
				tag = getattr(elem, 'tag_name', '').lower() if hasattr(elem, 'tag_name') else ''
				if tag == 'textarea':
					# Получаем текст и атрибуты textarea
					elem_text = ''
					if hasattr(elem, 'ax_node') and elem.ax_node and elem.ax_node.name:
						elem_text = elem.ax_node.name
					if not elem_text and hasattr(elem, 'get_all_children_text'):
						try:
							elem_text = elem.get_all_children_text()
						except:
							pass
					
					# Получаем placeholder, data-qa и другие атрибуты
					placeholder_value = None
					data_qa_textarea = None
					is_visible_textarea = getattr(elem, 'is_visible', False) if hasattr(elem, 'is_visible') else False
					if hasattr(elem, 'attributes') and elem.attributes:
						placeholder_value = elem.attributes.get('placeholder', '') or elem.attributes.get('aria-label', '')
						data_qa_textarea = elem.attributes.get('data-qa')
					
					# Объединяем текст для проверки
					combined_text = (elem_text or '') + ' ' + (placeholder_value or '')
					
					# УНИВЕРСАЛЬНАЯ ЛОГИКА: проверяем, что это textarea для формы (по атрибутам или позиции)
					# Используем универсальные признаки: data-qa с "response"/"letter"/"message", или позиция в DOM
					is_cover_letter_textarea = (
						(data_qa_textarea and ('response' in str(data_qa_textarea).lower() or 'letter' in str(data_qa_textarea).lower() or 'message' in str(data_qa_textarea).lower())) or
						('message' in combined_text.lower() or 'comment' in combined_text.lower() or 'note' in combined_text.lower())
					)
					
					textarea_info = {
						'index': idx,
						'text': elem_text[:100] if elem_text else '(без текста)',
						'placeholder': placeholder_value[:100] if placeholder_value else None,
						'data_qa': data_qa_textarea,
						'is_visible': is_visible_textarea,
						'is_cover_letter': is_cover_letter_textarea
					}
					
					if has_open_dialog and dialog_element_index:
						# Если модальное окно найдено, проверяем только textarea в его диапазоне
						if abs(idx - dialog_element_index) < 1000:
							textarea_info['is_after_dialog'] = idx > dialog_element_index
							textarea_info['index_diff'] = abs(idx - dialog_element_index)
							all_textareas_in_range.append(textarea_info)
							# Собираем информацию о textarea для внутренней логики
							# Агент видит все textarea в browser_state и должен сам определить, какой использовать
					else:
						# Если модальное окно НЕ найдено, ищем textarea по всей странице
						all_textareas_on_page.append(textarea_info)
			
			# ВАЖНО: выбираем textarea для использования в логике поиска кнопок
			# Это нужно для того, чтобы найти кнопки отправки после textarea
			if not textarea_in_dialog_index:
				if all_textareas_on_page:
					# Ищем textarea для сопроводительного письма
					cover_letter_textarea = [t for t in all_textareas_on_page if t.get('is_cover_letter')]
					if cover_letter_textarea:
						textarea_in_dialog_index = cover_letter_textarea[0]['index']
					else:
						# Используем первый textarea на странице
						textarea_in_dialog_index = all_textareas_on_page[0]['index']
			
			for element_index, element in selector_map.items():
				element_text = ''
				if hasattr(element, 'ax_node') and element.ax_node and element.ax_node.name:
					element_text = element.ax_node.name
				if not element_text and hasattr(element, 'get_all_children_text'):
					element_text = element.get_all_children_text()
				if not element_text and hasattr(element, 'get_meaningful_text_for_llm'):
					element_text = element.get_meaningful_text_for_llm()
				if not element_text and hasattr(element, 'node_value'):
					element_text = element.node_value or ''
				
				tag_name = getattr(element, 'tag_name', '').lower() if hasattr(element, 'tag_name') else ''
				
				# Определяем тип элемента для фильтрации
				tag_is_button = tag_name == 'button'
				tag_is_a = tag_name == 'a'
				has_role_button = hasattr(element, 'attributes') and element.attributes and element.attributes.get('role') == 'button'
				# КРИТИЧЕСКИ ВАЖНО: исключаем div/span с role="button" из поиска кнопок отправки
				# потому что они часто являются контейнерами с длинным текстом страницы внутри
				tag_is_div_or_span = tag_name in ('div', 'span')
				is_real_button_element = tag_is_button or (tag_is_a and has_role_button)
				# Исключаем div/span с role="button" - это контейнеры, не реальные кнопки
				is_fake_button_container = tag_is_div_or_span and has_role_button
				
				# ДЛЯ КНОПОК: дополнительно пробуем получить текст через get_all_children_text (для вложенных span)
				# ВАЖНО: кнопки могут иметь вложенную структуру с текстом в span внутри button
				if is_real_button_element:
					if not element_text or len(element_text.strip()) < 3:
						# Пробуем получить весь текст из дочерних элементов (включая вложенные span)
						if hasattr(element, 'get_all_children_text'):
							try:
								children_text = element.get_all_children_text()
								if children_text and len(children_text.strip()) > 0:
									# Ограничиваем длину текста - если слишком длинный, это контейнер, а не кнопка
									if len(children_text.strip()) < 200:
										element_text = children_text.strip()
							except:
								pass
						# Также пробуем get_meaningful_text_for_llm
						if (not element_text or len(element_text.strip()) < 3) and hasattr(element, 'get_meaningful_text_for_llm'):
							try:
								meaningful_text = element.get_meaningful_text_for_llm()
								if meaningful_text and len(meaningful_text.strip()) > 0:
									element_text = meaningful_text.strip()
							except:
								pass
				
				# ДЛЯ КНОПОК: дополнительно пробуем получить текст через aria-label или другие атрибуты
				if (tag_name == 'button' or tag_name == 'a') and not element_text and hasattr(element, 'attributes') and element.attributes:
					# Если это кнопка без текста, пробуем получить aria-label
					aria_label_check = element.attributes.get('aria-label')
					if aria_label_check:
						element_text = aria_label_check
				
				# Получаем атрибуты заранее для использования в логике определения is_likely_in_dialog
				data_qa_value = None
				aria_label_value = None
				button_type_value = None
				if hasattr(element, 'attributes') and element.attributes:
					data_qa_value = element.attributes.get('data-qa')
					aria_label_value = element.attributes.get('aria-label')
					button_type_value = element.attributes.get('type')
				
				has_role_button = hasattr(element, 'attributes') and element.attributes and element.attributes.get('role') == 'button'
				tag_is_button = tag_name == 'button'
				tag_is_a = tag_name == 'a'
				is_visible = getattr(element, 'is_visible', False) if hasattr(element, 'is_visible') else False
				is_clickable = False
				if hasattr(element, 'snapshot_node') and element.snapshot_node:
					is_clickable = getattr(element.snapshot_node, 'is_clickable', False) if element.snapshot_node else False
				
				# Проверяем, находится ли элемент внутри модального окна
				# Элемент считается внутри модального окна, если его индекс близок к индексу модального окна
				# (обычно элементы внутри модального окна идут после него в DOM, но могут быть и до него)
				is_likely_in_dialog = False
				if has_open_dialog and dialog_element_index:
					index_diff = abs(element_index - dialog_element_index)
					# Элементы внутри модального окна обычно имеют индекс больше индекса модального окна
					# но могут быть и до него (если модальное окно вложено или кнопки находятся выше в DOM)
					# Используем диапазон 1000 для учета всех возможных случаев
					if index_diff < 1000:
						# ВАЖНО: включаем элементы как после, так и ДО модального окна в широком диапазоне
						# Кнопки отправки могут находиться до модального окна в DOM, но быть его частью
						if element_index > dialog_element_index:
							# Элементы после модального окна - всегда включаем
							is_likely_in_dialog = True
						elif (dialog_element_index - element_index) < 1000:
							# Элементы до модального окна - включаем в широком диапазоне (до 1000 индексов)
							# Это нужно для кнопок отправки, которые могут быть выше в DOM
							is_likely_in_dialog = True
					
					# ДОПОЛНИТЕЛЬНО: если есть textarea и элемент находится ПОСЛЕ textarea (в пределах 1000 элементов),
					# это с высокой вероятностью кнопка отправки формы (даже если индекс дальше от модального окна)
					# ВАЖНО: textarea находится ДО модального окна в DOM, но кнопка отправки должна быть после textarea
					if textarea_in_dialog_index and element_index > textarea_in_dialog_index:
						textarea_diff = element_index - textarea_in_dialog_index
						# Расширяем диапазон поиска: кнопка может быть далеко после textarea (до 1000 элементов)
						if textarea_diff < 1000:
							# УНИВЕРСАЛЬНАЯ ЛОГИКА: дополнительно проверяем: если это button/a элемент или имеет type="submit"/data-qa с "submit", это точно может быть кнопка отправки
							has_submit_attrs_early = button_type_value == 'submit' or (data_qa_value and 'submit' in str(data_qa_value).lower())
							if tag_is_button or tag_is_a or has_role_button or has_submit_attrs_early:
								is_likely_in_dialog = True
				
				# Если это кнопка внутри модального окна, добавляем в список
				# УНИВЕРСАЛЬНАЯ ЛОГИКА: исключаем сам контейнер модального окна
				# Контейнер модального окна обычно имеет длинный текст и не является button или a
				is_dialog_container = (
					element_index == dialog_element_index or
					(len(element_text) > 200 and tag_name not in ('button', 'a'))  # Длинный текст без button/a - это контейнер
				)
				
				# Расширяем проверку: включаем не только button, но и <a> с role="button" и другие кликабельные элементы
				is_button_like = (has_role_button or tag_is_button or (tag_is_a and has_role_button))
				
				# Если элемент внутри модального окна и это реальная кнопка (не контейнер)
				if is_likely_in_dialog and not is_dialog_container:
					# ВАЖНО: включаем ВСЕ кнопки (button, a с role="button") внутри модального окна, даже без текста
					# Кнопки отправки могут не иметь текста в ax_node.name, но быть кликабельными button элементами
					# Расширяем проверку: включаем не только button с role="button", но и просто button элементы
					# НО: исключаем <a> элементы без текста, которые находятся ДО модального окна - это могут быть ссылки на другие страницы
					is_button_inside_dialog = (tag_is_button or (tag_is_a and has_role_button) or has_role_button) and is_clickable
					
					# Исключаем <a> элементы без текста, которые находятся ДО модального окна (это могут быть ссылки на другие страницы)
					# ВАЖНО: исключаем ВСЕ <a> элементы без текста, если они находятся ДО модального окна
					# Это нужно, чтобы избежать кликов по ссылкам, которые ведут на другие страницы
					is_suspicious_link = (
						tag_is_a and 
						(not element_text or element_text.strip() == '' or element_text == '(без текста)') and
						dialog_element_index and element_index < dialog_element_index
						# Исключаем ВСЕ <a> без текста, которые находятся ДО модального окна
					)
					
					if is_button_inside_dialog and not is_suspicious_link:
						# Исключаем кнопки отмены
						cancel_keywords = ['отменить', 'cancel', 'закрыть', 'close', 'назад', 'back', 'удалить', 'delete']
						has_cancel_text = any(keyword in element_text.lower() for keyword in cancel_keywords)
						
						# Получаем дополнительные атрибуты для лучшей идентификации
						data_qa_value = None
						aria_label_value = None
						button_type_value = None
						if hasattr(element, 'attributes') and element.attributes:
							data_qa_value = element.attributes.get('data-qa')
							aria_label_value = element.attributes.get('aria-label')
							button_type_value = element.attributes.get('type')
						
						# Проверяем, является ли это кнопкой отправки по атрибутам (даже если нет текста)
						is_submit_by_attrs = False
						if data_qa_value:
							data_qa_str = str(data_qa_value).lower()
							# Исключаем кнопки закрытия/отмены
							exclude_keywords = ['close', 'cancel', 'отмена', 'закрыть', 'popup-close']
							has_exclude_keyword = any(excl in data_qa_str for excl in exclude_keywords)
							# Ищем признаки отправки (но НЕ "response" - оно может быть в "response-popup-close")
							has_submit_keyword = any(kw in data_qa_str for kw in ['submit', 'отправ', 'send'])
							if has_submit_keyword and not has_exclude_keyword:
								is_submit_by_attrs = True
						if aria_label_value:
							aria_str = str(aria_label_value).lower()
							exclude_keywords = ['close', 'cancel', 'отмена', 'закрыть']
							has_exclude_keyword = any(excl in aria_str for excl in exclude_keywords)
							has_submit_keyword = any(kw in aria_str for kw in ['отправ', 'отклик', 'submit', 'send'])
							if has_submit_keyword and not has_exclude_keyword:
								is_submit_by_attrs = True
						if button_type_value == 'submit':
							is_submit_by_attrs = True
						
						if not has_cancel_text or is_submit_by_attrs:
							all_buttons_in_dialog.append((element_index, element_text[:50] if element_text else '(без текста)', is_visible))
				
				# Ищем кнопки с текстом отправки или по атрибутам
				submit_keywords = ['отправить', 'откликнуться', 'отклик', 'подтвердить', 'сохранить', 'готово', 'далее', 'send', 'submit']
				has_submit_text = False
				if element_text:
					has_submit_text = any(keyword in element_text.lower() for keyword in submit_keywords)
				# КРИТИЧЕСКИ ВАЖНО: если element_text пустой или короткий, пробуем get_all_children_text
				# Кнопки могут быть вложены в div/span, и текст может быть не извлечен через ax_node.name
				if not has_submit_text and hasattr(element, 'get_all_children_text'):
					try:
						all_children_text = element.get_all_children_text()
						if all_children_text:
							has_submit_text = any(keyword in all_children_text.lower() for keyword in submit_keywords)
							if has_submit_text and not element_text:
								element_text = all_children_text.strip()  # Обновляем element_text для дальнейшего использования
					except:
						pass
				
				# СНАЧАЛА проверяем на признаки отмены (это более приоритетно)
				# Исключаем кнопки отмены (по тексту и по атрибутам)
				cancel_keywords = ['отменить', 'cancel', 'закрыть', 'close', 'назад', 'back', 'удалить', 'delete', 'отмена']
				has_cancel_text = any(keyword in element_text.lower() for keyword in cancel_keywords)
				
				# Также проверяем атрибуты на признаки отмены/закрытия
				# (data_qa_value, aria_label_value, button_type_value уже получены выше)
				has_cancel_by_attrs = False
				if data_qa_value:
					data_qa_str = str(data_qa_value).lower()
					# Явные признаки кнопки закрытия/отмены в data-qa
					if any(kw in data_qa_str for kw in ['close', 'cancel', 'popup-close', 'отмена', 'закрыть']):
						has_cancel_by_attrs = True
				
				if aria_label_value:
					aria_str = str(aria_label_value).lower()
					if any(kw in aria_str for kw in ['close', 'cancel', 'отмена', 'закрыть']):
						has_cancel_by_attrs = True
				
				# Кнопка отмены - это либо текст, либо атрибуты указывают на cancel/close
				has_cancel_indicator = has_cancel_text or has_cancel_by_attrs
				
				# ТОЛЬКО ЕСЛИ НЕ кнопка отмены, проверяем на признаки отправки
				has_submit_by_attrs = False
				if not has_cancel_indicator and hasattr(element, 'attributes') and element.attributes:
					# Проверяем data-qa: ищем submit/отправ, но НЕ проверяем "response" (оно может быть в "response-popup-close")
					if data_qa_value:
						data_qa_str = str(data_qa_value).lower()
						# УНИВЕРСАЛЬНАЯ ЛОГИКА: проверяем data-qa с "submit" или type="submit"
						if 'submit' in data_qa_str:
							has_submit_by_attrs = True
						# Ищем другие явные признаки отправки
						has_submit_keyword = any(kw in data_qa_str for kw in ['submit', 'отправ', 'send'])
						if has_submit_keyword and not has_submit_by_attrs:
							has_submit_by_attrs = True
					
					# Проверяем aria-label: ищем submit/отправ/отклик
					if aria_label_value:
						aria_str = str(aria_label_value).lower()
						has_submit_keyword = any(kw in aria_str for kw in ['отправ', 'отклик', 'submit', 'send'])
						if has_submit_keyword:
							has_submit_by_attrs = True
					
					# type="submit" - явный признак кнопки отправки (НО только если нет явных признаков отмены)
					if button_type_value == 'submit' and not has_cancel_indicator:
						has_submit_by_attrs = True
				
				# Кнопка отправки - это либо текст, либо атрибуты указывают на submit (НО ТОЛЬКО ЕСЛИ НЕ кнопка отмены)
				has_submit_indicator = (has_submit_text or has_submit_by_attrs) and not has_cancel_indicator
				
				# УНИВЕРСАЛЬНАЯ ЛОГИКА: проверка на кнопки отправки для внутренней фильтрации
				already_in_submit_buttons_list = any(idx == element_index for idx, _, _, _ in clickable_submit_buttons)
				
				# Расширяем проверку: включаем не только button, но и <a> с role="button"
				# ВАЖНО: также включаем элементы с type="submit" - это явный признак кнопки отправки
				# КРИТИЧЕСКИ ВАЖНО: НЕ включаем div/span с role="button" - это контейнеры, не реальные кнопки
				is_button_like_for_submit = (tag_is_button or (tag_is_a and has_role_button) or button_type_value == 'submit')
				# Исключаем div/span с role="button" - это контейнеры, не реальные кнопки отправки
				if is_fake_button_container:
					is_button_like_for_submit = False
				
				# Если элемент внутри модального окна и имеет текст кнопки отправки, добавляем его даже если он не прошел все проверки
				# ВАЖНО: исключаем сам контейнер модального окна
				is_dialog_container_for_submit = (
					element_index == dialog_element_index or
					(len(element_text) > 200)  # Длинный текст - это контейнер, а не кнопка
				)
				
				# ДОПОЛНИТЕЛЬНО: если есть textarea внутри модального окна, ищем кнопки ПОСЛЕ неё
				# ВАЖНО: textarea может быть ДО модального окна в DOM (index < dialog_element_index), но визуально внутри него
				# Поэтому ищем кнопки ПОСЛЕ textarea (даже если они после модального окна в DOM)
				is_after_textarea = False
				if textarea_in_dialog_index:
					# Если элемент находится после textarea, это может быть кнопка отправки
					if element_index > textarea_in_dialog_index:
						# Расширяем диапазон поиска: кнопка может быть далеко после textarea (до 1000 элементов)
						if (element_index - textarea_in_dialog_index) < 1000:
							is_after_textarea = True
				
				# УНИВЕРСАЛЬНАЯ ЛОГИКА: исключаем кнопки, которые явно НЕ являются кнопками отправки формы
				# Проверяем только универсальные признаки: элементы с явными признаками ссылок/навигации вне формы
				is_not_submit_button = False
				if data_qa_value:
					data_qa_str = str(data_qa_value).lower()
					# Исключаем элементы с явными признаками навигации/ссылок (не кнопки отправки формы)
					if any(kw in data_qa_str for kw in ['link', 'nav', 'menu', 'breadcrumb']):
						is_not_submit_button = True
				
				# УНИВЕРСАЛЬНАЯ ЛОГИКА: приоритет кнопкам с type="submit" или явными признаками отправки
				# Это стандартные признаки кнопок отправки формы на любых сайтах
				if button_type_value == 'submit' or (data_qa_value and 'submit' in str(data_qa_value).lower()):
					# Приоритетная проверка: если элемент в модальном окне или после textarea, добавляем его
					is_in_submit_range = (
						is_likely_in_dialog or  # Внутри модального окна
						(textarea_in_dialog_index and element_index > textarea_in_dialog_index and (element_index - textarea_in_dialog_index) < 500)  # Или до 500 элементов после textarea
					)
					if is_in_submit_range and not has_cancel_indicator and not is_dialog_container_for_submit and not is_not_submit_button:
						submit_text_display = element_text[:50] if element_text else f'(кнопка отправки: type={button_type_value}, data-qa={data_qa_value})'
						submit_buttons_in_dialog.append((element_index, submit_text_display, is_visible))
				
				if is_likely_in_dialog and has_submit_indicator and not has_cancel_indicator and not is_dialog_container_for_submit and not is_not_submit_button:
					# Для элементов внутри модального окна с индикатором отправки (текст или атрибуты) требуем is_clickable и что это кнопка
					# ВАЖНО: даже если кнопка disabled, она может быть отправлена через JavaScript, но приоритет отдаем активным кнопкам
					if is_button_like_for_submit:
						# Приоритет кнопкам с type="submit" или явными признаками отправки
						is_high_priority_submit = button_type_value == 'submit' or (data_qa_value and 'submit' in str(data_qa_value).lower())
						if is_clickable or is_high_priority_submit:  # Даже disabled кнопки type="submit" могут быть важны
							submit_text_display = element_text[:50] if element_text else f'(кнопка отправки по атрибутам: data-qa={data_qa_value}, aria-label={aria_label_value}, type={button_type_value})'
							submit_buttons_in_dialog.append((element_index, submit_text_display, is_visible))
				elif is_likely_in_dialog and is_after_textarea and not has_cancel_indicator and not is_dialog_container_for_submit and not is_not_submit_button:
					# УНИВЕРСАЛЬНАЯ ЛОГИКА: если элемент находится ПОСЛЕ textarea внутри модального окна, это вероятно кнопка отправки
					# Эвристика на основе позиции: кнопка отправки обычно идет сразу после textarea в форме
					textarea_distance = element_index - textarea_in_dialog_index if textarea_in_dialog_index else 9999
					is_close_to_textarea = textarea_distance < 100  # Элементы в пределах 100 позиций после textarea
					
					# Исключаем элементы с явными признаками не-кнопки: scroll buttons, footer links, hidden inputs
					if tag_name == 'input' and button_type_value == 'hidden':
						is_not_submit_button = True
					
					# Добавляем элементы после textarea, которые:
					# 1. Являются button/a с текстом/атрибутами отправки
					# 2. ИЛИ кликабельны, находятся близко к textarea, и НЕ являются явно не-кнопками
					is_likely_submit_after_textarea = (
						is_button_like_for_submit or
						button_type_value == 'submit' or
						(data_qa_value and 'submit' in str(data_qa_value).lower()) or
						(is_clickable and is_close_to_textarea and not is_not_submit_button)  # Эвристика: кликабельный элемент близко к textarea
					)
					# Требуем clickable для обычных элементов, но для type="submit" или близких к textarea - не требуем
					can_add_after_textarea = is_likely_submit_after_textarea and (is_clickable or button_type_value == 'submit' or is_close_to_textarea)
					if can_add_after_textarea:
						submit_text_display = element_text[:50] if element_text else f'(кнопка отправки после textarea: data-qa={data_qa_value}, aria-label={aria_label_value}, type={button_type_value})'
						submit_buttons_in_dialog.append((element_index, submit_text_display, is_visible))
				
				# УНИВЕРСАЛЬНАЯ ЛОГИКА: если элемент находится внутри модального окна и является кнопкой с текстом отправки,
				# добавляем его в submit_buttons_in_dialog (последний шанс найти кнопку отправки)
				if is_likely_in_dialog and not is_dialog_container_for_submit and not has_cancel_indicator and not is_not_submit_button:
					# Проверяем универсальные ключевые слова для кнопок отправки
					has_submit_text_final = False
					if element_text:
						submit_keywords_universal = ['submit', 'send', 'confirm', 'save', 'apply', 'отправить', 'подтвердить', 'сохранить']
						has_submit_text_final = any(keyword in element_text.lower() for keyword in submit_keywords_universal)
					
					# Если элемент не был добавлен ранее и имеет признаки кнопки отправки, добавляем его
					already_added = any(idx == element_index for idx, _, _ in submit_buttons_in_dialog)
					if has_submit_text_final and not already_added and (is_button_like_for_submit or is_clickable):
						submit_text_display = element_text[:50] if element_text else '(кнопка отправки)'
						submit_buttons_in_dialog.append((element_index, submit_text_display, is_visible))
				
				elif has_submit_indicator and not has_cancel_indicator and is_button_like_for_submit:
					if is_clickable:
						if not already_in_submit_buttons_list:
							# Кнопки вне модального окна добавляем только если их нет в списке кнопок отправки
							submit_buttons.append((element_index, element_text[:50], is_visible))
				
				# УНИВЕРСАЛЬНАЯ ЛОГИКА: поиск кнопок отправки на страницах БЕЗ модального окна
				# Кнопки отправки могут находиться прямо на странице
				# ВАЖНО: кнопки отправки могут быть вверху или внизу страницы
				if not has_open_dialog and not is_likely_in_dialog:
					# Проверяем, является ли это кнопкой отправки на странице
					has_submit_text = False
					if element_text:
						submit_keywords = ['отправить', 'откликнуться', 'отклик', 'подтвердить', 'сохранить', 'готово', 'далее', 'send', 'submit', 'apply']
						has_submit_text = any(keyword in element_text.lower() for keyword in submit_keywords)
					# Также проверяем через get_all_children_text для вложенных span
					if not has_submit_text and hasattr(element, 'get_all_children_text'):
						try:
							all_children_text = element.get_all_children_text()
							if all_children_text:
								has_submit_text = any(keyword in all_children_text.lower() for keyword in submit_keywords)
								if has_submit_text and not element_text:
									element_text = all_children_text.strip()
						except:
							pass
					
					# Проверяем атрибуты на признаки кнопки отправки
					has_submit_attrs = False
					if data_qa_value:
						data_qa_str = str(data_qa_value).lower()
						has_submit_attrs = any(kw in data_qa_str for kw in ['submit', 'send', 'apply', 'response', 'confirm', 'save'])
					if aria_label_value:
						aria_str = str(aria_label_value).lower()
						has_submit_attrs = has_submit_attrs or any(kw in aria_str for kw in ['submit', 'send', 'apply', 'response', 'confirm', 'save'])
					
					# КРИТИЧЕСКИ ВАЖНО: кнопка может быть просто button с type="submit"
					# даже без явного текста, но с data-qa содержащим "submit" или "response"
					has_submit_type_or_attr = button_type_value == 'submit' or (data_qa_value and ('submit' in str(data_qa_value).lower() or 'response' in str(data_qa_value).lower()))
					
					# Если это кнопка отправки на странице (не в модальном окне)
					# ИЛИ это кнопка с type="submit" или data-qa="submit"
					# ВАЖНО: для кнопок с явными признаками отправки НЕ требуем is_clickable (может быть disabled)
					# КРИТИЧЕСКИ ВАЖНО: исключаем div/span контейнеры с role="button"
					if not is_fake_button_container and ((has_submit_text or has_submit_attrs) and is_button_like_for_submit and not has_cancel_indicator) or \
					   (not is_fake_button_container and has_submit_type_or_attr and is_button_like_for_submit and not has_cancel_indicator):
						# Проверяем, что элемент еще не добавлен
						already_in_submit_buttons = any(idx == element_index for idx, _, _ in submit_buttons)
						already_in_submit_buttons_dialog = any(idx == element_index for idx, _, _ in submit_buttons_in_dialog)
						if not already_in_submit_buttons and not already_in_submit_buttons_dialog:
							# Проверяем disabled
							is_disabled = False
							if hasattr(element, 'attributes') and element.attributes:
								is_disabled = element.attributes.get('disabled') is not None or element.attributes.get('aria-disabled') == 'true'
							disabled_note = ' (disabled)' if is_disabled else ''
							submit_text_display = element_text[:50] if element_text else f'(кнопка отправки: data-qa={data_qa_value}, aria-label={aria_label_value}, type={button_type_value})'
							submit_buttons.append((element_index, submit_text_display + disabled_note, is_visible))
				
				# УНИВЕРСАЛЬНАЯ ЛОГИКА: если модальное окно НЕ найдено, ищем кнопки отправки на всей странице
				# Если есть textarea, ищем кнопки до и после него
				# Если textarea нет, ищем кнопки по всей странице по признакам (текст, атрибуты, type="submit")
				if not has_open_dialog:
					# Если textarea найден, используем его для ограничения поиска
					# Если textarea НЕ найден, ищем кнопки по всей странице
					has_textarea_context = textarea_in_dialog_index is not None
					
					# Ищем кнопки как ДО, так и ПОСЛЕ textarea (если textarea есть)
					# Кнопки отправки могут быть вверху страницы (до textarea) или внизу (после textarea)
					is_near_textarea = False
					textarea_diff = 0
					if has_textarea_context:
						if element_index > textarea_in_dialog_index:
							# Элемент находится после textarea
							textarea_diff = element_index - textarea_in_dialog_index
							# Ищем кнопки в диапазоне до 2000 элементов после textarea (увеличили для очень длинных форм)
							if textarea_diff < 2000:
								is_near_textarea = True
						elif element_index < textarea_in_dialog_index:
							# Элемент находится ДО textarea - тоже может быть кнопка отправки (вверху страницы)
							textarea_diff = textarea_in_dialog_index - element_index
							# Ищем кнопки в диапазоне до 300 элементов ДО textarea (кнопки вверху обычно ближе)
							if textarea_diff < 300:
								is_near_textarea = True
					else:
						# Если textarea нет, считаем, что кнопка может быть где угодно (is_near_textarea = True для всех)
						# Это позволит искать кнопки по всей странице
						is_near_textarea = True
					
					# ВАЖНО: если textarea нет, ищем кнопки по всей странице по признакам
					# Если textarea есть, ищем только рядом с ним
					if is_near_textarea:
						# Проверяем, является ли это кнопкой отправки
						has_submit_text_for_page = False
						if element_text:
							submit_keywords_page = ['отправить', 'откликнуться', 'отклик', 'подтвердить', 'сохранить', 'готово', 'далее', 'send', 'submit']
							has_submit_text_for_page = any(keyword in element_text.lower() for keyword in submit_keywords_page)
						
						# Проверяем атрибуты
						has_submit_attrs_for_page = False
						if data_qa_value:
							data_qa_str = str(data_qa_value).lower()
							has_submit_attrs_for_page = any(kw in data_qa_str for kw in ['submit', 'отправ', 'send', 'response-submit', 'response'])
						has_submit_type = button_type_value == 'submit'
						
						# Проверяем, что это кнопка (button или a с role="button")
						# ВАЖНО: для кнопок с type="submit" или явными признаками отправки не требуем is_clickable
						# Кнопка может быть disabled, но все равно должна быть показана агенту
						is_button_for_page = tag_is_button or (tag_is_a and has_role_button) or has_role_button
						is_button_clickable = is_button_for_page and is_clickable
						
						# Проверяем, disabled ли кнопка
						is_disabled = False
						if hasattr(element, 'attributes') and element.attributes:
							is_disabled = element.attributes.get('disabled') is not None or element.attributes.get('aria-disabled') == 'true'
						
						# Исключаем кнопки отмены
						has_cancel_for_page = False
						if element_text:
							cancel_keywords_page = ['отменить', 'cancel', 'закрыть', 'close', 'назад', 'back', 'удалить', 'delete']
							has_cancel_for_page = any(keyword in element_text.lower() for keyword in cancel_keywords_page)
						
						# КРИТИЧЕСКИ ВАЖНО: кнопка отправки может быть просто button
						# без явного текста "отправить", но с type="submit" или просто быть первой button после textarea
						# Приоритет: 1) кнопки с текстом/атрибутами отправки, 2) button с type="submit", 3) первая button после textarea
						# ВАЖНО: также проверяем кнопки с текстом отправки - они могут быть после textarea
						is_likely_submit_button = False
						has_submit_text_in_element = False
						if element_text:
							submit_keywords_check = ['отправить', 'откликнуться', 'отклик', 'подтвердить', 'сохранить', 'готово', 'далее', 'send', 'submit', 'apply']
							has_submit_text_in_element = any(keyword in element_text.lower() for keyword in submit_keywords_check)
						if not has_submit_text_in_element and hasattr(element, 'get_all_children_text'):
							try:
								all_children_text = element.get_all_children_text()
								if all_children_text:
									has_submit_text_in_element = any(keyword in all_children_text.lower() for keyword in submit_keywords_check)
							except:
								pass
						
						# Для кнопок с явными признаками отправки не требуем is_clickable (может быть disabled)
						if has_submit_text_for_page or has_submit_attrs_for_page or has_submit_type or has_submit_text_in_element:
							# Явные признаки кнопки отправки - добавляем даже если disabled
							is_likely_submit_button = True
						elif tag_is_button and not has_cancel_for_page:
							# КРИТИЧЕСКИ ВАЖНО: если есть textarea, любая button после него может быть кнопкой отправки
							# Это особенно важно для форм, где кнопка может быть без явного текста "отправить"
							# Если textarea есть, проверяем расстояние от него
							# Если textarea нет, НЕ добавляем все button - нужно иметь явные признаки отправки
							if has_textarea_context:
								# Расширяем диапазон поиска - кнопка может быть далеко внизу страницы (до 2000 элементов)
								# Это важно для длинных форм, где кнопка в самом низу
								if element_index > textarea_in_dialog_index and textarea_diff < 2000:
									# Button после textarea - вероятно кнопка отправки
									is_likely_submit_button = True
								elif element_index < textarea_in_dialog_index and textarea_diff < 300:
									# Button до textarea (в пределах разумного) - может быть кнопка отправки
									is_likely_submit_button = True
							else:
								# Если textarea нет, добавляем только button с явными признаками отправки
								# Или с type="submit", или с data-qa содержащим submit/response
								has_submit_type_or_attr_for_no_textarea = button_type_value == 'submit' or (data_qa_value and ('submit' in str(data_qa_value).lower() or 'response' in str(data_qa_value).lower()))
								if has_submit_type_or_attr_for_no_textarea:
									is_likely_submit_button = True
						
						# Если это вероятная кнопка отправки и не кнопка отмены, добавляем в submit_buttons
						# ВАЖНО: добавляем даже если кнопка disabled - агент должен видеть её
						if is_likely_submit_button and not has_cancel_for_page:
							# Проверяем, что элемент еще не добавлен
							already_in_submit_buttons = any(idx == element_index for idx, _, _ in submit_buttons)
							already_in_submit_buttons_dialog = any(idx == element_index for idx, _, _ in submit_buttons_in_dialog)
							if not already_in_submit_buttons and not already_in_submit_buttons_dialog:
								disabled_note = ' (disabled)' if is_disabled else ''
								submit_text_display = element_text[:50] if element_text else f'(кнопка отправки после textarea: data-qa={data_qa_value}, type={button_type_value})'
								submit_buttons.append((element_index, submit_text_display + disabled_note, is_visible))
						else:
							pass
					
					# ДОПОЛНИТЕЛЬНО: кнопка может быть button с role="button" или просто button
					# даже если она не прошла все проверки выше - добавляем её как потенциальную кнопку отправки
					# КРИТИЧЕСКИ ВАЖНО для форм: любая button после textarea может быть кнопкой отправки
					if not has_open_dialog:
						# Расширяем поиск: ищем кнопки в большем диапазоне от textarea (если textarea есть)
						# Если textarea нет, добавляем только button с явными признаками отправки
						should_check_extended = False
						if textarea_in_dialog_index is not None:
							# КРИТИЧЕСКИ ВАЖНО: если есть textarea, любая button ПОСЛЕ него может быть кнопкой отправки
							# Убираем ограничение по расстоянию для кнопок после textarea - они могут быть в самом низу страницы
							if element_index > textarea_in_dialog_index:
								# Кнопка после textarea - включаем ВСЕ такие кнопки, независимо от расстояния
								is_near_textarea_extended = True
							else:
								# Кнопка до textarea - только близкие (в пределах 300 элементов)
								textarea_diff_extended = textarea_in_dialog_index - element_index
								is_near_textarea_extended = textarea_diff_extended < 300
							should_check_extended = is_near_textarea_extended
						else:
							# Если textarea нет, добавляем только button с явными признаками отправки
							# Проверяем type="submit" или data-qa с submit/response
							has_submit_type_or_attr_extended = button_type_value == 'submit' or (data_qa_value and ('submit' in str(data_qa_value).lower() or 'response' in str(data_qa_value).lower()))
							if has_submit_type_or_attr_extended:
								should_check_extended = True
						
						if should_check_extended:
							# Если это button с role="button" или просто button, это может быть кнопка отправки
							# НЕ требуем is_clickable - кнопка может быть disabled, но все равно должна быть показана
							# НО добавляем только если это действительно похоже на кнопку отправки (не кнопка отмены)
							# КРИТИЧЕСКИ ВАЖНО: исключаем div/span контейнеры с role="button"
							if not is_fake_button_container and (tag_is_button or (tag_is_a and has_role_button)) and not has_cancel_for_page:
								# Проверяем, что элемент еще не добавлен
								already_in_submit_buttons = any(idx == element_index for idx, _, _ in submit_buttons)
								already_in_submit_buttons_dialog = any(idx == element_index for idx, _, _ in submit_buttons_in_dialog)
								if not already_in_submit_buttons and not already_in_submit_buttons_dialog:
									# Проверяем disabled
									is_disabled_extended = False
									if hasattr(element, 'attributes') and element.attributes:
										is_disabled_extended = element.attributes.get('disabled') is not None or element.attributes.get('aria-disabled') == 'true'
									disabled_note = ' (disabled)' if is_disabled_extended else ''
									# Получаем текст кнопки для отображения
									button_text_for_display = element_text[:50] if element_text else ''
									if not button_text_for_display:
										button_text_for_display = f'button (data-qa={data_qa_value}, type={button_type_value})' if data_qa_value or button_type_value else 'button'
									submit_text_display = button_text_for_display
									submit_buttons.append((element_index, submit_text_display + disabled_note, is_visible))
			
			# Агент видит все элементы в browser_state с их текстом, атрибутами и индексами
			# Агент сам определяет, какие элементы использовать, анализируя содержимое страницы
			
			# Добавляем рекомендации для кнопок отправки
			# ПРИОРИТЕТ: если есть модальное окно, сначала показываем кнопки внутри него
			# Фильтруем submit_buttons_in_dialog: исключаем контейнер модального окна и кнопки для других элементов списка
			submit_buttons_in_dialog_filtered = []
			if submit_buttons_in_dialog:
				for btn_index, btn_text, btn_visible in submit_buttons_in_dialog:
					# Исключаем контейнер модального окна (длинный текст без button/a - это контейнер)
					if btn_index == dialog_element_index or (btn_text and len(btn_text) > 200):
						continue
					
					# КРИТИЧЕСКИ ВАЖНО: исключаем неправильные элементы - input type=hidden и div элементы, которые не являются button/a
					should_skip = False
					if btn_index in selector_map:
						elem = selector_map[btn_index]
						tag = getattr(elem, 'tag_name', '').lower() if hasattr(elem, 'tag_name') else ''
						button_type = None
						if hasattr(elem, 'attributes') and elem.attributes:
							button_type = elem.attributes.get('type')
						
						# Исключаем input type=hidden - это скрытые поля формы, а не кнопки
						if tag == 'input' and button_type == 'hidden':
							should_skip = True
						# Исключаем div элементы, которые содержат текст сопроводительного письма (это контейнеры, а не кнопки)
						elif tag == 'div' and btn_text and ('уважаемые коллеги' in btn_text.lower() or 'сопроводительн' in btn_text.lower() or len(btn_text) > 50):
							should_skip = True
					
					# Исключаем кнопки для других элементов из списка рекомендаций (не кнопки отправки формы)
					if not should_skip and btn_index in selector_map:
						elem = selector_map[btn_index]
						if hasattr(elem, 'attributes') and elem.attributes:
							data_qa = elem.attributes.get('data-qa')
							if data_qa:
								data_qa_str = str(data_qa).lower()
								# КРИТИЧЕСКИ ВАЖНО: если элемент находится внутри модального окна И имеет текст отправки формы,
								# это НЕ кнопка рекомендаций, а кнопка отправки формы в модальном окне!
								# Проверяем, находится ли элемент в диапазоне модального окна
								is_in_dialog_range = False
								if has_open_dialog and dialog_element_index:
									# Элемент находится в диапазоне модального окна (до 1000 элементов от него)
									if abs(btn_index - dialog_element_index) < 1000:
										is_in_dialog_range = True
								
								# Проверяем наличие текста отправки в элементе
								has_submit_text_in_button = False
								submit_keywords_check = ['отправить', 'откликнуться', 'отклик', 'подтвердить', 'сохранить', 'готово', 'далее', 'send', 'submit', 'apply']
								if btn_text and any(keyword in btn_text.lower() for keyword in submit_keywords_check):
									has_submit_text_in_button = True
								elif btn_index in selector_map:
									elem_check = selector_map[btn_index]
									if hasattr(elem_check, 'get_all_children_text'):
										try:
											all_children_text = elem_check.get_all_children_text()
											if all_children_text and any(keyword in all_children_text.lower() for keyword in submit_keywords_check):
												has_submit_text_in_button = True
										except:
											pass
								
								# Если элемент в модальном окне И имеет текст отправки - это кнопка отправки, НЕ рекомендация
								if is_in_dialog_range and has_submit_text_in_button:
									# НЕ пропускаем этот элемент - это кнопка отправки формы
									pass
								# Кнопки с этими data-qa относятся к рекомендациям элементов списка, а не к форме отправки
								# УНИВЕРСАЛЬНАЯ ЛОГИКА: исключаем элементы с признаками навигации/ссылок
								elif any(kw in data_qa_str for kw in ['link', 'nav', 'menu', 'breadcrumb']):
									should_skip = True
								elif 'submit' in data_qa_str:
									# Эта кнопка имеет наивысший приоритет - добавляем её первой
									submit_buttons_in_dialog_filtered.insert(0, (btn_index, btn_text or '(кнопка отправки)', btn_visible))
									should_skip = True  # Уже добавлена, пропускаем дальнейшую обработку
					if not should_skip:
						submit_buttons_in_dialog_filtered.append((btn_index, btn_text, btn_visible))
			else:
				# Расчет координат и рекомендации выполняются в блоке после фильтрации,
				# если submit_buttons_in_dialog_filtered пуст или не содержит реальных кнопок
				pass
			
			# КРИТИЧЕСКИ ВАЖНО: проверяем, есть ли в отфильтрованном списке реальные button/a элементы
			# Если нет - пытаемся рассчитать координаты на основе textarea
			has_real_button_elements = False
			if submit_buttons_in_dialog_filtered:
				for btn_idx, _, _ in submit_buttons_in_dialog_filtered:
					if btn_idx in selector_map:
						elem = selector_map[btn_idx]
						tag = getattr(elem, 'tag_name', '').lower() if hasattr(elem, 'tag_name') else ''
						button_type = None
						if hasattr(elem, 'attributes') and elem.attributes:
							button_type = elem.attributes.get('type')
						# Проверяем, что это реальная кнопка (button/a, не hidden input, не div с текстом)
						if (tag == 'button' or (tag == 'a' and hasattr(elem, 'attributes') and elem.attributes and elem.attributes.get('role') == 'button')) and button_type != 'hidden':
							has_real_button_elements = True
							break
			
			# Если нет реальных button элементов, пытаемся рассчитать координаты на основе textarea
			calculated_coordinates = None
			# Если кнопка не найдена по индексу, агент может использовать координатный клик, анализируя скриншот
			# Агент видит все элементы в browser_state, включая модальные окна и textarea
			# Агент сам определяет структуру формы и нужные элементы, анализируя текст и контекст
			
			# БАЛАНС: информативные рекомендации, которые помогают, но не ограничивают анализ
			# Агент должен сам анализировать страницу, но ему нужна помощь с сопоставлением визуальных элементов и индексов
			# Рекомендации информативные, не директивные - агент сам решает, что делать
			if submit_buttons_in_dialog_filtered:
				recommendations_text += '\n<page_analysis_hints>\n'
				recommendations_text += 'На странице обнаружены потенциальные кнопки отправки формы в модальном окне:\n'
				for btn_idx, btn_text, btn_visible in submit_buttons_in_dialog_filtered[:5]:  # Показываем только первые 5
					visibility_note = ' (скрыта)' if not btn_visible else ''
					recommendations_text += f'  - Элемент [{btn_idx}]: {btn_text}{visibility_note}\n'
				recommendations_text += 'Проанализируйте страницу и определите, какая кнопка подходит для вашей задачи.\n'
				recommendations_text += '</page_analysis_hints>\n'
			elif submit_buttons:
				recommendations_text += '\n<page_analysis_hints>\n'
				recommendations_text += 'На странице обнаружены потенциальные кнопки отправки формы:\n'
				for btn_idx, btn_text, btn_visible in submit_buttons[:5]:  # Показываем только первые 5
					visibility_note = ' (скрыта)' if not btn_visible else ''
					recommendations_text += f'  - Элемент [{btn_idx}]: {btn_text}{visibility_note}\n'
				recommendations_text += 'Проанализируйте страницу и определите, какая кнопка подходит для вашей задачи.\n'
				recommendations_text += '</page_analysis_hints>\n'
		
		browser_state = f"""{stats_text}{current_tab_text}
Available tabs:
{tabs_text}
{page_info_text}
{recent_events_text}{closed_popups_text}{pdf_message}{email_context_text}{recommendations_text}Interactive elements{truncated_text}:
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
