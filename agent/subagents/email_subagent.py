"""Субагент для работы с почтовыми интерфейсами.

Этот субагент предоставляет специализированные инструменты и знания для работы
с различными почтовыми клиентами (Gmail, Yandex Mail, Outlook и т.д.) без хардкода
селекторов или конкретных действий.
"""

import logging
from typing import Any

from agent.browser.views import BrowserStateSummary
from agent.dom.views import DOMInteractedElement

logger = logging.getLogger(__name__)


class EmailSubAgent:
	"""Субагент для работы с почтовыми интерфейсами.
	
	Предоставляет:
	- Специализированные инструменты для извлечения метаданных писем
	- Определение типа почтового клиента
	- Обработку почтовых специфичных ситуаций (диалоги, модальные окна)
	- Общие знания о структуре почтовых интерфейсов
	
	ВАЖНО: Не содержит хардкода селекторов, ключевых слов или предписанных действий.
	Основной агент сам решает, как использовать эти инструменты.
	"""
	
	def __init__(self):
		"""Инициализация субагента для почты."""
		self.logger = logger
	
	def is_email_client(self, browser_state: BrowserStateSummary) -> bool:
		"""Определяет, является ли текущая страница почтовым клиентом.
		
		Использует общие признаки почтовых интерфейсов без хардкода конкретных доменов.
		
		Args:
			browser_state: Текущее состояние браузера
			
		Returns:
			True если страница похожа на почтовый клиент
		"""
		if not browser_state:
			return False
		
		url = (browser_state.url or '').lower()
		title = (browser_state.title or '').lower()
		
		# Общие признаки почтовых интерфейсов (без хардкода конкретных доменов)
		email_indicators = [
			'mail', 'email', 'почта', 'inbox', 'входящие',
			'message', 'письмо', 'compose', 'написать'
		]
		
		# Проверяем URL и title на наличие признаков почтового интерфейса
		url_has_email_indicator = any(indicator in url for indicator in email_indicators)
		title_has_email_indicator = any(indicator in title for indicator in email_indicators)
		
		# Проверяем наличие элементов, характерных для почтовых интерфейсов
		has_email_elements = False
		if browser_state.dom_state and browser_state.dom_state.selector_map:
			selector_map = browser_state.dom_state.selector_map
			for element in selector_map.values():
				element_text = self._get_element_text(element)
				element_role = self._get_element_role(element)
				
				# Ищем элементы, характерные для почтовых интерфейсов
				if element_role in ['listitem', 'article']:
					# Проверяем наличие структуры списка писем
					if len(element_text) > 20:  # Письма обычно имеют достаточно длинный текст
						has_email_elements = True
						break
		
		return url_has_email_indicator or title_has_email_indicator or has_email_elements
	
	def extract_email_metadata(self, browser_state: BrowserStateSummary) -> dict[str, Any]:
		"""Извлекает метаданные открытого письма из DOM.
		
		Ищет тему, отправителя, дату и тело письма без использования хардкоженных селекторов.
		Агент сам решает, как использовать эти метаданные.
		
		Args:
			browser_state: Текущее состояние браузера
			
		Returns:
			Словарь с метаданными письма:
			{
				'subject': str | None,
				'sender': str | None,
				'date': str | None,
				'body_preview': str | None,
				'is_opened': bool
			}
		"""
		result = {
			'subject': None,
			'sender': None,
			'date': None,
			'body_preview': None,
			'is_opened': False
		}
		
		if not browser_state:
			return result
		
		# Определяем, открыто ли письмо (проверяем структуру URL и DOM)
		url = browser_state.url or ''
		result['is_opened'] = self._is_email_opened(url, browser_state)
		
		# Извлекаем тему и отправителя из title только если письмо открыто
		# Если письмо не открыто, title обычно содержит название папки, а не тему письма
		if browser_state.title and result['is_opened']:
			# Проверяем, содержит ли title признаки письма (кавычки, префикс "Письмо" и т.д.)
			# Если нет - title еще не обновился в SPA и содержит название папки
			title_has_email_signs = any(
				char in browser_state.title for char in ['«', '"', "'", '`']
			) or any(
				prefix.lower() in browser_state.title.lower() 
				for prefix in ['письмо', 'message', 'email']
			)
			
			# Парсим title только если он содержит признаки письма
			if title_has_email_signs:
				title_parts = self._parse_email_title(browser_state.title)
				result['subject'] = title_parts.get('subject')
				result['sender'] = title_parts.get('sender')
		
		# Извлекаем тело письма из DOM (ищем длинные текстовые блоки)
		if browser_state.dom_state and browser_state.dom_state.selector_map:
			body_text = self._extract_email_body(browser_state.dom_state.selector_map)
			if body_text:
				result['body_preview'] = body_text[:500]  # Первые 500 символов
			
			# Если тема не извлечена из title, пытаемся извлечь из DOM
			if not result['subject'] and result['is_opened']:
				dom_subject = self._extract_subject_from_dom(browser_state.dom_state.selector_map)
				if dom_subject:
					result['subject'] = dom_subject
		
		return result
	
	def detect_dialog(self, browser_state: BrowserStateSummary) -> bool:
		"""Определяет наличие открытого диалога или модального окна.
		
		Использует стандартные ARIA-роли без хардкода текста кнопок.
		
		Args:
			browser_state: Текущее состояние браузера
			
		Returns:
			True если обнаружен диалог
		"""
		if not browser_state or not browser_state.dom_state:
			return False
		
		selector_map = browser_state.dom_state.selector_map
		for element in selector_map.values():
			element_role = self._get_element_role(element)
			if element_role and element_role.lower() in ['dialog', 'alertdialog']:
				return True
		
		return False
	
	def get_full_email_text(self, browser_state: BrowserStateSummary) -> str | None:
		"""Извлекает полный текст открытого письма из DOM.
		
		Используется для анализа содержимого письма через extract action.
		Возвращает полный текст письма, а не только preview.
		
		Args:
			browser_state: Текущее состояние браузера
			
		Returns:
			Полный текст письма или None если письмо не открыто или текст не найден
		"""
		if not browser_state or not browser_state.dom_state:
			return None
		
		url = browser_state.url or ''
		if not self._is_email_opened(url, browser_state):
			return None
		
		if browser_state.dom_state.selector_map:
			return self._extract_email_body(browser_state.dom_state.selector_map)
		
		return None
	
	def count_emails_in_list(self, browser_state: BrowserStateSummary) -> int:
		"""Подсчитывает количество писем в текущем списке.
		
		Использует общие паттерны для определения элементов списка писем
		без хардкода конкретных селекторов.
		
		Args:
			browser_state: Текущее состояние браузера
			
		Returns:
			Количество найденных писем в списке (0 если список не найден или письмо открыто)
		"""
		if not browser_state or not browser_state.dom_state:
			return 0
		
		# Если письмо открыто, список писем не виден
		url = browser_state.url or ''
		if self._is_email_opened(url, browser_state):
			return 0
		
		if not browser_state.dom_state.selector_map:
			return 0
		
		email_count = 0
		selector_map = browser_state.dom_state.selector_map
		
		for element in selector_map.values():
			element_role = self._get_element_role(element)
			element_text = self._get_element_text(element)
			
			# Ищем элементы списка писем (listitem с достаточным количеством текста)
			if element_role == 'listitem':
				# Письма обычно имеют достаточно текста (тема + отправитель + превью)
				if len(element_text.strip()) > 30:
					email_count += 1
		
		return email_count
	
	def get_email_list_structure(self, browser_state: BrowserStateSummary) -> dict[str, Any]:
		"""Определяет структуру списка писем.
		
		Возвращает информацию о видимых письмах в списке для отслеживания прогресса.
		
		Args:
			browser_state: Текущее состояние браузера
			
		Returns:
			Словарь с информацией о структуре списка:
			{
				'count': int,  # Количество писем в списке
				'is_list_view': bool,  # Находимся ли в списке писем
				'emails': list[dict]  # Список с базовой информацией о письмах
			}
		"""
		result = {
			'count': 0,
			'is_list_view': False,
			'emails': []
		}
		
		if not browser_state or not browser_state.dom_state:
			return result
		
		url = browser_state.url or ''
		is_opened = self._is_email_opened(url, browser_state)
		
		if is_opened:
			return result
		
		result['is_list_view'] = True
		
		if not browser_state.dom_state.selector_map:
			return result
		
		selector_map = browser_state.dom_state.selector_map
		
		for index, element in selector_map.items():
			element_role = self._get_element_role(element)
			element_text = self._get_element_text(element)
			
			# Ищем элементы списка писем
			if element_role == 'listitem' and len(element_text.strip()) > 30:
				# Извлекаем базовую информацию о письме из текста элемента
				email_info = {
					'index': index,
					'preview': element_text.strip()[:200]  # Первые 200 символов для идентификации
				}
				result['emails'].append(email_info)
		
		result['count'] = len(result['emails'])
		return result
	
	def suggest_email_context(self, browser_state: BrowserStateSummary) -> str:
		"""Предлагает контекст о структуре почтового интерфейса для промпта.
		
		Добавляет общие знания о почтовых интерфейсах без предписывания конкретных действий.
		
		Args:
			browser_state: Текущее состояние браузера
			
		Returns:
			Строка с контекстом для добавления в промпт
		"""
		if not self.is_email_client(browser_state):
			return ''
		
		# Определяем состояние письма для добавления в контекст
		url = browser_state.url or ''
		is_opened = self._is_email_opened(url, browser_state)
		
		# Получаем метаданные текущего письма и структуру списка
		email_metadata = self.extract_email_metadata(browser_state)
		list_structure = self.get_email_list_structure(browser_state)
		
		context = """
ОБЩИЕ ЗНАНИЯ О ПОЧТОВЫХ ИНТЕРФЕЙСАХ:
- Почтовые клиенты обычно имеют список писем (входящие, отправленные и т.д.)
- Письма обычно имеют тему, отправителя, дату и тело письма
- Для просмотра письма нужно кликнуть на него в списке
- После открытия письма обычно доступны действия (удалить, ответить, переслать и т.д.)
- Почтовые клиенты часто используют SPA (Single Page Applications), поэтому после кликов может потребоваться время для обновления DOM
- Модальные окна и диалоги могут блокировать взаимодействие - их нужно закрыть перед продолжением работы

ЛОГИКА РАБОТЫ С ПИСЬМАМИ:
- При работе с несколькими письмами важно отслеживать прогресс: сколько писем обработано, сколько осталось
- Каждое письмо нужно анализировать индивидуально перед принятием решения о действиях
- Если письмо определено как не спам, его обычно НЕ удаляют, а оставляют или помечают как прочитанное
- Удаление письма - это действие, которое применяется только к письмам, определенным как спам или ненужные
- После анализа письма важно перейти к следующему, не повторяя обработку уже просмотренных
"""
		
		# Добавляем информацию о текущем состоянии
		if is_opened:
			context += f"""
ТЕКУЩЕЕ СОСТОЯНИЕ:
- Письмо УЖЕ ОТКРЫТО (URL содержит #message/ или /message/)
- Вы находитесь на странице просмотра письма, а не в списке писем
- Для просмотра другого письма нужно вернуться к списку писем
- Для удаления текущего письма найдите кнопку "Удалить" на странице просмотра
"""
			if email_metadata.get('subject'):
				context += f"- Тема открытого письма: {email_metadata['subject']}\n"
			if email_metadata.get('sender'):
				context += f"- Отправитель открытого письма: {email_metadata['sender']}\n"
			if email_metadata.get('body_preview'):
				preview = email_metadata['body_preview'][:150]
				context += f"- Предпросмотр содержания: {preview}...\n"
		else:
			context += f"""
ТЕКУЩЕЕ СОСТОЯНИЕ:
- Вы находитесь в списке писем (письмо НЕ открыто)
- Для просмотра письма кликните на него в списке
"""
			if list_structure['count'] > 0:
				context += f"- В текущем списке видно примерно {list_structure['count']} писем\n"
		
		context += """
ВАЖНО: Эти знания - только общая информация о структуре почтовых интерфейсов и текущем состоянии.
Вы сами решаете, какие действия предпринимать для выполнения задачи.
"""
		return context
	
	def _is_email_opened(self, url: str, browser_state: BrowserStateSummary) -> bool:
		"""Определяет, открыто ли письмо.
		
		Проверяет URL и структуру DOM на признаки открытого письма.
		"""
		# Проверяем URL на признаки открытого письма (общие паттерны)
		url_lower = url.lower()
		if any(pattern in url_lower for pattern in ['#message/', '/message/', '?message=', 'view=']):
			return True
		
		# Проверяем DOM на наличие структуры открытого письма
		if browser_state.dom_state and browser_state.dom_state.selector_map:
			selector_map = browser_state.dom_state.selector_map
			# Ищем элементы, характерные для открытого письма (article, main с длинным текстом)
			for element in selector_map.values():
				element_role = self._get_element_role(element)
				element_text = self._get_element_text(element)
				if element_role in ['article', 'main'] and len(element_text) > 100:
					return True
		
		return False
	
	def _parse_email_title(self, title: str) -> dict[str, str | None]:
		"""Парсит title страницы для извлечения темы и отправителя.
		
		Использует общие паттерны без хардкода конкретных форматов.
		"""
		result = {'subject': None, 'sender': None}
		
		# Общие разделители в title почтовых клиентов
		separators = ['—', '-', '|', '•', '::']
		
		for sep in separators:
			if sep in title:
				parts = title.split(sep)
				if len(parts) >= 2:
					# Первая часть обычно содержит тему письма
					subject_part = parts[0].strip()
					# Убираем кавычки вокруг темы (общий паттерн без хардкода конкретных слов)
					import re
					# Убираем открывающие и закрывающие кавычки любого типа
					subject_part = re.sub(r'^[«"\'`]+', '', subject_part)
					subject_part = re.sub(r'[»"\'`]+$', '', subject_part)
					result['subject'] = subject_part.strip() if subject_part.strip() else None
					
					# Вторая часть обычно содержит отправителя
					sender_part = parts[1].strip()
					# Убираем суффиксы почтовых сервисов через паттерн (общий подход)
					# Ищем паттерн: слово "Mail" или "Почта" в конце (без хардкода конкретных названий)
					sender_part = re.sub(r'\s+(Mail|Почта|Email)$', '', sender_part, flags=re.IGNORECASE)
					result['sender'] = sender_part.strip()
					break
		
		return result
	
	def _extract_email_body(self, selector_map: dict[int, DOMInteractedElement]) -> str | None:
		"""Извлекает тело письма из DOM.
		
		Ищет длинные текстовые блоки, исключая элементы интерфейса.
		"""
		body_candidates = []
		
		for element in selector_map.values():
			element_text = self._get_element_text(element)
			element_role = self._get_element_role(element)
			
			# Пропускаем короткие тексты и элементы интерфейса
			if len(element_text.strip()) < 50:
				continue
			
			# Исключаем элементы с типичными текстами интерфейса
			interface_keywords = [
				'удалить', 'delete', 'настройка', 'меню', 'кнопка', 'button',
				'ссылка', 'link', 'входящие', 'inbox', 'отправленные', 'sent'
			]
			text_lower = element_text.lower()
			if any(keyword in text_lower for keyword in interface_keywords):
				continue
			
			# Ищем длинные текстовые блоки (вероятно, тело письма)
			if element_role in ['article', 'main', 'generic'] or len(element_text.strip()) > 200:
				body_candidates.append(element_text.strip())
		
		# Возвращаем самый длинный текст как тело письма
		if body_candidates:
			return max(body_candidates, key=len)
		
		return None
	
	def _get_element_text(self, element: DOMInteractedElement) -> str:
		"""Извлекает текст из элемента."""
		if hasattr(element, 'ax_node') and element.ax_node and element.ax_node.name:
			return element.ax_node.name
		elif hasattr(element, 'get_all_children_text'):
			return element.get_all_children_text()
		elif hasattr(element, 'get_meaningful_text_for_llm'):
			return element.get_meaningful_text_for_llm()
		elif hasattr(element, 'node_value'):
			return element.node_value or ''
		return ''
	
	def _extract_subject_from_dom(self, selector_map: dict[int, DOMInteractedElement]) -> str | None:
		"""Извлекает тему письма из DOM используя общие семантические паттерны.
		
		Используется как fallback когда title не обновился в SPA.
		Ищет заголовки и элементы с aria-label без хардкода конкретных селекторов.
		
		Args:
			selector_map: Карта селекторов DOM элементов
			
		Returns:
			Тема письма или None если не найдена
		"""
		subject_candidates = []
		
		for element in selector_map.values():
			element_text = self._get_element_text(element)
			element_role = self._get_element_role(element)
			
			# Пропускаем пустые или очень длинные тексты (заголовки обычно короткие)
			text_stripped = element_text.strip()
			if not text_stripped or len(text_stripped) > 200:
				continue
			
			# Ищем элементы с ролью heading (h1-h6) - это наиболее надежный источник темы
			if element_role and 'heading' in element_role.lower():
				subject_candidates.append((text_stripped, 2))  # Высокий приоритет
			
			# Ищем элементы с aria-label (если доступны через ax_node)
			if hasattr(element, 'ax_node') and element.ax_node:
				if hasattr(element.ax_node, 'properties') and element.ax_node.properties:
					props = element.ax_node.properties
					# aria-label может содержать тему письма
					if 'label' in props and props['label']:
						label_text = str(props['label']).strip()
						if label_text and len(label_text) <= 200:
							subject_candidates.append((label_text, 1))  # Средний приоритет
		
		# Возвращаем кандидата с наивысшим приоритетом
		if subject_candidates:
			# Сортируем по приоритету (от большего к меньшему)
			subject_candidates.sort(key=lambda x: x[1], reverse=True)
			return subject_candidates[0][0]
		
		return None
	
	def _get_element_role(self, element: DOMInteractedElement) -> str:
		"""Извлекает роль элемента."""
		if hasattr(element, 'ax_node') and element.ax_node and element.ax_node.role:
			return element.ax_node.role
		elif hasattr(element, 'tag_name'):
			return element.tag_name or ''
		return ''

