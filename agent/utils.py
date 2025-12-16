import asyncio
import logging
import os
import platform
import re
import signal
import time
from collections.abc import Callable, Coroutine
from fnmatch import fnmatch
from functools import cache, wraps
from pathlib import Path
from sys import stderr
from typing import Any, ParamSpec, TypeVar
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv

load_dotenv()

# Pre-compiled regex for URL detection - used in URL shortening
URL_PATTERN = re.compile(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+|[^\s<>"\']+\.[a-z]{2,}(?:/[^\s<>"\']*)?', re.IGNORECASE)


logger = logging.getLogger(__name__)


def extract_json_from_text(text: str) -> str:
	"""
	Извлекает первый валидный JSON-объект из текста, даже если он обёрнут в markdown код или содержит лишний текст.
	Аналогично функции extractJSONObject из Rusik Original.
	Также удаляет комментарии (// и /* */) из JSON перед парсингом.
	
	Args:
		text: Текст, который может содержать JSON-объект
		
	Returns:
		Извлечённый JSON-объект как строка
		
	Raises:
		ValueError: Если JSON-объект не найден
	"""
	import re
	import json as json_module
	
	# Сначала пытаемся найти JSON в markdown блоках кода
	markdown_json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
	markdown_match = re.search(markdown_json_pattern, text, re.DOTALL)
	if markdown_match:
		json_str = markdown_match.group(1).strip()
	else:
		# Если не нашли в markdown, ищем обычный JSON объект
		depth = 0        # Глубина вложенности фигурных скобок
		start = -1       # Индекс начала JSON-объекта
		in_string = False  # Находимся ли внутри строки
		escape = False   # Обработка экранирования
		
		for i, ch in enumerate(text):
			if escape:
				escape = False
				continue
				
			if ch == '\\' and in_string:
				escape = True
				continue
			elif ch == '"':
				in_string = not in_string
			elif ch == '{' and not in_string:
				if depth == 0:
					start = i
				depth += 1
			elif ch == '}' and not in_string and depth > 0:
				depth -= 1
				if depth == 0 and start != -1:
					json_str = text[start:i+1]
					break
		else:
			raise ValueError("JSON object not found in text")
	
	# Удаляем комментарии из JSON используя более простой и надежный подход
	# Проходим по символам и удаляем комментарии, учитывая контекст строк
	result = []
	i = 0
	in_string = False
	escape = False
	
	while i < len(json_str):
		ch = json_str[i]
		
		if escape:
			result.append(ch)
			escape = False
			i += 1
			continue
			
		if ch == '\\' and in_string:
			result.append(ch)
			escape = True
			i += 1
			continue
		elif ch == '"':
			in_string = not in_string
			result.append(ch)
			i += 1
		elif ch == '/' and i + 1 < len(json_str) and not in_string:
			next_ch = json_str[i + 1]
			if next_ch == '/':
				# Однострочный комментарий - пропускаем до конца строки или конца файла
				# Важно: пропускаем все символы до \n включительно
				while i < len(json_str):
					if json_str[i] == '\n':
						i += 1  # Пропускаем \n
						break
					i += 1
				continue
			elif next_ch == '*':
				# Многострочный комментарий - пропускаем до */
				i += 2  # Пропускаем /*
				while i < len(json_str) - 1:
					if json_str[i] == '*' and json_str[i + 1] == '/':
						i += 2  # Пропускаем */
						break
					i += 1
				else:
					# Если не нашли закрывающий */, это ошибка, но продолжаем
					break
				continue
			else:
				# Это не комментарий, просто символ /
				result.append(ch)
				i += 1
		else:
			result.append(ch)
			i += 1
	
	json_str = ''.join(result)
	
	# Пытаемся исправить распространённые ошибки JSON
	# Убираем лишние запятые перед закрывающими скобками
	json_str = re.sub(r',\s*}', '}', json_str)
	json_str = re.sub(r',\s*]', ']', json_str)
	
	# Убираем множественные пробелы между объектами в массивах
	# Это исправляет случаи когда после удаления комментариев остаются лишние пробелы
	# Например: `},      {` должно стать `},{`
	json_str = re.sub(r'\}\s+,\s+\{', '},{', json_str)  # Убираем пробелы между объектами в массиве
	json_str = re.sub(r'\}\s+,\s+', '},', json_str)  # Убираем пробелы после запятой перед закрывающей скобкой
	
	# Проверяем валидность JSON, пытаясь его распарсить и переформатировать
	# Это нормализует все пробелы и форматирование
	try:
		# Парсим JSON - это автоматически исправит проблемы с форматированием
		parsed = json_module.loads(json_str)
		# Возвращаем нормализованный JSON (компактный формат без лишних пробелов)
		return json_module.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
	except json_module.JSONDecodeError as e:
		# Если не удалось распарсить, пробуем еще раз исправить
		# Убираем trailing commas более агрессивно
		json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
		# Убираем все пробелы вокруг запятых в массивах (но не внутри строк)
		# Это исправляет случаи когда после удаления комментариев остаются пробелы
		json_str = re.sub(r'\}\s*,\s*\{', '},{', json_str)  # Убираем пробелы между объектами
		json_str = re.sub(r'\]\s*,\s*\[', '],[', json_str)  # Убираем пробелы между массивами
		# Пробуем еще раз распарсить
		try:
			parsed = json_module.loads(json_str)
			return json_module.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
		except json_module.JSONDecodeError:
			# Если все еще не удалось, возвращаем как есть и пусть Pydantic разберется
			# Но логируем ошибку для отладки
			raise ValueError(f"Failed to parse JSON after comment removal: {e}. JSON string (first 500 chars): {json_str[:500]}")


# Import error types - these may need to be adjusted based on actual import paths
try:
	from openai import BadRequestError as OpenAIBadRequestError
except ImportError:
	OpenAIBadRequestError = None

try:
	from groq import BadRequestError as GroqBadRequestError  # type: ignore[import-not-found]
except ImportError:
	GroqBadRequestError = None


# Global flag to prevent duplicate exit messages
_exiting = False

# Define generic type variables for return type and parameters
R = TypeVar('R')
T = TypeVar('T')
P = ParamSpec('P')


class SignalHandler:
	"""
	A modular and reusable signal handling system for managing SIGINT (Ctrl+C), SIGTERM,
	and other signals in asyncio applications.

	This class provides:
	- Configurable signal handling for SIGINT and SIGTERM
	- Support for custom pause/resume callbacks
	- Management of event loop state across signals
	- Standardized handling of first and second Ctrl+C presses
	- Cross-platform compatibility (with simplified behavior on Windows)
	"""

	def __init__(
		self,
		loop: asyncio.AbstractEventLoop | None = None,
		pause_callback: Callable[[], None] | None = None,
		resume_callback: Callable[[], None] | None = None,
		custom_exit_callback: Callable[[], None] | None = None,
		exit_on_second_int: bool = True,
		interruptible_task_patterns: list[str] | None = None,
	):
		"""
		Initialize the signal handler.

		Args:
			loop: The asyncio event loop to use. Defaults to current event loop.
			pause_callback: Function to call when system is paused (first Ctrl+C)
			resume_callback: Function to call when system is resumed
			custom_exit_callback: Function to call on exit (second Ctrl+C or SIGTERM)
			exit_on_second_int: Whether to exit on second SIGINT (Ctrl+C)
			interruptible_task_patterns: List of patterns to match task names that should be
										 canceled on first Ctrl+C (default: ['step', 'multi_act', 'get_next_action'])
		"""
		self.loop = loop or asyncio.get_event_loop()
		self.pause_callback = pause_callback
		self.resume_callback = resume_callback
		self.custom_exit_callback = custom_exit_callback
		self.exit_on_second_int = exit_on_second_int
		self.interruptible_task_patterns = interruptible_task_patterns or ['step', 'multi_act', 'get_next_action']
		self.is_windows = platform.system() == 'Windows'

		# Initialize loop state attributes
		self._initialize_loop_state()

		# Store original signal handlers to restore them later if needed
		self.original_sigint_handler = None
		self.original_sigterm_handler = None

	def _initialize_loop_state(self) -> None:
		"""Initialize loop state attributes used for signal handling."""
		setattr(self.loop, 'ctrl_c_pressed', False)
		setattr(self.loop, 'waiting_for_input', False)

	def register(self) -> None:
		"""Register signal handlers for SIGINT and SIGTERM."""
		try:
			if self.is_windows:
				# On Windows, use simple signal handling with immediate exit on Ctrl+C
				def windows_handler(sig, frame):
					print('\n\n🛑 Got Ctrl+C. Exiting immediately on Windows...\n', file=stderr)
					# Run the custom exit callback if provided
					if self.custom_exit_callback:
						self.custom_exit_callback()
					os._exit(0)

				self.original_sigint_handler = signal.signal(signal.SIGINT, windows_handler)
			else:
				# On Unix-like systems, use asyncio's signal handling for smoother experience
				self.original_sigint_handler = self.loop.add_signal_handler(signal.SIGINT, lambda: self.sigint_handler())
				self.original_sigterm_handler = self.loop.add_signal_handler(signal.SIGTERM, lambda: self.sigterm_handler())

		except Exception:
			# there are situations where signal handlers are not supported, e.g.
			# - when running in a thread other than the main thread
			# - some operating systems
			# - inside jupyter notebooks
			pass

	def unregister(self) -> None:
		"""Unregister signal handlers and restore original handlers if possible."""
		try:
			if self.is_windows:
				# On Windows, just restore the original SIGINT handler
				if self.original_sigint_handler:
					signal.signal(signal.SIGINT, self.original_sigint_handler)
			else:
				# On Unix-like systems, use asyncio's signal handler removal
				self.loop.remove_signal_handler(signal.SIGINT)
				self.loop.remove_signal_handler(signal.SIGTERM)

				# Restore original handlers if available
				if self.original_sigint_handler:
					signal.signal(signal.SIGINT, self.original_sigint_handler)
				if self.original_sigterm_handler:
					signal.signal(signal.SIGTERM, self.original_sigterm_handler)
		except Exception as e:
			logger.warning(f'Error while unregistering signal handlers: {e}')

	def _handle_second_ctrl_c(self) -> None:
		"""
		Handle a second Ctrl+C press by performing cleanup and exiting.
		This is shared logic used by both sigint_handler and wait_for_resume.
		"""
		global _exiting

		if not _exiting:
			_exiting = True

			# Call custom exit callback if provided
			if self.custom_exit_callback:
				try:
					self.custom_exit_callback()
				except Exception as e:
					logger.error(f'Error in exit callback: {e}')

		# Force immediate exit - more reliable than sys.exit()
		print('\n\n🛑  Got second Ctrl+C. Exiting immediately...\n', file=stderr)

		# Reset terminal to a clean state by sending multiple escape sequences
		# Order matters for terminal resets - we try different approaches

		# Reset terminal modes for both stdout and stderr
		print('\033[?25h', end='', flush=True, file=stderr)  # Show cursor
		print('\033[?25h', end='', flush=True)  # Show cursor

		# Reset text attributes and terminal modes
		print('\033[0m', end='', flush=True, file=stderr)  # Reset text attributes
		print('\033[0m', end='', flush=True)  # Reset text attributes

		# Disable special input modes that may cause arrow keys to output control chars
		print('\033[?1l', end='', flush=True, file=stderr)  # Reset cursor keys to normal mode
		print('\033[?1l', end='', flush=True)  # Reset cursor keys to normal mode

		# Disable bracketed paste mode
		print('\033[?2004l', end='', flush=True, file=stderr)
		print('\033[?2004l', end='', flush=True)

		# Carriage return helps ensure a clean line
		print('\r', end='', flush=True, file=stderr)
		print('\r', end='', flush=True)

		# these ^^ attempts dont work as far as we can tell
		# we still dont know what causes the broken input, if you know how to fix it, please let us know
		print('(tip: press [Enter] once to fix escape codes appearing after chrome exit)', file=stderr)

		os._exit(0)

	def sigint_handler(self) -> None:
		"""
		SIGINT (Ctrl+C) handler.

		First Ctrl+C: Cancel current step and pause.
		Second Ctrl+C: Exit immediately if exit_on_second_int is True.
		"""
		global _exiting

		if _exiting:
			# Already exiting, force exit immediately
			os._exit(0)

		if getattr(self.loop, 'ctrl_c_pressed', False):
			# If we're in the waiting for input state, let the pause method handle it
			if getattr(self.loop, 'waiting_for_input', False):
				return

			# Second Ctrl+C - exit immediately if configured to do so
			if self.exit_on_second_int:
				self._handle_second_ctrl_c()

		# Mark that Ctrl+C was pressed
		setattr(self.loop, 'ctrl_c_pressed', True)

		# Cancel current tasks that should be interruptible - this is crucial for immediate pausing
		self._cancel_interruptible_tasks()

		# Call pause callback if provided - this sets the paused flag
		if self.pause_callback:
			try:
				self.pause_callback()
			except Exception as e:
				logger.error(f'Error in pause callback: {e}')

		# Log pause message after pause_callback is called (not before)
		print('----------------------------------------------------------------------', file=stderr)

	def sigterm_handler(self) -> None:
		"""
		SIGTERM handler.

		Always exits the program completely.
		"""
		global _exiting
		if not _exiting:
			_exiting = True
			print('\n\n🛑 SIGTERM received. Exiting immediately...\n\n', file=stderr)

			# Call custom exit callback if provided
			if self.custom_exit_callback:
				self.custom_exit_callback()

		os._exit(0)

	def _cancel_interruptible_tasks(self) -> None:
		"""Cancel current tasks that should be interruptible."""
		current_task = asyncio.current_task(self.loop)
		for task in asyncio.all_tasks(self.loop):
			if task != current_task and not task.done():
				task_name = task.get_name() if hasattr(task, 'get_name') else str(task)
				# Cancel tasks that match certain patterns
				if any(pattern in task_name for pattern in self.interruptible_task_patterns):
					logger.debug(f'Cancelling task: {task_name}')
					task.cancel()
					# Add exception handler to silence "Task exception was never retrieved" warnings
					task.add_done_callback(lambda t: t.exception() if t.cancelled() else None)

		# Also cancel the current task if it's interruptible
		if current_task and not current_task.done():
			task_name = current_task.get_name() if hasattr(current_task, 'get_name') else str(current_task)
			if any(pattern in task_name for pattern in self.interruptible_task_patterns):
				logger.debug(f'Cancelling current task: {task_name}')
				current_task.cancel()

	def wait_for_resume(self) -> None:
		"""
		Wait for user input to resume or exit.

		This method should be called after handling the first Ctrl+C.
		It temporarily restores default signal handling to allow catching
		a second Ctrl+C directly.
		"""
		# Set flag to indicate we're waiting for input
		setattr(self.loop, 'waiting_for_input', True)

		# Temporarily restore default signal handling for SIGINT
		# This ensures KeyboardInterrupt will be raised during input()
		original_handler = signal.getsignal(signal.SIGINT)
		try:
			signal.signal(signal.SIGINT, signal.default_int_handler)
		except ValueError:
			# we are running in a thread other than the main thread
			# or signal handlers are not supported for some other reason
			pass

		green = '\x1b[32;1m'
		red = '\x1b[31m'
		blink = '\033[33;5m'
		unblink = '\033[0m'
		reset = '\x1b[0m'

		try:  # escape code is to blink the ...
			print(
				f'➡️  Press {green}[Enter]{reset} to resume or {red}[Ctrl+C]{reset} again to exit{blink}...{unblink} ',
				end='',
				flush=True,
				file=stderr,
			)
			input()  # This will raise KeyboardInterrupt on Ctrl+C

			# Call resume callback if provided
			if self.resume_callback:
				self.resume_callback()
		except KeyboardInterrupt:
			# Use the shared method to handle second Ctrl+C
			self._handle_second_ctrl_c()
		finally:
			try:
				# Restore our signal handler
				signal.signal(signal.SIGINT, original_handler)
				setattr(self.loop, 'waiting_for_input', False)
			except Exception:
				pass

	def reset(self) -> None:
		"""Reset state after resuming."""
		# Clear the flags
		if hasattr(self.loop, 'ctrl_c_pressed'):
			setattr(self.loop, 'ctrl_c_pressed', False)
		if hasattr(self.loop, 'waiting_for_input'):
			setattr(self.loop, 'waiting_for_input', False)


def time_execution_sync(additional_text: str = '') -> Callable[[Callable[P, R]], Callable[P, R]]:
	def decorator(func: Callable[P, R]) -> Callable[P, R]:
		@wraps(func)
		def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
			start_time = time.time()
			result = func(*args, **kwargs)
			execution_time = time.time() - start_time
			# Only log if execution takes more than 0.25 seconds
			if execution_time > 0.25:
				self_has_logger = args and getattr(args[0], 'logger', None)
				if self_has_logger:
					logger = getattr(args[0], 'logger')
				elif 'agent' in kwargs:
					logger = getattr(kwargs['agent'], 'logger')
				elif 'browser_session' in kwargs:
					logger = getattr(kwargs['browser_session'], 'logger')
				else:
					logger = logging.getLogger(__name__)
				logger.debug(f'⏳ {additional_text.strip("-")}() took {execution_time:.2f}s')
			return result

		return wrapper

	return decorator


def time_execution_async(
	additional_text: str = '',
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
	def decorator(func: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, Coroutine[Any, Any, R]]:
		@wraps(func)
		async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
			start_time = time.time()
			result = await func(*args, **kwargs)
			execution_time = time.time() - start_time
			# Only log if execution takes more than 0.25 seconds to avoid spamming the logs
			# you can lower this threshold locally when you're doing dev work to performance optimize stuff
			if execution_time > 0.25:
				self_has_logger = args and getattr(args[0], 'logger', None)
				if self_has_logger:
					logger = getattr(args[0], 'logger')
				elif 'agent' in kwargs:
					logger = getattr(kwargs['agent'], 'logger')
				elif 'browser_session' in kwargs:
					logger = getattr(kwargs['browser_session'], 'logger')
				else:
					logger = logging.getLogger(__name__)
				logger.debug(f'⏳ {additional_text.strip("-")}() took {execution_time:.2f}s')
			return result

		return wrapper

	return decorator


def singleton(cls):
	instance = [None]

	def wrapper(*args, **kwargs):
		if instance[0] is None:
			instance[0] = cls(*args, **kwargs)
		return instance[0]

	return wrapper


def check_env_variables(keys: list[str], any_or_all=all) -> bool:
	"""Check if all required environment variables are set"""
	return any_or_all(os.getenv(key, '').strip() for key in keys)


def is_unsafe_pattern(pattern: str) -> bool:
	"""
	Check if a domain pattern has complex wildcards that could match too many domains.

	Args:
		pattern: The domain pattern to check

	Returns:
		bool: True if the pattern has unsafe wildcards, False otherwise
	"""
	# Extract domain part if there's a scheme
	if '://' in pattern:
		_, pattern = pattern.split('://', 1)

	# Remove safe patterns (*.domain and domain.*)
	bare_domain = pattern.replace('.*', '').replace('*.', '')

	# If there are still wildcards, it's potentially unsafe
	return '*' in bare_domain


def is_new_tab_page(url: str) -> bool:
	"""
	Check if a URL is a new tab page (about:blank, chrome://new-tab-page, or chrome://newtab).

	Args:
		url: The URL to check

	Returns:
		bool: True if the URL is a new tab page, False otherwise
	"""
	return url in ('about:blank', 'chrome://new-tab-page/', 'chrome://new-tab-page', 'chrome://newtab/', 'chrome://newtab')


def match_url_with_domain_pattern(url: str, domain_pattern: str, log_warnings: bool = False) -> bool:
	"""
	Check if a URL matches a domain pattern. SECURITY CRITICAL.

	Supports optional glob patterns and schemes:
	- *.example.com will match sub.example.com and example.com
	- *google.com will match google.com, agoogle.com, and www.google.com
	- http*://example.com will match http://example.com, https://example.com
	- chrome-extension://* will match chrome-extension://aaaaaaaaaaaa and chrome-extension://bbbbbbbbbbbbb

	When no scheme is specified, https is used by default for security.
	For example, 'example.com' will match 'https://example.com' but not 'http://example.com'.

	Note: New tab pages (about:blank, chrome://new-tab-page) must be handled at the callsite, not inside this function.

	Args:
		url: The URL to check
		domain_pattern: Domain pattern to match against
		log_warnings: Whether to log warnings about unsafe patterns

	Returns:
		bool: True if the URL matches the pattern, False otherwise
	"""
	try:
		# Note: new tab pages should be handled at the callsite, not here
		if is_new_tab_page(url):
			return False

		parsed_url = urlparse(url)

		# Extract only the hostname and scheme components
		scheme = parsed_url.scheme.lower() if parsed_url.scheme else ''
		domain = parsed_url.hostname.lower() if parsed_url.hostname else ''

		if not scheme or not domain:
			return False

		# Normalize the domain pattern
		domain_pattern = domain_pattern.lower()

		# Handle pattern with scheme
		if '://' in domain_pattern:
			pattern_scheme, pattern_domain = domain_pattern.split('://', 1)
		else:
			pattern_scheme = 'https'  # Default to matching only https for security
			pattern_domain = domain_pattern

		# Handle port in pattern (we strip ports from patterns since we already
		# extracted only the hostname from the URL)
		if ':' in pattern_domain and not pattern_domain.startswith(':'):
			pattern_domain = pattern_domain.split(':', 1)[0]

		# If scheme doesn't match, return False
		if not fnmatch(scheme, pattern_scheme):
			return False

		# Check for exact match
		if pattern_domain == '*' or domain == pattern_domain:
			return True

		# Handle glob patterns
		if '*' in pattern_domain:
			# Check for unsafe glob patterns
			# First, check for patterns like *.*.domain which are unsafe
			if pattern_domain.count('*.') > 1 or pattern_domain.count('.*') > 1:
				if log_warnings:
					logger = logging.getLogger(__name__)
					logger.error(f'⛔️ Multiple wildcards in pattern=[{domain_pattern}] are not supported')
				return False  # Don't match unsafe patterns

			# Check for wildcards in TLD part (example.*)
			if pattern_domain.endswith('.*'):
				if log_warnings:
					logger = logging.getLogger(__name__)
					logger.error(f'⛔️ Wildcard TLDs like in pattern=[{domain_pattern}] are not supported for security')
				return False  # Don't match unsafe patterns

			# Then check for embedded wildcards
			bare_domain = pattern_domain.replace('*.', '')
			if '*' in bare_domain:
				if log_warnings:
					logger = logging.getLogger(__name__)
					logger.error(f'⛔️ Only *.domain style patterns are supported, ignoring pattern=[{domain_pattern}]')
				return False  # Don't match unsafe patterns

			# Special handling so that *.google.com also matches bare google.com
			if pattern_domain.startswith('*.'):
				parent_domain = pattern_domain[2:]
				if domain == parent_domain or fnmatch(domain, parent_domain):
					return True

			# Normal case: match domain against pattern
			if fnmatch(domain, pattern_domain):
				return True

		return False
	except Exception as e:
		logger = logging.getLogger(__name__)
		logger.error(f'⛔️ Error matching URL {url} with pattern {domain_pattern}: {type(e).__name__}: {e}')
		return False


def merge_dicts(a: dict, b: dict, path: tuple[str, ...] = ()):
	for key in b:
		if key in a:
			if isinstance(a[key], dict) and isinstance(b[key], dict):
				merge_dicts(a[key], b[key], path + (str(key),))
			elif isinstance(a[key], list) and isinstance(b[key], list):
				a[key] = a[key] + b[key]
			elif a[key] != b[key]:
				raise Exception('Conflict at ' + '.'.join(path + (str(key),)))
		else:
			a[key] = b[key]
	return a


@cache
def get_agent_version() -> str:
	"""Получить версию текущего проекта (из pyproject.toml, если он есть)."""
	try:
		package_root = Path(__file__).parent.parent
		pyproject_path = package_root / 'pyproject.toml'

		if pyproject_path.exists():
			import re

			with open(pyproject_path, encoding='utf-8') as f:
				content = f.read()
				match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
				if match:
					version = f'{match.group(1)}'
					os.environ['LIBRARY_VERSION'] = version
					return version
	except Exception as e:
		logger.debug(f'Ошибка определения версии проекта: {type(e).__name__}: {e}')
		return 'unknown'
	return 'unknown'


async def check_latest_agent_version() -> str | None:
	"""Заглушка: проверка последней версии на PyPI отключена."""
	return None


@cache
def get_git_info() -> dict[str, str] | None:
	"""Get git information if installed from git repository"""
	try:
		import subprocess

		package_root = Path(__file__).parent.parent
		git_dir = package_root / '.git'
		if not git_dir.exists():
			return None

		# Get git commit hash
		commit_hash = (
			subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=package_root, stderr=subprocess.DEVNULL).decode().strip()
		)

		# Get git branch
		branch = (
			subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=package_root, stderr=subprocess.DEVNULL)
			.decode()
			.strip()
		)

		# Get remote URL
		remote_url = (
			subprocess.check_output(['git', 'config', '--get', 'remote.origin.url'], cwd=package_root, stderr=subprocess.DEVNULL)
			.decode()
			.strip()
		)

		# Get commit timestamp
		commit_timestamp = (
			subprocess.check_output(['git', 'show', '-s', '--format=%ci', 'HEAD'], cwd=package_root, stderr=subprocess.DEVNULL)
			.decode()
			.strip()
		)

		return {'commit_hash': commit_hash, 'branch': branch, 'remote_url': remote_url, 'commit_timestamp': commit_timestamp}
	except Exception as e:
		logger.debug(f'Error getting git info: {type(e).__name__}: {e}')
		return None


def _log_pretty_path(path: str | Path | None) -> str:
	"""Pretty-print a path, shorten home dir to ~ and cwd to ."""

	if not path or not str(path).strip():
		return ''  # always falsy in -> falsy out so it can be used in ternaries

	# dont print anything thats not a path
	if not isinstance(path, (str, Path)):
		# no other types are safe to just str(path) and log to terminal unless we know what they are
		# e.g. what if we get storage_date=dict | Path and the dict version could contain real cookies
		return f'<{type(path).__name__}>'

	# replace home dir and cwd with ~ and .
	pretty_path = str(path).replace(str(Path.home()), '~').replace(str(Path.cwd().resolve()), '.')

	# wrap in quotes if it contains spaces
	if pretty_path.strip() and ' ' in pretty_path:
		pretty_path = f'"{pretty_path}"'

	return pretty_path


def _log_pretty_url(s: str, max_len: int | None = 22) -> str:
	"""Truncate/pretty-print a URL with a maximum length, removing the protocol and www. prefix"""
	s = s.replace('https://', '').replace('http://', '').replace('www.', '')
	if max_len is not None and len(s) > max_len:
		return s[:max_len] + '…'
	return s


def create_task_with_error_handling(
	coro: Coroutine[Any, Any, T],
	*,
	name: str | None = None,
	logger_instance: logging.Logger | None = None,
	suppress_exceptions: bool = False,
) -> asyncio.Task[T]:
	"""
	Create an asyncio task with proper exception handling to prevent "Task exception was never retrieved" warnings.

	Args:
		coro: The coroutine to wrap in a task
		name: Optional name for the task (useful for debugging)
		logger_instance: Optional logger instance to use. If None, uses module logger.
		suppress_exceptions: If True, logs exceptions at ERROR level. If False, logs at WARNING level
			and exceptions remain retrievable via task.exception() if the caller awaits the task.
			Default False.

	Returns:
		asyncio.Task: The created task with exception handling callback

	Example:
		# Fire-and-forget with suppressed exceptions
		create_task_with_error_handling(some_async_function(), name="my_task", suppress_exceptions=True)

		# Task with retrievable exceptions (if you plan to await it)
		task = create_task_with_error_handling(critical_function(), name="critical")
		result = await task  # Will raise the exception if one occurred
	"""
	task = asyncio.create_task(coro, name=name)
	log = logger_instance or logger

	def _handle_task_exception(t: asyncio.Task[T]) -> None:
		"""Callback to handle task exceptions"""
		exc_to_raise = None
		try:
			# This will raise if the task had an exception
			exc = t.exception()
			if exc is not None:
				task_name = t.get_name() if hasattr(t, 'get_name') else 'unnamed'
				if suppress_exceptions:
					log.error(f'Exception in background task [{task_name}]: {type(exc).__name__}: {exc}', exc_info=exc)
				else:
					# Log at warning level then mark for re-raising
					log.warning(
						f'Exception in background task [{task_name}]: {type(exc).__name__}: {exc}',
						exc_info=exc,
					)
					exc_to_raise = exc
		except asyncio.CancelledError:
			# Task was cancelled, this is normal behavior
			pass
		except Exception as e:
			# Catch any other exception during exception handling (e.g., t.exception() itself failing)
			task_name = t.get_name() if hasattr(t, 'get_name') else 'unnamed'
			log.error(f'Error handling exception in task [{task_name}]: {type(e).__name__}: {e}')

		# Re-raise outside the try-except block so it propagates to the event loop
		if exc_to_raise is not None:
			raise exc_to_raise

	task.add_done_callback(_handle_task_exception)
	return task


def sanitize_surrogates(text: str) -> str:
	"""Remove surrogate characters that can't be encoded in UTF-8.

	Surrogate pairs (U+D800 to U+DFFF) are invalid in UTF-8 when unpaired.
	These often appear in DOM content from mathematical symbols or emojis.

	Args:
		text: The text to sanitize

	Returns:
		Text with surrogate characters removed
	"""
	return text.encode('utf-8', errors='ignore').decode('utf-8')
