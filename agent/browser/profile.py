import sys
import tempfile
from collections.abc import Iterable
from enum import Enum
from functools import cache
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlparse

from pydantic import AfterValidator, AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

# Облачный браузер - опционально (не используется в упрощённой версии)
try:
	from agent.browser.cloud.views import CloudBrowserParams
except ImportError:
	from typing import Any
	CloudBrowserParams = Any  # Заглушка типа, если облачный модуль недоступен
from agent.config import CONFIG
from agent.utils import _log_pretty_path, logger

CHROME_DEBUG_PORT = 9242  # отдельный порт CDP, чтобы не конфликтовать с другими инструментами/браузерами на 9222
DOMAIN_OPTIMIZATION_THRESHOLD = 100  # начиная с такого размера список доменов переводим в set для O(1) поиска
CHROME_DISABLED_COMPONENTS = [
	# Список фич Chromium, которые отключаем для более стабильной и предсказуемой работы агента.
	# Полный перечень: AcceptCHFrame,AutoExpandDetailsElement,AvoidUnnecessaryBeforeUnloadCheckSync,CertificateTransparencyComponentUpdater,DeferRendererTasksAfterInput,DestroyProfileOnBrowserClose,DialMediaRouteProvider,ExtensionManifestV2Disabled,GlobalMediaControls,HttpsUpgrades,ImprovedCookieControls,LazyFrameLoading,LensOverlay,MediaRouter,PaintHolding,ThirdPartyStoragePartitioning,Translate
	'AcceptCHFrame',
	'AutoExpandDetailsElement',
	'AvoidUnnecessaryBeforeUnloadCheckSync',
	'CertificateTransparencyComponentUpdater',
	'DestroyProfileOnBrowserClose',
	'DialMediaRouteProvider',
	# Chromium постепенно отключает manifest v2, но мы оставляем возможность тестировать его, пока это поддерживается.
	'ExtensionManifestV2Disabled',
	'GlobalMediaControls',
	'HttpsUpgrades',
	'ImprovedCookieControls',
	'LazyFrameLoading',
	# Скрывает функцию Lens в адресной строке
	'LensOverlay',
	'MediaRouter',
	'PaintHolding',
	'ThirdPartyStoragePartitioning',
	'Translate',
	# 3
		# Добавлено нами:
	'AutomationControlled',
	'BackForwardCache',
	'OptimizationHints',
	'ProcessPerSiteUpToMainFrameThreshold',
	'InterestFeedContentSuggestions',
	'CalculateNativeWinOcclusion',  # Chrome обычно останавливает рендеринг вкладок, если они не видны (перекрыты другим окном)
	# 'BackForwardCache',  # агент действительно использует навигацию назад/вперёд, но можно отключить, если уберём эту функциональность
	'HeavyAdPrivacyMitigations',
	'PrivacySandboxSettings4',
	'AutofillServerCommunication',
	'CrashReporting',
	'OverscrollHistoryNavigation',
	'InfiniteSessionRestore',
	'ExtensionDisableUnsupportedDeveloper',
	'ExtensionManifestV2Unsupported',
]

CHROME_HEADLESS_ARGS = [
	'--headless=new',
]

CHROME_DOCKER_ARGS = [
	# '--disable-gpu',    # GPU уже поддерживается в headless-режиме в Docker, но иногда удобно тестировать без него
	'--no-sandbox',
	'--disable-gpu-sandbox',
	'--disable-setuid-sandbox',
	'--disable-dev-shm-usage',
	'--no-xshm',
	'--no-zygote',
	# '--single-process',  # может приводить к ошибкам вида \"Target page, context or browser has been closed\" при CDP page.captureScreenshot
	'--disable-site-isolation-trials',  # уменьшает потребление RAM в Docker, но немного повышает детектируемость бота
]


CHROME_DISABLE_SECURITY_ARGS = [
	'--disable-site-isolation-trials',
	'--disable-web-security',
	'--disable-features=IsolateOrigins,site-per-process',
	'--allow-running-insecure-content',
	'--ignore-certificate-errors',
	'--ignore-ssl-errors',
	'--ignore-certificate-errors-spki-list',
]

CHROME_DETERMINISTIC_RENDERING_ARGS = [
	'--deterministic-mode',
	'--js-flags=--random-seed=1157259159',
	'--force-device-scale-factor=2',
	'--enable-webgl',
	# '--disable-skia-runtime-opts',
	# '--disable-2d-canvas-clip-aa',
	'--font-render-hinting=none',
	'--force-color-profile=srgb',
]

CHROME_DEFAULT_ARGS = [
	'--disable-field-trial-config',  # отключаем вариации/эксперименты Chromium для предсказуемого поведения
	'--disable-background-networking',
	'--disable-background-timer-throttling',  # агент может работать во вкладке в фоне, не даём таймерам \"засыпать\"
	'--disable-backgrounding-occluded-windows',  # не фризим окна, даже если они перекрыты другими
	'--disable-back-forward-cache',  # отключаем кеш back/forward, чтобы не было сюрпризов с перехватом запросов
	'--disable-breakpad',
	'--disable-client-side-phishing-detection',
	'--disable-component-extensions-with-background-pages',
	'--disable-component-update',  # убираем лишнюю сетевую активность обновлений после старта
	'--no-default-browser-check',
	# '--disable-default-apps',
	'--disable-dev-shm-usage',  # важно для Docker, в обычной среде не мешает
	# '--disable-extensions',
	# '--disable-features=' + disabledFeatures(assistantMode).join(','),
	# '--allow-pre-commit-input',  # duplicate removed
	'--disable-hang-monitor',
	'--disable-ipc-flooding-protection',  # снимаем защиту от \"частых IPC\", чтобы CDP-вызовы не душились
	'--disable-popup-blocking',
	'--disable-prompt-on-repost',
	'--disable-renderer-backgrounding',
	# '--force-color-profile=srgb',  # moved to CHROME_DETERMINISTIC_RENDERING_ARGS
	'--metrics-recording-only',  # минимальный сбор метрик
	'--no-first-run',
	'--no-service-autorun',  # не запускаем фоновые сервисы Chromium
	'--export-tagged-pdf',  # включаем экспорт pdf с оглавлением
	'--disable-search-engine-choice-screen',  # убираем экран выбора поисковика
	'--unsafely-disable-devtools-self-xss-warnings',  # отключаем предупреждения DevTools про self-XSS
	# добавлено нами:
	'--enable-features=NetworkService,NetworkServiceInProcess',
	'--enable-network-information-downlink-max',
	'--test-type=gpu',  # включаем gpu-режим тестового типа
	'--disable-sync',
	'--allow-legacy-extension-manifests',
	'--allow-pre-commit-input',
	'--disable-blink-features=AutomationControlled',  # уменьшаем очевидность автоматизации
	'--install-autogenerated-theme=0,0,0',
	# '--hide-scrollbars',                     # оставляем скроллбары, агент по ним понимает, что можно ещё проскроллить
	'--log-level=2',
	# '--enable-logging=stderr',
	'--disable-focus-on-load',
	'--disable-window-activation',
	'--generate-pdf-document-outline',
	'--no-pings',
	'--ash-no-nudges',
	'--disable-infobars',
	'--simulate-outdated-no-au="Tue, 31 Dec 2099 23:59:59 GMT"',
	'--hide-crash-restore-bubble',
	'--suppress-message-center-popups',
	'--disable-domain-reliability',
	'--disable-datasaver-prompt',
	'--disable-speech-synthesis-api',
	'--disable-speech-api',
	'--disable-print-preview',
	'--safebrowsing-disable-auto-update',
	'--disable-external-intent-requests',
	'--disable-desktop-notifications',
	'--noerrdialogs',
	'--silent-debugger-extension-api',
	# Подавляем приветственные вкладки расширений при автоматизации
	'--disable-extensions-http-throttling',
	'--extensions-on-chrome-urls',
	'--disable-default-apps',
	f'--disable-features={",".join(CHROME_DISABLED_COMPONENTS)}',
]


class ViewportSize(BaseModel):
	width: int = Field(ge=0)
	height: int = Field(ge=0)

	def __getitem__(self, key: str) -> int:
		return dict(self)[key]

	def __setitem__(self, key: str, value: int) -> None:
		setattr(self, key, value)


@cache
def get_display_size() -> ViewportSize | None:
	# macOS
	try:
		from AppKit import NSScreen  # type: ignore[import]

		screen = NSScreen.mainScreen().frame()
		size = ViewportSize(width=int(screen.size.width), height=int(screen.size.height))
		logger.debug(f'Display size: {size}')
		return size
	except Exception:
		pass

	# Windows & Linux
	try:
		from screeninfo import get_monitors

		monitors = get_monitors()
		monitor = monitors[0]
		size = ViewportSize(width=int(monitor.width), height=int(monitor.height))
		logger.debug(f'Display size: {size}')
		return size
	except Exception:
		pass

	logger.debug('No display size found')
	return None


def get_window_adjustments() -> tuple[int, int]:
	"""Вернуть рекомендуемые смещения окна по осям x, y для аккуратного позиционирования."""

	if sys.platform == 'darwin':  # macOS
		return -4, 24  # macOS имеет небольшую строку заголовка, без рамки
	elif sys.platform == 'win32':  # Windows
		return -8, 0  # Windows имеет рамку слева
	else:  # Linux
		return 0, 0


def validate_url(url: str, schemes: Iterable[str] = ()) -> str:
	"""Проверить формат URL и при необходимости проконтролировать допустимые схемы (http/https и т.п.)."""
	parsed_url = urlparse(url)
	if not parsed_url.netloc:
		raise ValueError(f'Invalid URL format: {url}')
	if schemes and parsed_url.scheme and parsed_url.scheme.lower() not in schemes:
		raise ValueError(f'URL has invalid scheme: {url} (expected one of {schemes})')
	return url


def validate_float_range(value: float, min_val: float, max_val: float) -> float:
	"""Проверить, что число с плавающей точкой лежит в заданном диапазоне."""
	if not min_val <= value <= max_val:
		raise ValueError(f'Value {value} outside of range {min_val}-{max_val}')
	return value


def validate_cli_arg(arg: str) -> str:
	"""Проверить, что аргумент командной строки имеет корректный формат (начинается с --)."""
	if not arg.startswith('--'):
		raise ValueError(f'Invalid CLI argument: {arg} (should start with --, e.g. --some-key="some value here")')
	return arg


# ===== Enum definitions =====


class RecordHarContent(str, Enum):
	OMIT = 'omit'
	EMBED = 'embed'
	ATTACH = 'attach'


class RecordHarMode(str, Enum):
	FULL = 'full'
	MINIMAL = 'minimal'


class BrowserChannel(str, Enum):
	CHROMIUM = 'chromium'
	CHROME = 'chrome'
	CHROME_BETA = 'chrome-beta'
	CHROME_DEV = 'chrome-dev'
	CHROME_CANARY = 'chrome-canary'
	MSEDGE = 'msedge'
	MSEDGE_BETA = 'msedge-beta'
	MSEDGE_DEV = 'msedge-dev'
	MSEDGE_CANARY = 'msedge-canary'


# Using constants from central location in agent.config
AGENT_DEFAULT_CHANNEL = BrowserChannel.CHROMIUM


# ===== Type definitions with validators =====

UrlStr = Annotated[str, AfterValidator(validate_url)]
NonNegativeFloat = Annotated[float, AfterValidator(lambda x: validate_float_range(x, 0, float('inf')))]
CliArgStr = Annotated[str, AfterValidator(validate_cli_arg)]


# ===== Base Models =====


class BrowserContextArgs(BaseModel):
	"""
	Base model for common browser context parameters used by
	both BrowserType.new_context() and BrowserType.launch_persistent_context().
	"""

	model_config = ConfigDict(extra='ignore', validate_assignment=False, revalidate_instances='always', populate_by_name=True)

	# Browser context parameters
	accept_downloads: bool = True

	# Security options
	# proxy: ProxySettings | None = None
	permissions: list[str] = Field(
		default_factory=lambda: ['clipboardReadWrite', 'notifications'],
		description='Browser permissions to grant (CDP Browser.grantPermissions).',
		# clipboardReadWrite is for google sheets and pyperclip automations
		# notifications are to avoid browser fingerprinting
	)
	# client_certificates: list[ClientCertificate] = Field(default_factory=list)
	# http_credentials: HttpCredentials | None = None

	# Viewport options
	user_agent: str | None = None
	screen: ViewportSize | None = None
	viewport: ViewportSize | None = Field(default=None)
	no_viewport: bool | None = None
	device_scale_factor: NonNegativeFloat | None = None
	# geolocation: Geolocation | None = None

	# Recording Options
	record_har_content: RecordHarContent = RecordHarContent.EMBED
	record_har_mode: RecordHarMode = RecordHarMode.FULL
	record_har_path: str | Path | None = Field(default=None, validation_alias=AliasChoices('save_har_path', 'record_har_path'))
	record_video_dir: str | Path | None = Field(
		default=None, validation_alias=AliasChoices('save_recording_path', 'record_video_dir')
	)


class BrowserConnectArgs(BaseModel):
	"""
	Base model for common browser connect parameters used by
	both connect_over_cdp() and connect_over_ws().
	"""

	model_config = ConfigDict(extra='ignore', validate_assignment=True, revalidate_instances='always', populate_by_name=True)

	headers: dict[str, str] | None = Field(default=None, description='Additional HTTP headers to be sent with connect request')


class BrowserLaunchArgs(BaseModel):
	"""
	Base model for common browser launch parameters used by
	both launch() and launch_persistent_context().
	"""

	model_config = ConfigDict(
		extra='ignore',
		validate_assignment=True,
		revalidate_instances='always',
		from_attributes=True,
		validate_by_name=True,
		validate_by_alias=True,
		populate_by_name=True,
	)

	env: dict[str, str | float | bool] | None = Field(
		default=None,
		description='Extra environment variables to set when launching the browser. If None, inherits from the current process.',
	)
	executable_path: str | Path | None = Field(
		default=None,
		validation_alias=AliasChoices('browser_binary_path', 'chrome_binary_path'),
		description='Path to the chromium-based browser executable to use.',
	)
	headless: bool | None = Field(default=None, description='Whether to run the browser in headless or windowed mode.')
	args: list[CliArgStr] = Field(
		default_factory=list, description='List of *extra* CLI args to pass to the browser when launching.'
	)
	ignore_default_args: list[CliArgStr] | Literal[True] = Field(
		default_factory=lambda: [
			'--enable-automation',  # маскируем отпечаток автоматизации через JS и другие флаги
			'--disable-extensions',  # разрешаем расширения браузера
			'--hide-scrollbars',  # всегда показываем скроллбары на скриншотах, чтобы агент понимал, что есть ещё контент ниже
			'--disable-features=AcceptCHFrame,AutoExpandDetailsElement,AvoidUnnecessaryBeforeUnloadCheckSync,CertificateTransparencyComponentUpdater,DeferRendererTasksAfterInput,DestroyProfileOnBrowserClose,DialMediaRouteProvider,ExtensionManifestV2Disabled,GlobalMediaControls,HttpsUpgrades,ImprovedCookieControls,LazyFrameLoading,LensOverlay,MediaRouter,PaintHolding,ThirdPartyStoragePartitioning,Translate',
		],
		description='List of default CLI args to stop playwright from applying',
	)
	channel: BrowserChannel | None = None
	chromium_sandbox: bool = Field(
		default=not CONFIG.IN_DOCKER, description='Whether to enable Chromium sandboxing (recommended unless inside Docker).'
	)
	devtools: bool = Field(
		default=False, description='Whether to open DevTools panel automatically for every page, only works when headless=False.'
	)

	# proxy: ProxySettings | None = Field(default=None, description='Proxy settings to use to connect to the browser.')
	downloads_path: str | Path | None = Field(
		default=None,
		description='Directory to save downloads to.',
		validation_alias=AliasChoices('downloads_dir', 'save_downloads_path'),
	)
	traces_dir: str | Path | None = Field(
		default=None,
		description='Directory for saving playwright trace.zip files (playwright actions, screenshots, DOM snapshots, HAR traces).',
		validation_alias=AliasChoices('trace_path', 'traces_dir'),
	)

	# firefox_user_prefs: dict[str, str | float | bool] = Field(default_factory=dict)

	@model_validator(mode='after')
	def validate_devtools_headless(self) -> Self:
		"""Защитная проверка: нельзя одновременно включать headless и devtools."""
		assert not (self.headless and self.devtools), 'headless=True and devtools=True cannot both be set at the same time'
		return self

	@model_validator(mode='after')
	def set_default_downloads_path(self) -> Self:
		"""Назначить уникальный путь для загрузок, если он не указан явно."""
		if self.downloads_path is None:
			import uuid

			# Создаём уникальную папку в /tmp для загрузок
			unique_id = str(uuid.uuid4())[:8]  # 8 символов
			downloads_path = Path(f'/tmp/agent-downloads-{unique_id}')

			# Убеждаемся, что путь ещё не существует (крайне маловероятно, но возможно)
			while downloads_path.exists():
				unique_id = str(uuid.uuid4())[:8]
				downloads_path = Path(f'/tmp/agent-downloads-{unique_id}')

			self.downloads_path = downloads_path
			self.downloads_path.mkdir(parents=True, exist_ok=True)
		return self

	@staticmethod
	def args_as_dict(args: list[str]) -> dict[str, str]:
		"""Преобразовать список CLI-аргументов запуска в словарь."""
		args_dict = {}
		for arg in args:
			key, value, *_ = [*arg.split('=', 1), '', '', '']
			args_dict[key.strip().lstrip('-')] = value.strip()
		return args_dict

	@staticmethod
	def args_as_list(args: dict[str, str]) -> list[str]:
		"""Преобразовать словарь CLI-аргументов запуска в список строк."""
		return [f'--{key.lstrip("-")}={value}' if value else f'--{key.lstrip("-")}' for key, value in args.items()]


# ===== API-specific Models =====


class BrowserNewContextArgs(BrowserContextArgs):
	"""
	Pydantic-модель для аргументов new_context().
Расширяет базовые параметры контекста полем storage_state.
	"""

	model_config = ConfigDict(extra='ignore', validate_assignment=False, revalidate_instances='always', populate_by_name=True)

	# storage_state is not supported in launch_persistent_context()
	storage_state: str | Path | dict[str, Any] | None = None
	# Примечание: можно использовать тип StorageState вместо dict[str, Any]

	# to apply this to existing contexts (incl cookies, localStorage, IndexedDB)

	# @field_validator('storage_state', mode='after')
	# def load_storage_state_from_file(self) -> Self:
	# 	"""Load storage_state from file if it's a path."""
	# 	if isinstance(self.storage_state, (str, Path)):
	# 		storage_state_file = Path(self.storage_state)
	# 		try:
	# 			parsed_storage_state = json.loads(storage_state_file.read_text())
	# 			validated_storage_state = StorageState(**parsed_storage_state)
	# 			self.storage_state = validated_storage_state
	# 		except Exception as e:
	# 			raise ValueError(f'Failed to load storage state file {self.storage_state}: {e}') from e
	# 	return self
	pass


class BrowserLaunchPersistentContextArgs(BrowserLaunchArgs, BrowserContextArgs):
	"""
	Pydantic-модель для аргументов launch_persistent_context().
Объединяет параметры запуска браузера и контекста,
дополнительно добавляет параметр user_data_dir.
	"""

	model_config = ConfigDict(extra='ignore', validate_assignment=False, revalidate_instances='always')

	# Required parameter specific to launch_persistent_context, but can be None to use incognito temp dir
	user_data_dir: str | Path | None = None

	@field_validator('user_data_dir', mode='after')
	@classmethod
	def validate_user_data_dir(cls, v: str | Path | None) -> str | Path | None:
		"""Убедиться, что каталог пользовательских данных не указывает на дефолтный путь."""
		# Если user_data_dir явно не указан, возвращаем None
		# Временный каталог будет создан только если не используется storage_state
		# Это предотвращает конфликты и рекурсию
		if v is None:
			return None
		return Path(v).expanduser().resolve()


class ProxySettings(BaseModel):
	"""Настройки прокси для трафика Chromium.

- server: полный URL прокси (например, \"http://host:8080\" или \"socks5://host:1080\")
- bypass: список хостов через запятую, которые нужно обходить (например, \"localhost,127.0.0.1,*.internal\")
- username/password: при необходимости — учётные данные для авторизации на прокси
	"""

	server: str | None = Field(default=None, description='Proxy URL, e.g. http://host:8080 or socks5://host:1080')
	bypass: str | None = Field(default=None, description='Comma-separated hosts to bypass, e.g. localhost,127.0.0.1,*.internal')
	username: str | None = Field(default=None, description='Proxy auth username')
	password: str | None = Field(default=None, description='Proxy auth password')

	def __getitem__(self, key: str) -> str | None:
		return getattr(self, key)


class BrowserProfile(BrowserConnectArgs, BrowserLaunchPersistentContextArgs, BrowserLaunchArgs, BrowserNewContextArgs):
	"""
	A BrowserProfile is a static template collection of kwargs that can be passed to:
		- BrowserType.launch(**BrowserLaunchArgs)
		- BrowserType.connect(**BrowserConnectArgs)
		- BrowserType.connect_over_cdp(**BrowserConnectArgs)
		- BrowserType.launch_persistent_context(**BrowserLaunchPersistentContextArgs)
		- BrowserContext.new_context(**BrowserNewContextArgs)
		- BrowserSession(**BrowserProfile)
	"""

	model_config = ConfigDict(
		extra='ignore',
		validate_assignment=True,
		revalidate_instances='always',
		from_attributes=True,
		validate_by_name=True,
		validate_by_alias=True,
	)

	# ... extends options defined in:
	# BrowserLaunchPersistentContextArgs, BrowserLaunchArgs, BrowserNewContextArgs, BrowserConnectArgs

	# Session/connection configuration
	cdp_url: str | None = Field(default=None, description='CDP URL for connecting to existing browser instance')
	is_local: bool = Field(default=False, description='Whether this is a local browser instance')
	use_cloud: bool = Field(
		default=False,
		description='Использовать облачный браузер вместо локального (в упрощённой версии выключено)',
	)

	@property
	def cloud_browser(self) -> bool:
		"""Псевдоним для поля use_cloud для совместимости."""
		return self.use_cloud

	cloud_browser_params: CloudBrowserParams | None = Field(
		default=None, description='Parameters for creating a cloud browser instance'
	)

	# custom options we provide that aren't native playwright kwargs
	disable_security: bool = Field(default=False, description='Disable browser security features.')
	deterministic_rendering: bool = Field(default=False, description='Enable deterministic rendering flags.')
	allowed_domains: list[str] | set[str] | None = Field(
		default=None,
		description='List of allowed domains for navigation e.g. ["*.google.com", "https://example.com", "chrome-extension://*"]. Lists with 100+ items are auto-optimized to sets (no pattern matching).',
	)
	prohibited_domains: list[str] | set[str] | None = Field(
		default=None,
		description='List of prohibited domains for navigation e.g. ["*.google.com", "https://example.com", "chrome-extension://*"]. Allowed domains take precedence over prohibited domains. Lists with 100+ items are auto-optimized to sets (no pattern matching).',
	)
	block_ip_addresses: bool = Field(
		default=False,
		description='Block navigation to URLs containing IP addresses (both IPv4 and IPv6). When True, blocks all IP-based URLs including localhost and private networks.',
	)
	keep_alive: bool | None = Field(default=None, description='Keep browser alive after agent run.')

	# --- Proxy settings ---
	# New consolidated proxy config (typed)
	proxy: ProxySettings | None = Field(
		default=None,
		description='Proxy settings. Use agent.browser.profile.ProxySettings(server, bypass, username, password)',
	)
	enable_default_extensions: bool = Field(
		default=True,
		description="Enable automation-optimized extensions: ad blocking (uBlock Origin), cookie handling (I still don't care about cookies), and URL cleaning (ClearURLs). All extensions work automatically without manual intervention. Extensions are automatically downloaded and loaded when enabled.",
	)
	demo_mode: bool = Field(
		default=False,
		description='Enable demo mode side panel that streams agent logs directly inside the browser window (requires headless=False).',
	)
	cookie_whitelist_domains: list[str] = Field(
		default_factory=lambda: ['nature.com', 'qatarairways.com'],
		description='List of domains to whitelist in the "I still don\'t care about cookies" extension, preventing automatic cookie banner handling on these sites.',
	)

	window_size: ViewportSize | None = Field(
		default=None,
		description='Browser window size to use when headless=False.',
	)
	window_height: int | None = Field(default=None, description='DEPRECATED, use window_size["height"] instead', exclude=True)
	window_width: int | None = Field(default=None, description='DEPRECATED, use window_size["width"] instead', exclude=True)
	window_position: ViewportSize | None = Field(
		default=ViewportSize(width=0, height=0),
		description='Window position to use for the browser x,y from the top left when headless=False.',
	)
	cross_origin_iframes: bool = Field(
		default=True,
		description='Enable cross-origin iframe support (OOPIF/Out-of-Process iframes). When False, only same-origin frames are processed to avoid complexity and hanging.',
	)
	max_iframes: int = Field(
		default=100,
		description='Maximum number of iframe documents to process to prevent crashes.',
	)
	max_iframe_depth: int = Field(
		ge=0,
		default=5,
		description='Maximum depth for cross-origin iframe recursion (default: 5 levels deep).',
	)

	# --- Page load/wait timings ---

	minimum_wait_page_load_time: float = Field(default=0.5, description='Minimum time to wait before capturing page state.')
	wait_for_network_idle_page_load_time: float = Field(default=0.3, description='Time to wait for network idle.')

	wait_between_actions: float = Field(default=0.1, description='Time to wait between actions.')

	# --- UI/viewport/DOM ---
	highlight_elements: bool = Field(default=True, description='Highlight interactive elements on the page.')
	dom_highlight_elements: bool = Field(
		default=False, description='Highlight interactive elements in the DOM (only for debugging purposes).'
	)
	filter_highlight_ids: bool = Field(
		default=True, description='Only show element IDs in highlights if llm_representation is less than 10 characters.'
	)
	paint_order_filtering: bool = Field(default=True, description='Enable paint order filtering. Slightly experimental.')
	interaction_highlight_color: str = Field(
		default='rgb(255, 127, 39)',
		description='Color to use for highlighting elements during interactions (CSS color string).',
	)
	interaction_highlight_duration: float = Field(default=1.0, description='Duration in seconds to show interaction highlights.')

	# --- Downloads ---
	auto_download_pdfs: bool = Field(default=True, description='Automatically download PDFs when navigating to PDF viewer pages.')

	profile_directory: str = 'Default'  # e.g. 'Profile 1', 'Profile 2', 'Custom Profile', etc.

	# these can be found in BrowserLaunchArgs, BrowserLaunchPersistentContextArgs, BrowserNewContextArgs, BrowserConnectArgs:
	# save_recording_path: alias of record_video_dir
	# save_har_path: alias of record_har_path
	# trace_path: alias of traces_dir

	# these shadow the old playwright args on BrowserContextArgs, but it's ok
	# because we handle them ourselves in a watchdog and we no longer use playwright, so they should live in the scope for our own config in BrowserProfile long-term
	record_video_dir: Path | None = Field(
		default=None,
		description='Directory to save video recordings. If set, a video of the session will be recorded.',
		validation_alias=AliasChoices('save_recording_path', 'record_video_dir'),
	)
	record_video_size: ViewportSize | None = Field(
		default=None, description='Video frame size. If not set, it will use the viewport size.'
	)
	record_video_framerate: int = Field(default=30, description='The framerate to use for the video recording.')

	# Примечание: поддержка расширений реализована частично
	# extension_ids_to_preinstall: list[str] = Field(
	# 	default_factory=list, description='List of Chrome extension IDs to preinstall.'
	# )
	# extensions_dir: Path = Field(
	# 	default_factory=lambda: Path('~/.config/agent/cache/extensions').expanduser(),
	# 	description='Directory containing .crx extension files.',
	# )

	def __repr__(self) -> str:
		short_dir = _log_pretty_path(self.user_data_dir) if self.user_data_dir else '<incognito>'
		return f'BrowserProfile(user_data_dir= {short_dir}, headless={self.headless})'

	def __str__(self) -> str:
		return 'BrowserProfile'

	@field_validator('allowed_domains', 'prohibited_domains', mode='after')
	@classmethod
	def optimize_large_domain_lists(cls, v: list[str] | set[str] | None) -> list[str] | set[str] | None:
		"""Преобразует большие списки доменов (>=100 элементов) в множества для O(1) поиска."""
		if v is None or isinstance(v, set):
			return v

		if len(v) >= DOMAIN_OPTIMIZATION_THRESHOLD:
			logger.warning(
				f'🔧 Optimizing domain list with {len(v)} items to set for O(1) lookup. '
				f'Note: Pattern matching (*.domain.com, etc.) is not supported for lists >= {DOMAIN_OPTIMIZATION_THRESHOLD} items. '
				f'Use exact domains only or keep list size < {DOMAIN_OPTIMIZATION_THRESHOLD} for pattern support.'
			)
			return set(v)

		return v

	@model_validator(mode='after')
	def copy_old_config_names_to_new(self) -> Self:
		"""Копирует старые настройки window_width и window_height в window_size."""
		if self.window_width or self.window_height:
			logger.warning(
				f'⚠️ BrowserProfile(window_width=..., window_height=...) are deprecated, use BrowserProfile(window_size={"width": 1920, "height": 1080}) instead.'
			)
			window_size = self.window_size or ViewportSize(width=0, height=0)
			window_size['width'] = window_size['width'] or self.window_width or 1920
			window_size['height'] = window_size['height'] or self.window_height or 1080
			self.window_size = window_size

		return self

	@model_validator(mode='after')
	def warn_storage_state_user_data_dir_conflict(self) -> Self:
		"""Предупреждает, когда одновременно установлены storage_state и user_data_dir, так как это может вызвать конфликты."""
		has_storage_state = self.storage_state is not None
		
		# Для CDP браузера всегда нужен user_data_dir для запуска
		# Если user_data_dir не указан, создаем временный каталог
		if self.user_data_dir is None:
			# Используем object.__setattr__ чтобы избежать повторной валидации
			temp_dir = tempfile.mkdtemp(prefix='agent-user-data-dir-')
			object.__setattr__(self, 'user_data_dir', temp_dir)
		
		# Если используется storage_state и user_data_dir явно указан пользователем (не временный),
		# предупреждаем о потенциальном конфликте
		if has_storage_state and self.user_data_dir is not None:
			user_data_dir_str = str(self.user_data_dir)
			is_temp_dir = (
				'tmp' in user_data_dir_str.lower() or
				'agent-user-data-dir-' in user_data_dir_str
			)
			
			# Предупреждаем только если user_data_dir явно указан пользователем (не временный)
			if not is_temp_dir:
				logger.warning(
					f'⚠️ BrowserSession(...) was passed both storage_state AND user_data_dir. storage_state={self.storage_state} will forcibly overwrite '
					f'cookies/localStorage/sessionStorage in user_data_dir={self.user_data_dir}. '
					f'For multiple browsers in parallel, use only storage_state with user_data_dir=None, '
					f'or use a separate user_data_dir for each browser and set storage_state=None.'
				)
		
		return self

	@model_validator(mode='after')
	def warn_user_data_dir_non_default_version(self) -> Self:
		"""
		If user is using default profile dir with a non-default channel, force-change it
		to avoid corrupting the default data dir created with a different channel.
		"""

		is_not_using_default_chromium = self.executable_path or self.channel not in (AGENT_DEFAULT_CHANNEL, None)
		if self.user_data_dir == CONFIG.AGENT_DEFAULT_USER_DATA_DIR and is_not_using_default_chromium:
			alternate_name = (
				Path(self.executable_path).name.lower().replace(' ', '-')
				if self.executable_path
				else self.channel.name.lower()
				if self.channel
				else 'None'
			)
			logger.warning(
				f'⚠️ {self} Changing user_data_dir= {_log_pretty_path(self.user_data_dir)} ➡️ .../default-{alternate_name} to avoid {alternate_name.upper()} corruping default profile created by {AGENT_DEFAULT_CHANNEL.name}'
			)
			self.user_data_dir = CONFIG.AGENT_DEFAULT_USER_DATA_DIR.parent / f'default-{alternate_name}'
		return self

	@model_validator(mode='after')
	def warn_deterministic_rendering_weirdness(self) -> Self:
		if self.deterministic_rendering:
			logger.warning(
				'⚠️ BrowserSession(deterministic_rendering=True) is NOT RECOMMENDED. It breaks many sites and increases chances of getting blocked by anti-bot systems. '
				'It hardcodes the JS random seed and forces browsers across Linux/Mac/Windows to use the same font rendering engine so that identical screenshots can be generated.'
			)
		return self

	@model_validator(mode='after')
	def validate_proxy_settings(self) -> Self:
		"""Обеспечивает согласованность конфигурации прокси."""
		if self.proxy and (self.proxy.bypass and not self.proxy.server):
			logger.warning('BrowserProfile.proxy.bypass provided but proxy has no server; bypass will be ignored.')
		return self

	@model_validator(mode='after')
	def validate_highlight_elements_conflict(self) -> Self:
		"""Обеспечивает, что highlight_elements и dom_highlight_elements не включены одновременно, приоритет у dom_highlight_elements."""
		if self.highlight_elements and self.dom_highlight_elements:
			logger.warning(
				'⚠️ Both highlight_elements and dom_highlight_elements are enabled. '
				'dom_highlight_elements takes priority. Setting highlight_elements=False.'
			)
			self.highlight_elements = False
		return self

	def model_post_init(self, __context: Any) -> None:
		"""Вызывается после инициализации модели для настройки конфигурации отображения."""
		self.detect_display_configuration()
		self._copy_profile()

	def _copy_profile(self) -> None:
		"""Копирует профиль во временную директорию, если user_data_dir не None и ещё не является временной директорией."""
		if self.user_data_dir is None:
			return

		user_data_str = str(self.user_data_dir)
		if 'agent-user-data-dir-' in user_data_str.lower():
			# Уже используем временную директорию, копировать не нужно
			return

		is_chrome = (
			'chrome' in user_data_str.lower()
			or ('chrome' in str(self.executable_path).lower())
			or self.channel
			in (BrowserChannel.CHROME, BrowserChannel.CHROME_BETA, BrowserChannel.CHROME_DEV, BrowserChannel.CHROME_CANARY)
		)

		if not is_chrome:
			return

		temp_dir = tempfile.mkdtemp(prefix='agent-user-data-dir-')
		path_original_user_data = Path(self.user_data_dir)
		path_original_profile = path_original_user_data / self.profile_directory
		path_temp_profile = Path(temp_dir) / self.profile_directory

		if path_original_profile.exists():
			import shutil

			shutil.copytree(path_original_profile, path_temp_profile)
			local_state_src = path_original_user_data / 'Local State'
			local_state_dst = Path(temp_dir) / 'Local State'
			if local_state_src.exists():
				shutil.copy(local_state_src, local_state_dst)
			logger.info(f'Copied profile ({self.profile_directory}) and Local State to temp directory: {temp_dir}')

		else:
			Path(temp_dir).mkdir(parents=True, exist_ok=True)
			path_temp_profile.mkdir(parents=True, exist_ok=True)
			logger.info(f'Created new profile ({self.profile_directory}) in temp directory: {temp_dir}')

		self.user_data_dir = temp_dir

	def get_args(self) -> list[str]:
		"""Получает список всех аргументов командной строки Chrome для этого профиля (собран из значений по умолчанию, пользовательских и системных)."""

		if isinstance(self.ignore_default_args, list):
			default_args = set(CHROME_DEFAULT_ARGS) - set(self.ignore_default_args)
		elif self.ignore_default_args is True:
			default_args = []
		elif not self.ignore_default_args:
			default_args = CHROME_DEFAULT_ARGS

		assert self.user_data_dir is not None, 'user_data_dir must be set to a non-default path'

		# Сохраняем аргументы до преобразования для логирования
		pre_conversion_args = [
			*default_args,
			*self.args,
			f'--user-data-dir={self.user_data_dir}',
			f'--profile-directory={self.profile_directory}',
			*(CHROME_DOCKER_ARGS if (CONFIG.IN_DOCKER or not self.chromium_sandbox) else []),
			*(CHROME_HEADLESS_ARGS if self.headless else []),
			*(CHROME_DISABLE_SECURITY_ARGS if self.disable_security else []),
			*(CHROME_DETERMINISTIC_RENDERING_ARGS if self.deterministic_rendering else []),
			*(
				[f'--window-size={self.window_size["width"]},{self.window_size["height"]}']
				if self.window_size
				else (['--start-maximized'] if not self.headless else [])
			),
			*(
				[f'--window-position={self.window_position["width"]},{self.window_position["height"]}']
				if self.window_position
				else []
			),
			*(self._get_extension_args() if self.enable_default_extensions else []),
		]

		# Флаги прокси
		proxy_server = self.proxy.server if self.proxy else None
		proxy_bypass = self.proxy.bypass if self.proxy else None

		if proxy_server:
			pre_conversion_args.append(f'--proxy-server={proxy_server}')
			if proxy_bypass:
				pre_conversion_args.append(f'--proxy-bypass-list={proxy_bypass}')

		# Флаг User-Agent
		if self.user_agent:
			pre_conversion_args.append(f'--user-agent={self.user_agent}')

		# Специальная обработка для --disable-features: объединяем значения вместо перезаписи
		# Это предотвращает поломку расширений при disable_security=True, сохраняя
		# как стандартные функции (включая связанные с расширениями), так и функции безопасности
		disable_features_values = []
		non_disable_features_args = []

		# Извлекаем и объединяем все значения --disable-features
		for arg in pre_conversion_args:
			if arg.startswith('--disable-features='):
				features = arg.split('=', 1)[1]
				disable_features_values.extend(features.split(','))
			else:
				non_disable_features_args.append(arg)

		# Удаляем дубликаты, сохраняя порядок
		if disable_features_values:
			unique_features = []
			seen = set()
			for feature in disable_features_values:
				feature = feature.strip()
				if feature and feature not in seen:
					unique_features.append(feature)
					seen.add(feature)

			# Добавляем объединённые disable-features обратно
			non_disable_features_args.append(f'--disable-features={",".join(unique_features)}')

		# convert to dict and back to dedupe and merge other duplicate args
		final_args_list = BrowserLaunchArgs.args_as_list(BrowserLaunchArgs.args_as_dict(non_disable_features_args))

		return final_args_list

	def _get_extension_args(self) -> list[str]:
		"""
		Получает аргументы Chrome для включения расширений по умолчанию.

		В упрощённой версии агента расширения (uBlock Origin, ClearURLs и т.п.) отключены,
		чтобы не тянуть лишние зависимости из интернета и не засорять логи предупреждениями.
		"""
		# Расширения отключены — не добавляем никаких extra-флагов
		return []

	def _ensure_default_extensions_downloaded(self) -> list[str]:
		"""
		Ensure default extensions are downloaded and cached locally.
		Returns list of paths to extension directories.
		"""

		# Определения расширений - оптимизированы для автоматизации и извлечения контента
		# Объединяет uBlock Origin (блокировка рекламы) + "I still don't care about cookies" (обработка баннеров cookie)
		extensions = [
			{
				'name': 'uBlock Origin',
				'id': 'cjpalhdlnbpafiamejdnhcphjbkeiagm',
				'url': 'https://clients2.google.com/service/update2/crx?response=redirect&prodversion=133&acceptformat=crx3&x=id%3Dcjpalhdlnbpafiamejdnhcphjbkeiagm%26uc',
			},
			{
				'name': "I still don't care about cookies",
				'id': 'edibdbjcniadpccecjdfdjjppcpchdlm',
				'url': 'https://clients2.google.com/service/update2/crx?response=redirect&prodversion=133&acceptformat=crx3&x=id%3Dedibdbjcniadpccecjdfdjjppcpchdlm%26uc',
			},
			{
				'name': 'ClearURLs',
				'id': 'lckanjgmijmafbedllaakclkaicjfmnk',
				'url': 'https://clients2.google.com/service/update2/crx?response=redirect&prodversion=133&acceptformat=crx3&x=id%3Dlckanjgmijmafbedllaakclkaicjfmnk%26uc',
			},
			{
				'name': 'Force Background Tab',
				'id': 'gidlfommnbibbmegmgajdbikelkdcmcl',
				'url': 'https://clients2.google.com/service/update2/crx?response=redirect&prodversion=133&acceptformat=crx3&x=id%3Dgidlfommnbibbmegmgajdbikelkdcmcl%26uc',
			},
			# {
			# 	'name': 'Captcha Solver: Auto captcha solving service',
			# 	'id': 'pgojnojmmhpofjgdmaebadhbocahppod',
			# 	'url': 'https://clients2.google.com/service/update2/crx?response=redirect&prodversion=130&acceptformat=crx3&x=id%3Dpgojnojmmhpofjgdmaebadhbocahppod%26uc',
			# },
			# Consent-O-Matic отключён - используем списки cookie из uBlock Origin для простоты
			# {
			# 	'name': 'Consent-O-Matic',
			# 	'id': 'mdjildafknihdffpkfmmpnpoiajfjnjd',
			# 	'url': 'https://clients2.google.com/service/update2/crx?response=redirect&prodversion=130&acceptformat=crx3&x=id%3Dmdjildafknihdffpkfmmpnpoiajfjnjd%26uc',
			# },
			# {
			# 	'name': 'Privacy | Protect Your Payments',
			# 	'id': 'hmgpakheknboplhmlicfkkgjipfabmhp',
			# 	'url': 'https://clients2.google.com/service/update2/crx?response=redirect&prodversion=130&acceptformat=crx3&x=id%3Dhmgpakheknboplhmlicfkkgjipfabmhp%26uc',
			# },
		]

		# Создаём директорию кеша расширений
		cache_dir = CONFIG.AGENT_EXTENSIONS_DIR
		cache_dir.mkdir(parents=True, exist_ok=True)
		# logger.debug(f'📁 Extensions cache directory: {_log_pretty_path(cache_dir)}')

		extension_paths = []
		loaded_extension_names = []

		for ext in extensions:
			ext_dir = cache_dir / ext['id']
			crx_file = cache_dir / f'{ext["id"]}.crx'

			# Проверяем, извлечено ли расширение уже
			if ext_dir.exists() and (ext_dir / 'manifest.json').exists():
				# logger.debug(f'✅ Using cached {ext["name"]} extension from {_log_pretty_path(ext_dir)}')
				extension_paths.append(str(ext_dir))
				loaded_extension_names.append(ext['name'])
				continue

			try:
				# Скачиваем расширение, если не закешировано
				if not crx_file.exists():
					logger.info(f'📦 Downloading {ext["name"]} extension...')
					self._download_extension(ext['url'], crx_file)
				else:
					logger.debug(f'📦 Found cached {ext["name"]} .crx file')

				# Извлекаем расширение
				logger.info(f'📂 Extracting {ext["name"]} extension...')
				self._extract_extension(crx_file, ext_dir)

				extension_paths.append(str(ext_dir))
				loaded_extension_names.append(ext['name'])

			except Exception as e:
				logger.warning(f'⚠️ Failed to setup {ext["name"]} extension: {e}')
				continue

		# Применяем минимальный патч к расширению cookie с настраиваемым белым списком
		for i, path in enumerate(extension_paths):
			if loaded_extension_names[i] == "I still don't care about cookies":
				self._apply_minimal_extension_patch(Path(path), self.cookie_whitelist_domains)

		if extension_paths:
			logger.debug(f'[BrowserProfile] 🧩 Extensions loaded ({len(extension_paths)}): [{", ".join(loaded_extension_names)}]')
		else:
			logger.warning('[BrowserProfile] ⚠️ No default extensions could be loaded')

		return extension_paths

	def _apply_minimal_extension_patch(self, ext_dir: Path, whitelist_domains: list[str]) -> None:
		"""Минимальный патч: предзаполняем chrome.storage.local настраиваемым белым списком доменов."""
		try:
			bg_path = ext_dir / 'data' / 'background.js'
			if not bg_path.exists():
				return

			with open(bg_path, encoding='utf-8') as f:
				content = f.read()

			# Создаём объект доменов из белого списка для JavaScript с правильными отступами
			whitelist_entries = [f'        "{domain}": true' for domain in whitelist_domains]
			whitelist_js = '{\n' + ',\n'.join(whitelist_entries) + '\n      }'

			# Находим функцию initialize() и вставляем настройку хранилища перед updateSettings()
			# Реальная функция использует отступы в 2 пробела, не табы
			old_init = """async function initialize(checkInitialized, magic) {
  if (checkInitialized && initialized) {
    return;
  }
  loadCachedRules();
  await updateSettings();
  await recreateTabList(magic);
  initialized = true;
}"""

			# Новая функция с настраиваемой инициализацией белого списка
			new_init = f"""// Pre-populate storage with configurable domain whitelist if empty
async function ensureWhitelistStorage() {{
  const result = await chrome.storage.local.get({{ settings: null }});
  if (!result.settings) {{
    const defaultSettings = {{
      statusIndicators: true,
      whitelistedDomains: {whitelist_js}
    }};
    await chrome.storage.local.set({{ settings: defaultSettings }});
  }}
}}

async function initialize(checkInitialized, magic) {{
  if (checkInitialized && initialized) {{
    return;
  }}
  loadCachedRules();
  await ensureWhitelistStorage(); // Add storage initialization
  await updateSettings();
  await recreateTabList(magic);
  initialized = true;
}}"""

			if old_init in content:
				content = content.replace(old_init, new_init)

				with open(bg_path, 'w', encoding='utf-8') as f:
					f.write(content)

				domain_list = ', '.join(whitelist_domains)
				logger.info(f'[BrowserProfile] ✅ Cookie extension: {domain_list} pre-populated in storage')
			else:
				logger.debug('[BrowserProfile] Initialize function not found for patching')

		except Exception as e:
			logger.debug(f'[BrowserProfile] Could not patch extension storage: {e}')

	def _download_extension(self, url: str, output_path: Path) -> None:
		"""Скачивает файл расширения .crx."""
		import urllib.request

		try:
			with urllib.request.urlopen(url) as response:
				with open(output_path, 'wb') as f:
					f.write(response.read())
		except Exception as e:
			raise Exception(f'Failed to download extension: {e}')

	def _extract_extension(self, crx_path: Path, extract_dir: Path) -> None:
		"""Извлекает файл .crx в директорию."""
		import os
		import zipfile

		# Удаляем существующую директорию
		if extract_dir.exists():
			import shutil

			shutil.rmtree(extract_dir)

		extract_dir.mkdir(parents=True, exist_ok=True)

		try:
			# Файлы CRX - это ZIP-файлы с заголовком, пробуем извлечь как ZIP
			with zipfile.ZipFile(crx_path, 'r') as zip_ref:
				zip_ref.extractall(extract_dir)

			# Проверяем, что манифест существует
			if not (extract_dir / 'manifest.json').exists():
				raise Exception('No manifest.json found in extension')

		except zipfile.BadZipFile:
			# Файлы CRX имеют заголовок перед ZIP-данными
			# Пропускаем заголовок CRX и извлекаем ZIP-часть
			with open(crx_path, 'rb') as f:
				# Читаем заголовок CRX, чтобы найти начало ZIP
				magic = f.read(4)
				if magic != b'Cr24':
					raise Exception('Invalid CRX file format')

				version = int.from_bytes(f.read(4), 'little')
				if version == 2:
					pubkey_len = int.from_bytes(f.read(4), 'little')
					sig_len = int.from_bytes(f.read(4), 'little')
					f.seek(16 + pubkey_len + sig_len)  # Переходим к ZIP-данным
				elif version == 3:
					header_len = int.from_bytes(f.read(4), 'little')
					f.seek(12 + header_len)  # Переходим к ZIP-данным

				# Извлекаем ZIP-данные
				zip_data = f.read()

			# Записываем ZIP-данные во временный файл и извлекаем
			import tempfile

			with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
				temp_zip.write(zip_data)
				temp_zip.flush()

				with zipfile.ZipFile(temp_zip.name, 'r') as zip_ref:
					zip_ref.extractall(extract_dir)

				os.unlink(temp_zip.name)

	def detect_display_configuration(self) -> None:
		"""
		Detect the system display size and initialize the display-related config defaults:
		        screen, window_size, window_position, viewport, no_viewport, device_scale_factor
		"""

		display_size = get_display_size()
		has_screen_available = bool(display_size)
		self.screen = self.screen or display_size or ViewportSize(width=1920, height=1080)

		# if no headless preference specified, prefer headful if there is a display available
		if self.headless is None:
			self.headless = not has_screen_available

		# Определяем поведение viewport на основе режима и пользовательских настроек
		user_provided_viewport = self.viewport is not None

		if self.headless:
			# Режим headless: всегда используем viewport для контроля размера контента
			self.viewport = self.viewport or self.window_size or self.screen
			self.window_position = None
			self.window_size = None
			self.no_viewport = False
		else:
			# Режим headful: учитываем предпочтения пользователя по viewport
			self.window_size = self.window_size or self.screen

			if user_provided_viewport:
				# Пользователь явно установил viewport - включаем режим viewport
				self.no_viewport = False
			else:
				# По умолчанию в headful: контент подстраивается под окно (без viewport)
				self.no_viewport = True if self.no_viewport is None else self.no_viewport

		# Обрабатываем особые требования (device_scale_factor принудительно включает режим viewport)
		if self.device_scale_factor and self.no_viewport is None:
			self.no_viewport = False

		# Завершаем конфигурацию
		if self.no_viewport:
			# Режим без viewport: контент адаптируется под окно
			self.viewport = None
			self.device_scale_factor = None
			self.screen = None
			assert self.viewport is None
			assert self.no_viewport is True
		else:
			# Режим viewport: убеждаемся, что viewport установлен
			self.viewport = self.viewport or self.screen
			self.device_scale_factor = self.device_scale_factor or 1.0
			assert self.viewport is not None
			assert self.no_viewport is False

		assert not (self.headless and self.no_viewport), 'headless=True and no_viewport=True cannot both be set at the same time'
