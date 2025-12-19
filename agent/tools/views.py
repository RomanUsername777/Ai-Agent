from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field


# Action Input Models
class ExtractAction(BaseModel):
	query: str
	extract_links: bool = Field(
		default=False, description='Set True to true if the query requires links, else false to safe tokens'
	)
	start_from_char: int = Field(
		default=0, description='Use this for long markdowns to start from a specific character (not index in browser_state)'
	)


class NavigateAction(BaseModel):
	url: str
	new_tab: bool = Field(default=False)


# Backward compatibility alias
GoToUrlAction = NavigateAction


class ClickElementAction(BaseModel):
	index: int | None = Field(default=None, ge=0, description='Element index from browser_state (0-based indexing)')
	coordinate_x: int | None = Field(default=None, description='Horizontal coordinate relative to viewport left edge')
	coordinate_y: int | None = Field(default=None, description='Vertical coordinate relative to viewport top edge')
	# expect_download: bool = Field(default=False, description='set True if expecting a download, False otherwise')  # moved to downloads_watchdog.py
	# click_count: int = 1  # TODO


class InputTextAction(BaseModel):
	index: int = Field(ge=0, description='from browser_state')
	text: str
	clear: bool = Field(default=True, description='1=clear, 0=append')
	press_enter: bool = Field(default=False, description='If True, press Enter after typing (useful for search fields)')


class DoneAction(BaseModel):
	text: str = Field(description='Final user message in the format the user requested')
	success: bool = Field(default=True, description='True if user_request completed successfully')
	files_to_display: list[str] | None = Field(default=[])


T = TypeVar('T', bound=BaseModel)


class StructuredOutputAction(BaseModel, Generic[T]):
	success: bool = Field(default=True, description='True if user_request completed successfully')
	data: T = Field(description='The actual output data matching the requested schema')


class ScrollAction(BaseModel):
	down: bool = Field(default=True, description='down=True=scroll down, down=False scroll up')
	pages: float = Field(default=1.0, description='0.5=half page, 1=full page, 10=to bottom/top')
	index: int | None = Field(default=None, description='Optional element index to scroll within specific container')


class SendKeysAction(BaseModel):
	keys: str = Field(description='keys (Escape, Enter, PageDown) or shortcuts (Control+o)')


class NoParamsAction(BaseModel):
	model_config = ConfigDict(extra='ignore')


class GetDropdownOptionsAction(BaseModel):
	index: int


class SelectDropdownOptionAction(BaseModel):
	index: int
	text: str = Field(description='exact text/value')


class RequestUserInputAction(BaseModel):
	prompt: str = Field(description='Сообщение для пользователя с запросом на действие (например, решение капчи)')


class ClickTextAction(BaseModel):
	text: str = Field(description='Visible text to click on (e.g., "Откликнуться", "Submit", "Login")')
	exact: bool = Field(default=False, description='If True, match text exactly; if False, match substring')


class ClickRoleAction(BaseModel):
	role: str = Field(default='button', description='ARIA role: button, link, menuitem, checkbox, radio')
	name: str = Field(default='', description='Accessible name/text of the element')
	exact: bool = Field(default=False, description='If True, match name exactly')


class WaitForUserInputAction(BaseModel):
	message: str | None = Field(
		default=None,
		description='Опциональное сообщение для пользователя. Если не указано, используется сообщение по умолчанию.'
	)
