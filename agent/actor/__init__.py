"""Высокоуровневая обёртка над CDP (Chrome DevTools Protocol) для удобной работы с элементами и страницей."""

from .element import Element
from .mouse import Mouse
from .page import Page
from .utils import Utils

__all__ = ['Page', 'Element', 'Mouse', 'Utils']
