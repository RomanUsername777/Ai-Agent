# @file purpose: Concise evaluation serializer for DOM trees - optimized for LLM query writing

import logging

from agent.dom.utils import cap_text_length
from agent.dom.views import (
	EnhancedDOMTreeNode,
	NodeType,
	SimplifiedNode,
)

logger = logging.getLogger(__name__)

# Critical attributes for query writing and form interaction
# NOTE: Removed 'id' and 'class' to force more robust structural selectors
EVAL_KEY_ATTRIBUTES = [
	'id',  # Removed - can have special chars, forces structural selectors
	'class',  # Removed - can have special chars like +, forces structural selectors
	'name',
	'type',
	'placeholder',
	'aria-label',
	'role',
	'value',
	'data-qa',  # ВОССТАНОВЛЕНО: data-qa критически важен для идентификации кнопок на hh.ru
	# 'href',
	'data-testid',
	'alt',  # for images
	'title',  # useful for tooltips/link context
	# State attributes (critical for form interaction)
	'checked',
	'selected',
	'disabled',
	'required',
	'readonly',
	# ARIA states
	'aria-expanded',
	'aria-pressed',
	'aria-checked',
	'aria-selected',
	'aria-invalid',
	# Validation attributes (help agents avoid brute force)
	'pattern',
	'min',
	'max',
	'minlength',
	'maxlength',
	'step',
	'aria-valuemin',
	'aria-valuemax',
	'aria-valuenow',
]

# Semantic elements that should always be shown
SEMANTIC_ELEMENTS = {
	'html',  # Always show document root
	'body',  # Always show body
	'h1',
	'h2',
	'h3',
	'h4',
	'h5',
	'h6',
	'a',
	'button',
	'input',
	'textarea',
	'select',
	'form',
	'label',
	'nav',
	'header',
	'footer',
	'main',
	'article',
	'section',
	'table',
	'thead',
	'tbody',
	'tr',
	'th',
	'td',
	'ul',
	'ol',
	'li',
	'img',
	'iframe',
	'video',
	'audio',
}

# Container elements that can be collapsed if they only wrap one child
COLLAPSIBLE_CONTAINERS = {'div', 'span', 'section', 'article'}

# SVG child elements to skip (decorative only, no interaction value)
SVG_ELEMENTS = {
	'path',
	'rect',
	'g',
	'circle',
	'ellipse',
	'line',
	'polyline',
	'polygon',
	'use',
	'defs',
	'clipPath',
	'mask',
	'pattern',
	'image',
	'text',
	'tspan',
}


class DOMEvalSerializer:
	"""Ultra-concise DOM serializer for quick LLM query writing."""

	@staticmethod
	def serialize_tree(node: SimplifiedNode | None, include_attributes: list[str], depth: int = 0) -> str:
		"""
		Serialize complete DOM tree structure for LLM understanding.

		Strategy:
		- Show ALL elements to preserve DOM structure
		- Non-interactive elements show just tag name
		- Interactive elements show full attributes + [index]
		- Self-closing tags only (no closing tags)
		"""
		if not node:
			return ''

		# Skip excluded nodes but process children
		if hasattr(node, 'excluded_by_parent') and node.excluded_by_parent:
			return DOMEvalSerializer._serialize_children(node, include_attributes, depth)

		# Skip nodes marked as should_display=False
		if not node.should_display:
			return DOMEvalSerializer._serialize_children(node, include_attributes, depth)

		formatted_text = []
		depth_str = depth * '\t'

		if node.original_node.node_type == NodeType.ELEMENT_NODE:
			tag = node.original_node.tag_name.lower()
			is_visible = node.original_node.snapshot_node and node.original_node.is_visible

			# Container elements that should be shown even if invisible (might have visible children)
			container_tags = {'html', 'body', 'div', 'main', 'section', 'article', 'aside', 'header', 'footer', 'nav'}

			# CRITICAL: Always serialize interactive elements (in selector_map) even if invisible
			# This is essential for buttons in modals/overlays that Chrome considers clickable
			# but may not pass visibility checks (e.g., "Откликнуться" button on hh.ru)
			
			# DIAGNOSTIC: Log "Откликнуться" elements to track serialization
			element_text = node.original_node.get_all_children_text() if hasattr(node.original_node, 'get_all_children_text') else ''
			has_otkliknitesya = 'Откликнуться' in element_text
			if has_otkliknitesya:
				logger.warning(
					f'🔍 EVAL_SERIALIZER [_serialize_tree]: "Откликнуться" элемент: '
					f'is_visible={is_visible}, is_interactive={node.is_interactive}, tag={tag}, '
					f'backend_node_id={node.original_node.backend_node_id}, '
					f'will_skip={not is_visible and tag not in container_tags and tag not in ["iframe", "frame"] and not node.is_interactive}'
				)
			
			if not is_visible and tag not in container_tags and tag not in ['iframe', 'frame'] and not node.is_interactive:
				if has_otkliknitesya:
					logger.warning(
						f'❌ EVAL_SERIALIZER: "Откликнуться" ПРОПУЩЕН в сериализации (невидим и не интерактивен): '
						f'is_visible={is_visible}, is_interactive={node.is_interactive}, tag={tag}'
					)
				return DOMEvalSerializer._serialize_children(node, include_attributes, depth)
			
			if has_otkliknitesya:
				logger.warning(
					f'✅ EVAL_SERIALIZER: "Откликнуться" БУДЕТ сериализован: '
					f'is_visible={is_visible}, is_interactive={node.is_interactive}, tag={tag}, backend_node_id={node.original_node.backend_node_id}'
				)

			# Special handling for iframes - show them with their content
			if tag in ['iframe', 'frame']:
				return DOMEvalSerializer._serialize_iframe(node, include_attributes, depth)

			# Skip SVG elements entirely - they're just decorative graphics with no interaction value
			# Show the <svg> tag itself to indicate graphics, but don't recurse into children
			if tag == 'svg':
				line = f'{depth_str}'
				# Add [i_X] for interactive SVG elements only
				if node.is_interactive:
					line += f'[i_{node.original_node.backend_node_id}] '
				line += '<svg'
				attributes_str = DOMEvalSerializer._build_compact_attributes(node.original_node)
				if attributes_str:
					line += f' {attributes_str}'
				line += ' /> <!-- SVG content collapsed -->'
				return line

			# Skip SVG child elements entirely (path, rect, g, circle, etc.)
			if tag in SVG_ELEMENTS:
				return ''

			# Build compact attributes string
			attributes_str = DOMEvalSerializer._build_compact_attributes(node.original_node)

			# Decide if this element should be shown
			is_semantic = tag in SEMANTIC_ELEMENTS
			has_useful_attrs = bool(attributes_str)
			has_text_content = DOMEvalSerializer._has_direct_text(node)
			has_children = len(node.children) > 0

			# Build compact element representation
			line = f'{depth_str}'
			# Add backend node ID notation - [i_X] for interactive elements only
			if node.is_interactive:
				line += f'[i_{node.original_node.backend_node_id}] '
				# DIAGNOSTIC: Log when interactive element gets index notation
				if has_otkliknitesya:
					logger.warning(
						f'✅ EVAL_SERIALIZER: "Откликнуться" получил индекс [i_{node.original_node.backend_node_id}] в сериализации'
					)
			# Non-interactive elements don't get an index notation
			line += f'<{tag}'

			if attributes_str:
				line += f' {attributes_str}'

			# Add scroll info if element is scrollable
			if node.original_node.should_show_scroll_info:
				scroll_text = node.original_node.get_scroll_info_text()
				if scroll_text:
					line += f' scroll="{scroll_text}"'

			# Add inline text if present (keep it on same line for compactness)
			inline_text = DOMEvalSerializer._get_inline_text(node)

			# ВАЖНО: для кнопок и ссылок всегда показываем текст, даже если это контейнер
			# Это критично для кнопок типа "Откликнуться" на hh.ru, которые могут быть вложены в span
			is_button_or_link = tag in ('button', 'a') or (node.original_node.attributes and node.original_node.attributes.get('role') == 'button')
			
			# For containers (html, body, div, etc.), always show children even if there's inline text
			# For other elements, inline text replaces children (more compact)
			is_container = tag in container_tags

			if inline_text and (not is_container or is_button_or_link):
				line += f'>{inline_text}'
			else:
				line += ' />'

			formatted_text.append(line)

			# Process children (always for containers, only if no inline_text for others)
			if has_children and (is_container or not inline_text):
				children_text = DOMEvalSerializer._serialize_children(node, include_attributes, depth + 1)
				if children_text:
					formatted_text.append(children_text)

		elif node.original_node.node_type == NodeType.TEXT_NODE:
			# Text nodes are handled inline with their parent
			pass

		elif node.original_node.node_type == NodeType.DOCUMENT_FRAGMENT_NODE:
			# Shadow DOM - just show children directly with minimal marker
			if node.children:
				formatted_text.append(f'{depth_str}#shadow')
				children_text = DOMEvalSerializer._serialize_children(node, include_attributes, depth + 1)
				if children_text:
					formatted_text.append(children_text)

		return '\n'.join(formatted_text)

	@staticmethod
	def _serialize_children(node: SimplifiedNode, include_attributes: list[str], depth: int) -> str:
		"""Helper to serialize all children of a node."""
		children_output = []

		# Check if parent is a list container (ul, ol)
		is_list_container = node.original_node.node_type == NodeType.ELEMENT_NODE and node.original_node.tag_name.lower() in [
			'ul',
			'ol',
		]

		# Track list items and consecutive links
		li_count = 0
		max_list_items = 50
		consecutive_link_count = 0
		max_consecutive_links = 50
		total_links_skipped = 0

		for child in node.children:
			# Get tag name for this child
			current_tag = None
			if child.original_node.node_type == NodeType.ELEMENT_NODE:
				current_tag = child.original_node.tag_name.lower()

			# If we're in a list container and this child is an li element
			if is_list_container and current_tag == 'li':
				li_count += 1
				# Skip li elements after the 5th one
				if li_count > max_list_items:
					continue

			# Track consecutive anchor tags (links)
			if current_tag == 'a':
				consecutive_link_count += 1
				# Skip links after the 5th consecutive one
				if consecutive_link_count > max_consecutive_links:
					total_links_skipped += 1
					continue
			else:
				# Reset counter when we hit a non-link element
				# But first add truncation message if we skipped links
				if total_links_skipped > 0:
					depth_str = depth * '\t'
					children_output.append(f'{depth_str}... ({total_links_skipped} more links in this list)')
					total_links_skipped = 0
				consecutive_link_count = 0

			child_text = DOMEvalSerializer.serialize_tree(child, include_attributes, depth)
			if child_text:
				children_output.append(child_text)

		# Add truncation message if we skipped items at the end
		if is_list_container and li_count > max_list_items:
			depth_str = depth * '\t'
			children_output.append(
				f'{depth_str}... ({li_count - max_list_items} more items in this list (truncated) use evaluate to get more.'
			)

		# Add truncation message for links if we skipped any at the end
		if total_links_skipped > 0:
			depth_str = depth * '\t'
			children_output.append(
				f'{depth_str}... ({total_links_skipped} more links in this list) (truncated) use evaluate to get more.'
			)

		return '\n'.join(children_output)

	@staticmethod
	def _build_compact_attributes(node: EnhancedDOMTreeNode) -> str:
		"""Build ultra-compact attributes string with only key attributes."""
		attrs = []

		# Prioritize attributes that help with query writing
		if node.attributes:
			for attr in EVAL_KEY_ATTRIBUTES:
				if attr in node.attributes:
					value = str(node.attributes[attr]).strip()
					if not value:
						continue

					# Special handling for different attributes
					if attr == 'class':
						# For class, limit to first 2 classes to save space
						classes = value.split()[:3]
						value = ' '.join(classes)
					elif attr == 'href':
						# For href, cap at 20 chars to save space
						value = cap_text_length(value, 80)
					else:
						# Cap at 25 chars for other attributes
						value = cap_text_length(value, 80)

					attrs.append(f'{attr}="{value}"')

		# Note: We intentionally don't add role from ax_node here because:
		# 1. If role is explicitly set in HTML, it's already captured above via EVAL_KEY_ATTRIBUTES
		# 2. Inferred roles from AX tree (like link, listitem, LineBreak) are redundant with the tag name
		# 3. This reduces noise - <a href="..." role="link"> is redundant, we already know <a> is a link

		return ' '.join(attrs)

	@staticmethod
	def _has_direct_text(node: SimplifiedNode) -> bool:
		"""Check if node has direct text children (not nested in other elements)."""
		for child in node.children:
			if child.original_node.node_type == NodeType.TEXT_NODE:
				text = child.original_node.node_value.strip() if child.original_node.node_value else ''
				if len(text) > 1:
					return True
		return False

	@staticmethod
	def _get_inline_text(node: SimplifiedNode) -> str:
		"""Get text content to display inline (max 80 chars).
		
		Uses original_node.get_all_children_text() to get text from the ORIGINAL DOM,
		not from SimplifiedNode.children which may have filtered out nested spans.
		This is critical for buttons like hh.ru's <button><span><span><span>Откликнуться</span></span></span></button>
		"""
		# Use the original DOM node to get ALL nested text, bypassing SimplifiedNode filtering
		text = node.original_node.get_all_children_text().strip() if node.original_node else ''
		if not text or len(text) <= 1:
			return ''
		return cap_text_length(text, 80)

	@staticmethod
	def _serialize_iframe(node: SimplifiedNode, include_attributes: list[str], depth: int) -> str:
		"""Handle iframe serialization with content document."""
		formatted_text = []
		depth_str = depth * '\t'
		tag = node.original_node.tag_name.lower()

		# Build minimal iframe marker with key attributes
		attributes_str = DOMEvalSerializer._build_compact_attributes(node.original_node)
		line = f'{depth_str}<{tag}'
		if attributes_str:
			line += f' {attributes_str}'

		# Add scroll info for iframe content
		if node.original_node.should_show_scroll_info:
			scroll_text = node.original_node.get_scroll_info_text()
			if scroll_text:
				line += f' scroll="{scroll_text}"'

		line += ' />'
		formatted_text.append(line)

		# If iframe has content document, serialize its content
		if node.original_node.content_document:
			# Add marker for iframe content
			formatted_text.append(f'{depth_str}\t#iframe-content')

			# Process content document children
			for child_node in node.original_node.content_document.children_nodes or []:
				# Process html documents
				if child_node.tag_name.lower() == 'html':
					# Find and serialize body content only (skip head)
					for html_child in child_node.children:
						if html_child.tag_name.lower() == 'body':
							for body_child in html_child.children:
								# Recursively process body children (iframe content)
								DOMEvalSerializer._serialize_document_node(
									body_child, formatted_text, include_attributes, depth + 2, is_iframe_content=True
								)
							break  # Stop after processing body
				else:
					# Not an html element - serialize directly
					DOMEvalSerializer._serialize_document_node(
						child_node, formatted_text, include_attributes, depth + 1, is_iframe_content=True
					)

		return '\n'.join(formatted_text)

	@staticmethod
	def _serialize_document_node(
		dom_node: EnhancedDOMTreeNode,
		output: list[str],
		include_attributes: list[str],
		depth: int,
		is_iframe_content: bool = True,
	) -> None:
		"""Helper to serialize a document node without SimplifiedNode wrapper.

		Args:
			is_iframe_content: If True, be more permissive with visibility checks since
				iframe content might not have snapshot data from parent page.
		"""
		depth_str = depth * '\t'

		if dom_node.node_type == NodeType.ELEMENT_NODE:
			tag = dom_node.tag_name.lower()

			# For iframe content, be permissive - show all semantic elements even without snapshot data
			# For regular content, skip invisible elements
			if is_iframe_content:
				# Only skip if we have snapshot data AND it's explicitly invisible
				# If no snapshot data, assume visible (cross-origin iframe content)
				is_visible = (not dom_node.snapshot_node) or dom_node.is_visible
			else:
				# Regular strict visibility check
				is_visible = dom_node.snapshot_node and dom_node.is_visible

			if not is_visible:
				return

			# Check if semantic or has useful attributes
			is_semantic = tag in SEMANTIC_ELEMENTS
			attributes_str = DOMEvalSerializer._build_compact_attributes(dom_node)

			if not is_semantic and not attributes_str:
				# Skip but process children
				for child in dom_node.children:
					DOMEvalSerializer._serialize_document_node(
						child, output, include_attributes, depth, is_iframe_content=is_iframe_content
					)
				return

			# Build element line
			line = f'{depth_str}<{tag}'
			if attributes_str:
				line += f' {attributes_str}'

			# Get direct text content
			text_parts = []
			for child in dom_node.children:
				if child.node_type == NodeType.TEXT_NODE and child.node_value:
					text = child.node_value.strip()
					if text and len(text) > 1:
						text_parts.append(text)

			if text_parts:
				combined = ' '.join(text_parts)
				line += f'>{cap_text_length(combined, 100)}'
			else:
				line += ' />'

			output.append(line)

			# Process non-text children
			for child in dom_node.children:
				if child.node_type != NodeType.TEXT_NODE:
					DOMEvalSerializer._serialize_document_node(
						child, output, include_attributes, depth + 1, is_iframe_content=is_iframe_content
					)
