"""Bounded, source-qualified execution evidence shared by planning and task output."""

import json
from copy import deepcopy

from .model import action_space, direct_actions
from .questions import MAX_STEPS


def document_key(page):
	"""A reload creates a new document even when its URL is unchanged."""
	marker = page.get('page_key') or page.get('marker') or []
	return page.get('url'), marker[0] if marker else None


def native_states(page):
	"""Compare only previously observed controls; never expose unseen form values."""
	key = page.get('page_key') or []
	rows = key[6] if len(key) > 6 and isinstance(key[6], list) else []
	return {row[0]: row[1:] for row in rows if isinstance(row, list) and len(row) == 6}


def page_content(page):
	"""Share indexed visible content without duplicating the same unindexed text; retain legacy evidence."""
	content = page.get('content')
	if isinstance(content, str) and content:
		return {'content': content[:6000]}
	return {'text': page.get('text', '')[:1500]}


def action_history(state):
	return [
		{
			k: deepcopy(h[k])
			for k in (
				'step',
				'operation',
				'kind',
				'action',
				'node',
				'text',
				'previous_value',
				'previous_checked',
				'page_changed',
				'url',
				'source_url',
				'source_document',
				'context',
			)
			if k in h
		}
		for h in state['history'][-MAX_STEPS:]
	]


def observed_elements(page, history):
	elements = action_space(page['actions'], include_node_refs=True)[0]
	for element in elements:
		element['direct_actions'] = direct_actions(page, element['node_ref'], history)
	return elements


def observation_history(state, *, current_document_only=True):
	"""Keep the latest raw observation per region, without inferring completion."""
	current_document = document_key(state['page'])
	current_states = native_states(state['page'])
	observations = {}
	for observation in state.get('observations', []):
		page = observation.get('page', {})
		source_document = document_key(page)
		if current_document_only and source_document != current_document:
			continue
		observed_states = native_states(page)
		after_step = observation.get('step', 0)
		prior_actions = [h for h in state['history'] if isinstance(h.get('step'), int) and h['step'] <= after_step]
		controls = {}
		for action in page.get('actions', []):
			if not (
				action.get('kind') in {'fill', 'select'}
				or (action.get('role') in {'textbox', 'combobox'} and 'value' in action)
				or any(k in action for k in ('checked', 'selected', 'expanded'))
			):
				continue
			node = action.get('node')
			control = {
				k: deepcopy(action[k])
				for k in ('node', 'role', 'value', 'checked', 'selected', 'expanded', 'context')
				if k in action
			}
			control['label'] = action.get('label', '').split(' → ')[0][:200]
			if action.get('kind') == 'select':
				control['value'] = action.get('current_value', '')
			previous = observed_states.get(node)
			control['state_unchanged'] = (
				previous == current_states.get(node) if previous is not None and source_document == current_document else None
			)
			control['direct_actions'] = direct_actions(page, node, prior_actions)
			controls[node] = control
		record = {
			'source_url': source_document[0],
			'source_document': source_document[1],
			'event_seq': observation.get('event_seq'),
			'after_step': after_step,
			'time': observation.get('time') or page.get('observed_at'),
			'scroll': deepcopy(page.get('scroll')),
			**page_content(page),
			'controls': list(controls.values())[:48],
		}
		if len(controls) > 48:
			record['omitted_controls'] = len(controls) - 48
		page_key = page.get('page_key') or []
		regions = page_key[7] if len(page_key) > 7 else []
		key = json.dumps(
			{
				'document': source_document,
				'scroll': page.get('scroll'),
				'w': page.get('w'),
				'h': page.get('h'),
				'regions': regions,
			},
			sort_keys=True,
			ensure_ascii=False,
		)
		if regions:
			record['scroll_regions'] = deepcopy(regions)
		observations.pop(key, None)
		observations[key] = record
	return list(observations.values())[-12:]


def prior_control_states(state, *, records=None):
	"""Return unchanged, previously observed results with execution evidence, never action targets."""
	visible = {action.get('node') for action in state['page'].get('actions', [])}
	latest = {}
	for record in observation_history(state) if records is None else records:
		for control in record['controls']:
			node = control.get('node')
			latest.pop(node, None)
			latest[node] = (record, control)
	results = []
	for node, (record, control) in latest.items():
		if (
			node in visible
			or record['source_document'] is None
			or control['state_unchanged'] is not True
			or not control['direct_actions']
		):
			continue
		results.append(
			{
				'node_ref': node,
				'source_url': record['source_url'],
				'source_document': record['source_document'],
				**{
					k: deepcopy(control[k])
					for k in ('label', 'role', 'value', 'checked', 'selected', 'context', 'direct_actions')
					if k in control
				},
			}
		)
	return results[-30:]
