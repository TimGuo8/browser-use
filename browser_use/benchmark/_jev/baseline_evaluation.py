"""Frozen, read-only final evaluation shared by every baseline task and archive replay."""

import hashlib
import json
from copy import deepcopy
from datetime import datetime

import httpx

from .evidence import document_key
from .model import ModelProviderError
from .research_evaluation import VERIFY_PROMPT, verification_page_evidence, verify_report

CONTEXT_VERSION = 1
PROMPT_VERSION = '2026-11-02.1'


def timestamp():
	return datetime.now().astimezone().isoformat(timespec='milliseconds')


def digest(value):
	return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def factual_evidence(value):
	"""Keep raw legacy observations, excluding old evaluator conclusions posing as facts."""
	if isinstance(value, list):
		return [factual_evidence(item) for item in value]
	if isinstance(value, dict):
		return {
			key: factual_evidence(item)
			for key, item in value.items()
			if key not in {'violations', 'checks', 'semantic_review', 'not_submitted'} and not key.endswith('_verified')
		}
	return deepcopy(value)


def build_context(state, samples, *, task_started_at, task_ended_at, execution_status, execution_reason='', execution_events=()):
	"""Preserve chronological samples, including changed rankings in the same viewport/document."""
	independent = []
	control_records, control_keys, sources, source_keys, texts, text_keys = {}, {}, {}, {}, {}, {}
	descriptions, description_keys = {}, {}

	def source_ref(source):
		if source not in source_keys:
			ident = f'document-{len(sources) + 1}'
			source_keys[source] = ident
			sources[ident] = {'source_url': source[0], 'source_document': source[1]}
		return source_keys[source]

	def page_texts(page, source):
		for field in ('text', 'content'):
			value = page.get(field)
			if not isinstance(value, str):
				continue
			key = source, value
			if key not in text_keys:
				ident = f'text-{len(texts) + 1}'
				text_keys[key] = ident
				texts[ident] = {'source_id': source_ref(source), 'text': value}
			page[field] = {'text_ref': text_keys[key]}
		return page

	def control_ref(control, source):
		# Geometry and execution guards cannot help a text-only evaluator; native/semantic identity remains.
		reading = {key: deepcopy(value) for key, value in control.items() if key not in {'rect', 'guard', 'id'}}

		def description_ref(description):
			content = {
				key: value for key, value in description.items() if key not in {'observed_at', 'source_url', 'source_document'}
			}
			content['source_id'] = source_ref((description.get('source_url'), description.get('source_document')))
			key = digest(content)
			if key not in description_keys:
				ident = f'description-{len(descriptions) + 1}'
				description_keys[key] = ident
				descriptions[ident] = {**content, 'first_observed_at': description.get('observed_at')}
			return {'description_ref': description_keys[key]}

		if isinstance(reading.get('observed_description'), dict):
			reading['observed_description'] = description_ref(reading['observed_description'])
		if isinstance(reading.get('description_history'), list):
			reading['description_history'] = [description_ref(item) for item in reading['description_history']]
		record = {'source_id': source_ref(source), 'reading': reading}
		key = digest(record)
		if key not in control_keys:
			ident = f'control-{len(control_records) + 1}'
			control_keys[key] = ident
			control_records[ident] = record
		return control_keys[key]

	for index, sample in enumerate(samples):
		evidence = verification_page_evidence(factual_evidence(sample.get('evidence', {})))
		page = evidence.get('sampled_page', {})
		source = page.get('url'), page.get('source_document')
		page_texts(page, source)
		form = evidence.get('form_state')
		if isinstance(form, dict) and isinstance(form.get('controls'), list):
			source = form.get('source_url'), form.get('source_document')
			readings = []
			for control in form['controls']:
				# Earlier samples already retain these descriptions with their own timestamps.
				reading = {
					key: value
					for key, value in control.items()
					if index == len(samples) - 1 or key not in {'description_history', 'omitted_descriptions'}
				}
				readings.append(control_ref(reading, source))
			form['controls'] = readings
		if isinstance(evidence.get('observed_controls'), list):
			evidence['observed_controls'] = [control_ref(control, source) for control in evidence['observed_controls']]
		independent.append(
			{
				'sample_id': f'sample-{index + 1}',
				**{
					key: deepcopy(sample[key])
					for key in ('step', 'time', 'event_seq', 'phase', 'sample_status', 'reason')
					if key in sample
				},
				'evidence': evidence,
			}
		)
	history = []
	for index, observation in enumerate(state.get('observations', [])):
		page = observation.get('page', {})
		source_url, source_document = document_key(page)
		record = {
			'observation_id': f'observation-{index + 1}',
			'source_url': source_url,
			'source_document': source_document,
			**{key: deepcopy(observation[key]) for key in ('step', 'time', 'event_seq', 'phase') if key in observation},
			'page': {
				key: deepcopy(page[key])
				for key in ('url', 'title', 'text', 'content', 'actions', 'scroll', 'observed_at', 'truncation')
				if key in page
			},
		}
		# Reuse fields only from the matching, independently sampled observation, never by viewport/URL alone.
		if index < len(samples):
			sample = samples[index]
			evidence = sample.get('evidence', {})
			sampled = evidence.get('sampled_page', {})
			same_event = (
				sample.get('event_seq') == observation.get('event_seq')
				if sample.get('event_seq') is not None
				else len(samples) == len(state.get('observations', []))
				and (sample.get('step'), sample.get('phase')) == (observation.get('step'), observation.get('phase'))
			)
			if (
				same_event
				and source_document is not None
				and (source_url, source_document) == (sampled.get('url'), sampled.get('source_document'))
			):
				fields = {**sampled}
				if 'observed_controls' in evidence:
					fields['actions'] = evidence['observed_controls']
				record['independent_sample_id'] = independent[index]['sample_id']
				record['page'] = {key: value for key, value in record['page'].items() if fields.get(key) != value}
		if isinstance(record['page'].get('actions'), list):
			record['page']['actions'] = [
				control_ref(control, (source_url, source_document)) for control in record['page']['actions']
			]
		page_texts(record['page'], (source_url, source_document))
		history.append(record)

	def execution_action(action):
		# Retain operations, identities, input and execution results, excluding model-cost/loop telemetry.
		return {
			key: deepcopy(value)
			for key, value in action.items()
			if key not in {'repeat_key', 'usage', 'probability', 'confidence', 'latency_ms', 'text_helper', 'text_latency_ms'}
		}

	events = []
	for event in execution_events:
		record = deepcopy(event)
		if isinstance(record.get('action'), dict):
			record['action'] = execution_action(record['action'])
		events.append(record)
	return {
		'goal': state['goal'],
		'task_started_at': task_started_at,
		'task_ended_at': task_ended_at,
		'execution_status': execution_status,
		'execution_reason': execution_reason,
		'task_output': deepcopy(state.get('result')),
		'collected_pages': deepcopy(state.get('research', {}).get('pages', [])),
		'independent_evidence': {
			'samples': independent,
			'control_records': control_records,
			'sources': sources,
			'texts': texts,
			'descriptions': descriptions,
			'final_sample_id': independent[-1]['sample_id'] if independent else None,
		},
		'observation_history': history,
		'actions': [execution_action(action) for action in state.get('history', [])],
		'execution_events': events,
		'evidence_limits': {
			'images_sent': False,
			'executor_snapshot': 'Visible text is capped at 6000 characters; labels/context may be shortened upstream. '
			'Only recorded text is available, not the complete page or image pixels.',
			'sampling': 'Sampler truncation and form-description history omissions are recorded with each sample; '
			'legacy archives may lack truncation metadata. No evaluation-history region limit is applied.',
		},
	}


def freeze_context(context, *, reconstructed=False):
	context = deepcopy(context)
	return {
		'schema_version': CONTEXT_VERSION,
		'context_id': digest(context),
		'created_at': timestamp(),
		'prompt_version': PROMPT_VERSION,
		'prompt_sha256': hashlib.sha256(VERIFY_PROMPT.encode()).hexdigest(),
		'reconstructed_from_legacy': reconstructed,
		'context': context,
	}


def validate_frozen(frozen):
	if (
		frozen.get('schema_version') != CONTEXT_VERSION
		or not isinstance(frozen.get('context'), dict)
		or frozen.get('context_id') != digest(frozen['context'])
	):
		raise ValueError('Invalid or modified frozen evaluation context')


def error_status(error):
	"""Only transient transport/provider failures are retryable; never retry a received verdict."""
	if isinstance(error, ModelProviderError):
		return 'transport_error', error.status_code in {408, 429} or error.status_code >= 500
	if isinstance(error, (httpx.TransportError, TimeoutError, ConnectionError)):
		return 'transport_error', True
	if str(error).startswith('Model connection failed'):
		return 'transport_error', True
	if isinstance(error, (ValueError, TypeError, KeyError)):
		return ('transport_error', False) if 'API_KEY' in str(error) else ('invalid_response', False)
	return 'transport_error', False


def evaluate_frozen(frozen, config, call_model, *, should_cancel=lambda: False):
	"""At most two identical, text-only requests. No execution-budget checks or browser access."""
	validate_frozen(frozen)
	result = {
		'method': 'llm',
		'evaluation_status': 'not_run',
		'reason': '尚未调用评估模型',
		'context_id': frozen['context_id'],
		'model': config.get('model'),
		'prompt_version': PROMPT_VERSION,
		'prompt_sha256': hashlib.sha256(VERIFY_PROMPT.encode()).hexdigest(),
		'started_at': timestamp(),
		'attempts': 0,
	}
	for attempt in range(1, 3):
		if should_cancel():
			if not result['attempts']:
				result['reason'] = '用户取消，未追加评估调用'
			break
		result['attempts'] = attempt
		try:
			reply = call_model(
				'verify',
				verify_report,
				config,
				deepcopy(frozen['context']),
				evaluation_context_id=frozen['context_id'],
				evaluation_attempt=attempt,
				evaluation_prompt_version=PROMPT_VERSION,
			)
			result.update(
				evaluation_status='completed',
				verdict=reply['verdict'],
				outcome=reply['verdict'],
				reason=reply['reason'],
				semantic_review=reply,
			)
			result.pop('error_type', None)
			break
		except Exception as error:
			if getattr(error, 'is_cancellation', False):
				result.update(evaluation_status='not_run', reason='用户取消，未追加评估调用')
				break
			status, retryable = error_status(error)
			result.update(evaluation_status=status, reason=f'评估调用失败：{error}', error_type=type(error).__name__)
			if not retryable:
				break
	result['ended_at'] = timestamp()
	return result


def attempt_status(execution_status, evaluation):
	if execution_status not in {'done', 'blocked'}:
		return execution_status
	status = evaluation['evaluation_status']
	if status == 'completed':
		return {'met': 'passed', 'not_met': 'failed', 'unknown': 'evidence_insufficient'}[evaluation['verdict']]
	return 'evaluation_error' if status in {'transport_error', 'invalid_response'} else 'not_evaluated'


def outcome_statistics(attempts):
	"""All scheduled attempts stay in the denominator; old unknowns retain their legacy meaning."""
	counts = dict(attempts=len(attempts), met=0, not_met=0, unknown=0, evaluation_errors=0, not_run=0, legacy=0, legacy_unknown=0)
	for row in attempts:
		status = row.get('evaluation_status')
		final = (row.get('evaluation') or {}).get('final', {}) if isinstance(row.get('evaluation'), dict) else {}
		if status == 'completed':
			counts[final['verdict']] += 1
		elif status in {'transport_error', 'invalid_response'}:
			counts['evaluation_errors'] += 1
		elif status is not None:
			counts['not_run'] += 1
		else:
			counts['legacy'] += 1
			if row.get('status') == 'passed':
				counts['met'] += 1
			elif row.get('status') == 'failed':
				counts['not_met'] += 1
			else:
				counts['legacy_unknown'] += 1
	judged = counts['met'] + counts['not_met'] + counts['unknown']
	return {
		**counts,
		'success_rate': counts['met'] / len(attempts) if attempts else None,
		'success_rate_denominator': len(attempts),
		'judged_success_rate': counts['met'] / judged if judged else None,
		'judged_denominator': judged,
		'unjudged_rate': (len(attempts) - judged) / len(attempts) if attempts else None,
	}
