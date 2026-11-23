"""Public fingerprints of the code and prompts loaded by this server process."""

import hashlib
import os

DEBUG_PROFILE = 'debug'
FINAL_REPORT_PROFILE = 'final_report'
REPORT_MODEL = {'model': 'deepseek-flash', 'base_url': 'https://api.deepseek.com/v1'}


def validate_model_profile(value):
	if value is None:
		return DEBUG_PROFILE
	if value not in (DEBUG_PROFILE, FINAL_REPORT_PROFILE):
		raise ValueError('model_profile must be debug or final_report')
	return value


def report_text_config():
	"""Load the explicit report service without falling back to any debug credential."""
	if (os.getenv('REPORT_TEXT_MODEL_BASE_URL') or REPORT_MODEL['base_url']).rstrip('/') != REPORT_MODEL['base_url']:
		raise ValueError('REPORT_TEXT_MODEL_BASE_URL must be https://api.deepseek.com/v1')
	if (os.getenv('REPORT_TEXT_MODEL') or REPORT_MODEL['model']) != REPORT_MODEL['model']:
		raise ValueError('REPORT_TEXT_MODEL must be deepseek-flash')
	if (os.getenv('REPORT_TEXT_MODEL_REASONING') or 'disabled') != 'disabled':
		raise ValueError('REPORT_TEXT_MODEL_REASONING must be disabled')
	key = os.getenv('REPORT_TEXT_MODEL_API_KEY', '').strip()
	if not key:
		raise ValueError('Final report needs REPORT_TEXT_MODEL_API_KEY')
	return {**REPORT_MODEL, 'api_key': key, 'reasoning': {'thinking': {'type': 'disabled'}}}


def redact_secrets(data, *keys):
	"""Scrub a known credential even if an upstream diagnostic echoes it."""
	secrets = tuple(key for key in keys if isinstance(key, str) and key)
	if not secrets:
		return data

	def clean(value):
		if isinstance(value, str):
			for secret in secrets:
				value = value.replace(secret, '[REDACTED]')
		elif isinstance(value, dict):
			return {clean(k): clean(v) for k, v in value.items()}
		elif isinstance(value, list):
			return [clean(v) for v in value]
		return value

	return clean(data)


def runtime_info():
	# Import at call time so Agent/Planner initialization does not introduce a cycle.
	from .browser import READ_STATE
	from .planner import PROMPT
	from .questions import NESTED_CHOICE, NEXT_ACTION, SUBGOAL_ACTION, SUBGOAL_TARGET, TARGET

	def digest(value):
		return hashlib.sha256(value.encode('utf-8')).hexdigest()

	return {
		'process_id': os.getpid(),
		'observation_sha256': digest(READ_STATE),
		'planner_prompt_sha256': digest(PROMPT),
		'decision_rules_sha256': digest('\n'.join((NEXT_ACTION, TARGET, SUBGOAL_ACTION, SUBGOAL_TARGET, NESTED_CHOICE))),
	}
