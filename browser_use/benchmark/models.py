"""Freeze the execution model and the independent evaluator for a run."""

import os

from ._jev.model import text_config
from ._jev.planner import planner_config, public_config
from ._jev.runtime import DEBUG_PROFILE, FINAL_REPORT_PROFILE, REPORT_MODEL, report_text_config, validate_model_profile


def model_profiles():
	debug_execution = planner_config()
	debug_text = text_config()
	execution_public = public_config(debug_execution)
	text_public = public_config(debug_text)
	try:
		report_text_config()
		report_ready = True
	except ValueError:
		report_ready = False
	return [
		{
			'id': DEBUG_PROFILE,
			'label': '调试',
			'configured': bool(debug_execution['api_key'] and debug_text['api_key']),
			'planner': {k: execution_public[k] for k in ('model', 'base_url')},
			'text': {k: text_public[k] for k in ('model', 'base_url')},
		},
		{
			'id': FINAL_REPORT_PROFILE,
			'label': '最终报告',
			'configured': report_ready,
			'planner': REPORT_MODEL,
			'text': REPORT_MODEL,
		},
	]


def system1_options(profiles):
	debug = profiles[0]['planner']
	return [
		{
			'provider': 'browser_use',
			'label': 'Browser Use',
			'model': debug['model'],
			'base_url': debug['base_url'],
			'configured': any(profile['configured'] for profile in profiles),
		},
	]


def runtime_settings(provider, mode, screenshot_mode=False, *, screenshot_policy=None, model_profile=None):
	if mode != 'direct':
		raise ValueError('Browser Use 目前支持直接执行。旧记录中的 Planner 详情仍可查看。')
	if provider not in (None, '', 'browser_use'):
		raise ValueError('测试台使用 Browser Use 执行。')
	policy = screenshot_policy if screenshot_policy is not None else 'always' if screenshot_mode else 'off'
	if policy != 'off':
		raise ValueError('当前两套 DeepSeek 配置使用文本输入，实时预览仍会生成。')
	profile = validate_model_profile(model_profile)
	if profile == FINAL_REPORT_PROFILE:
		verifier = report_text_config()
		execution = planner_config(service=verifier)
	else:
		execution = planner_config(require_key=True)
		verifier = text_config()
		if not verifier['api_key']:
			raise ValueError('调试终验需要原 Jev 的 TEXT_MODEL_API_KEY。')

	public = public_config(execution)
	return {
		'system1': {'provider': 'browser_use', 'model': public['model'], 'base_url': public['base_url']},
		'system1_key': execution['api_key'],
		'execution_config': execution,
		'text': verifier,
		'model_profile': profile,
		'screenshot_mode': False,
		'screenshot_policy': policy,
		'headless': os.getenv('BROWSER_USE_BENCHMARK_HEADLESS', 'true').lower() not in {'0', 'false'},
		'user_data_dir': os.getenv('BROWSER_USE_BENCHMARK_USER_DATA_DIR'),
		'planner': None,
	}
