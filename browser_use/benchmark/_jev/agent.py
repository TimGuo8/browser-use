"""The complete agent loop. Typed choices, observable state, bounded execution."""

import base64
import json
import os
import time
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path

from .browser import Browser, StalePage
from .model import (
	action_space,
	choose,
	deepseek_key,
	field_context,
	field_text,
	system1_config,
	text_config,
	validate_screenshot_mode,
	validate_screenshot_policy,
)
from .planner import Planning, document_key, planner_config, semantic_key
from .planner import execution_mode as validate_mode
from .questions import MAX_DECISION_ROUNDS, MAX_STEPS
from .research import Research
from .runtime import (
	DEBUG_PROFILE,
	FINAL_REPORT_PROFILE,
	redact_secrets,
	report_text_config,
	runtime_info,
	validate_model_profile,
)
from .token_usage import token_usage


class RunCancelled(RuntimeError):
	"""Cooperative cancellation at a boundary before browser input."""

	is_cancellation = True


def runtime_settings(provider, mode, screenshot_mode=False, *, screenshot_policy=None, model_profile=None):
	"""一次冻结整次运行的服务与凭据，批次中每个任务复用该快照。"""
	profile = validate_model_profile(model_profile)
	config = system1_config(provider, require_key=True)
	validate_screenshot_mode(screenshot_mode, config['provider'])
	policy = validate_screenshot_policy(
		screenshot_policy if screenshot_policy is not None else 'always' if screenshot_mode else 'off', config['provider'], mode
	)
	key = (
		deepseek_key()
		if config['provider'] == 'deepseek'
		else os.getenv('TYPESAFE_API_KEY' if config['provider'] == 'jev' else 'LAYA_API_KEY', '')
	)
	report = report_text_config() if profile == FINAL_REPORT_PROFILE else None
	return {
		'system1': config,
		'system1_key': key,
		'text': report if report is not None else text_config(),
		'model_profile': profile,
		'screenshot_mode': policy == 'always',
		'screenshot_policy': policy,
		'planner': planner_config(require_key=True, service=report) if mode == 'planned' else None,
	}


class Agent:
	def __init__(
		self,
		url,
		goals,
		*,
		record_dir=None,
		screenshots=False,
		on_event=None,
		should_cancel=None,
		prepare=None,
		system1_provider=None,
		execution_mode=None,
		screenshot_mode=False,
		screenshot_policy=None,
		model_profile=None,
		_settings=None,
	):
		# 先验证并冻结配置，配置错误不得创建浏览器或触发模型请求。
		mode = validate_mode(execution_mode)
		settings = deepcopy(
			_settings
			or runtime_settings(
				system1_provider, mode, screenshot_mode, screenshot_policy=screenshot_policy, model_profile=model_profile
			)
		)
		screenshot_mode = validate_screenshot_mode(settings.get('screenshot_mode', False), settings['system1']['provider'])
		policy = validate_screenshot_policy(
			settings.get('screenshot_policy', 'always' if screenshot_mode else 'off'), settings['system1']['provider'], mode
		)
		screenshot_mode = policy == 'always'
		planning_config = settings['planner']
		self.planning = None
		task = goals.strip() if isinstance(goals, str) else '\n'.join(goals).strip()
		# 空任务不能消耗任何浏览器或模型资源。
		if not task:
			raise ValueError('Supply a task')
		# 在连接浏览器之前固定任务的本地日期和 UTC offset，后续观察不推进时间参照。
		task_started = datetime.now().astimezone()
		plan = [task]
		config = settings['system1']
		self.system1_config = config
		self.system1_key = settings['system1_key']
		self.helper_config = settings['text']
		profile = settings.get('model_profile', DEBUG_PROFILE)
		self._report_key = self.helper_config['api_key'] if profile == FINAL_REPORT_PROFILE else None
		self.pending_text = None
		self.on_event = on_event
		self.should_cancel = should_cancel
		self.browser = Browser(url)
		self.record_dir = Path(record_dir) if record_dir else None
		self.screenshots = screenshots or bool(self.record_dir)
		try:
			# 基线准备只改变起始状态，执行策略不接收准备脚本。
			if prepare:
				prepare(self.browser)
			self.check_cancelled()
			page = self.browser.observe(screenshot=self.screenshots or screenshot_mode)
		except Exception:
			self.browser.close()
			raise
		self.state = dict(
			browser=self.browser,
			runtime=runtime_info(),
			system1_provider=config['provider'],
			system1_model=config['model'],
			system1_base_url=config['base_url'],
			screenshot_mode=screenshot_mode,
			screenshot_policy=policy,
			execution_mode=mode,
			model_profile=profile,
			text_model=self.helper_config['model'],
			text_model_base_url=self.helper_config['base_url'],
			goal='\n'.join(plan),
			task_time={'started_at': task_started.isoformat(timespec='seconds'), 'date': task_started.date().isoformat()},
			page=page,
			decision=None,
			history=[],
			status='ready',
			plan=plan,
			plan_index=0,
			decisions=[],
			decision_rounds=0,
			text_calls=[],
			model_calls=[],
			elapsed_ms=0,
			started_at=None,
			event_seq=0,
			observation_index=-1,
			record=bool(self.record_dir),
		)
		self.research = Research(self)
		# 结构化状态在初始观察事件之前就可被归档，但此时不请求云端。
		if planning_config is not None:
			self.planning = Planning(self, planning_config)
			self.state.update(plan=[], status='planning')
		try:
			self.emit('observation', phase='initial')
		except Exception:
			self.browser.close()
			raise
		# 仅显式启用帧录制时写入自由任务的截图文件。
		if self.record_dir:
			self.record_dir.mkdir(parents=True, exist_ok=True)
			(self.record_dir / '000000.jpg').write_bytes(base64.b64decode(page['screenshot']))

	def check_cancelled(self):
		"""在模型与浏览器操作边界统一处理暂停和超时。"""
		callback = getattr(self, 'should_cancel', None)
		reason = callback() if callback else None
		# 回调只提供停止原因，不进行额外模型判断。
		if reason:
			raise RunCancelled(str(reason))

	def emit(self, event, **payload):
		"""将事件交给归档器；自由任务则在内存保留观察。"""
		self.state['event_seq'] = self.state.get('event_seq', 0) + 1
		if event == 'observation':
			self.state['observation_index'] = self.state.get('observation_index', -1) + 1
		payload = {
			'event_seq': self.state['event_seq'],
			'research_revision': self.state.get('research', {}).get('revision'),
			**payload,
		}
		# 让观察和动作与当时子任务绑定，不从最新计划反推旧记录。
		if getattr(self, 'planning', None):
			payload = {**self.planning.metadata(), **payload}
		callback = getattr(self, 'on_event', None)
		# 原始观察记忆独立于归档回调；基线和自由任务使用同一份可追溯证据。
		# 图片仍由当前观察和归档器持有，不在历史记忆中重复附带。
		if event == 'observation':
			page = {k: v for k, v in self.state['page'].items() if k != 'screenshot'}
			self.state.setdefault('observations', []).append(
				{
					'page': page,
					'phase': payload.get('phase'),
					'step': len(self.state['history']),
					'elements': action_space(page['actions'])[0],
					'event_seq': payload['event_seq'],
					'research_revision': payload['research_revision'],
					**{k: payload[k] for k in ('plan_revision', 'subgoal_id') if k in payload},
				}
			)
		# 基线回调负责截图、独立验收及增量落盘。
		if callback:
			callback(event, self, payload)

	def observe(self, phase='step'):
		"""只读当前页面，再通知日志和独立验收器。"""
		self.state['page'] = self.state['browser'].observe(
			screenshot=self.screenshots or self.state.get('screenshot_mode', False)
		)
		self.emit('observation', phase=phase)

	def snapshot(self):
		"""公开状态不包含浏览器对象或冻结的服务凭据。"""
		return redact_secrets(
			{
				**{k: v for k, v in self.state.items() if k != 'browser'},
				'token_usage': token_usage(self.state.get('model_calls', [])),
				'elements': action_space(self.state['page']['actions'])[0],
			},
			getattr(self, '_report_key', None),
		)

	def call_model(self, kind, function, *args, **metadata):
		"""记录每次实际模型调用，包括失败响应和耗时。"""
		# 所有模型共用日志入口，Planner 不计入 System1 轮数。
		if getattr(self, 'planning', None):
			metadata = {**self.planning.metadata(), **metadata}
		calls = self.state.setdefault('model_calls', [])
		record = dict(id=len(calls) + 1, kind=kind, status='pending', **metadata)
		record.setdefault('research_before', self.state.get('research', {}).get('revision'))
		record['event_seq'] = self.state.get('event_seq', 0) + 1
		# 历史记录保存当时计划，而不是查看时的最新计划。
		if getattr(self, 'planning', None):
			record['planning_snapshot'] = self.planning.snapshot()
		# 基线的观察编号由其回调补充，自由任务直接使用内存编号。
		if self.state.get('observations'):
			record['observation_index'] = len(self.state['observations']) - 1
		calls.append(record)
		started = time.perf_counter()
		self.emit('model_start', call=record)
		try:
			result = function(*args, trace=record)
			record['status'] = 'success'
			# Planner 返回状态转换，不能当作操作概率解析。
			if kind == 'planner':
				record['planner_result'] = result
			# 只有 System1 记录才能提供 NEXT ACTION 的操作与目标。
			if kind in {'laya', 'deepseek', 'jev'}:
				record.update(operation=result['operation'], target=result['target'])
				record['decision'] = {
					k: result.get(k)
					for k in (
						'choice',
						'operation',
						'target',
						'confidence',
						'target_confidence',
						'latency_ms',
						'operation_probabilities',
						'target_probabilities',
						'provider',
						'forced_unable',
					)
				}
			return result
		except Exception as error:
			record.update(status='error', error=str(error))
			raise
		finally:
			record['latency_ms'] = round((time.perf_counter() - started) * 1000)
			# 统一以包含 HTTP 等待的真实耗时更新决策展示。
			if 'decision' in record:
				record['decision']['latency_ms'] = record['latency_ms']
			self.emit('model_end', call=record)

	def command(self, name, body=None):
		"""在所有推进路径统计失效和耗时；不自动重试任何浏览器变更。"""
		try:
			return self._command(name, body)
		except RunCancelled as error:
			# 用户暂停时废弃待执行选择，恢复后重新观察，不重放旧动作。
			self.state['decision'] = None
			# 自由任务的暂停可继续；基线取消由外部管理器结束。
			if str(error) == 'paused':
				if self.state['status'] not in {'finalizing', 'collecting'}:
					self.state['status'] = (
						'planning' if getattr(self, 'planning', None) and self.planning.data['pending'] else 'ready'
					)
			self.emit('run_cancelled', reason=str(error))
			raise
		except StalePage:
			# 手动选择、两阶段选择和自动执行均经过相同的失效计数。
			if getattr(self, 'planning', None):
				self.planning.stale()
			raise
		finally:
			# 未开始首次推进时不计算运行时长。
			if self.state.get('started_at') is not None:
				self.state['elapsed_ms'] = round((time.perf_counter() - self.state['started_at']) * 1000)

	def _command(self, name, body=None):
		"""每次推进至多完成云端交接或一个本地动作。"""
		body = body or {}
		state = self.state
		planning = getattr(self, 'planning', None)
		self.check_cancelled()
		# 自动推进复用同一套选择和执行边界。
		if name == 'tick':
			try:
				self.command('predict', {})
				# 云端交接不会产生浏览器决策，必须留到下一次推进。
				if state['decision'] is not None:
					return self.command('act', {'fingerprint': state['page']['fingerprint']})
				return self.snapshot()
			except StalePage:
				state['decision'] = None
				state['status'] = 'planning' if planning and planning.data['pending'] else 'ready'
				self.observe('stale_retry')
				error = state.get('execution_error')
				if not planning and error and error['repeats'] >= 3 and error['fingerprint'] == state['page']['fingerprint']:
					state.update(status='blocked', stop_reason=error['reason'])
				state['elapsed_ms'] = round((time.perf_counter() - state['started_at']) * 1000)
				return self.snapshot()
		# 手动选择不会直接变更页面。
		elif name == 'predict':
			# 关闭后的运行必须重新建立受控标签页。
			if not state['browser']:
				raise ValueError('Start a demo first')
			# 计时从首次推进开始，包含初始规划耗时。
			if state['started_at'] is None:
				state['started_at'] = time.perf_counter()
			# 先检查终态和总预算，不能用重规划绕过已结束任务。
			if state['status'] in {'done', 'blocked'}:
				raise ValueError('This run has stopped. Start a fresh demo.')
			if state['status'] == 'finalizing':
				self.research.finalize()
				return self.snapshot()
			if state['status'] == 'collecting':
				self.research.resume_collection()
				return self.snapshot()
			# 云端交接优先于下一轮本地选择。
			if planning:
				planning.guard()
				# 一次请求只处理一次交接，不继续预测或点击。
				if planning.data['pending']:
					planning.run()
					state['elapsed_ms'] = round((time.perf_counter() - state['started_at']) * 1000)
					return self.snapshot()
			# 已变更页面重新观察，禁止使用旧候选表。
			if not state['browser'].fresh(state['page']):
				self.observe('refresh')
				# 连续失效由规划器处理，而不是无限刷新。
				if planning:
					planning.stale()
					# 达到失效阈值后留到下一次推进交接。
					if planning.data['pending']:
						return self.snapshot()
			elif (
				state.get('screenshot_policy') == 'smart' and state.get('screenshot_mode') and not state['page'].get('screenshot')
			):
				# 智能截图开启时，页面和图片必须来自同一次观察。
				self.observe('screenshot_required')
			self.check_cancelled()
			state['decision'] = None
			rounds = state.get('decision_rounds', len(state['decisions']))
			# 失败的已开始决策同样消耗轮数。
			if rounds >= MAX_DECISION_ROUNDS:
				raise ValueError("Reached the demo's decision-round budget")
			state['decision_rounds'] = rounds + 1
			provider = state.get('system1_provider', 'laya')

			def stage_call(function, question):
				"""两阶段之间重新检查停止和新鲜度，不重发已执行的动作。"""
				self.check_cancelled()
				# 操作与目标必须来自同一份观察。
				if not state['browser'].fresh(state['page']):
					raise StalePage('Page changed between decision stages. Choose again.')
				return self.call_model('deepseek', function, question, decision_round=rounds + 1, stage=question)

			# 同一逻辑子任务的重规划保留执行历史，ADVANCE 后才从新阶段重新累计。
			history = state['history'][planning.data['history_start'] :] if planning else state['history']
			chooser = partial(choose, provider=provider, screenshot_mode=state.get('screenshot_mode', False))
			if provider == 'jev' and state.get('task_time'):
				chooser = partial(chooser, task_time=state['task_time'])
			error = state.get('execution_error')
			if error and error['fingerprint'] == state['page']['fingerprint']:
				chooser = partial(chooser, execution_error=error)
			# 正常运行固定模型，兼容旧的轻量测试实例。
			if hasattr(self, 'system1_config'):
				chooser = partial(chooser, config=self.system1_config, api_key=self.system1_key)
			if getattr(self, 'research', None):
				chooser = partial(
					chooser, research=self.research.progress(), collect_available=self.research.available(state['page'])
				)
			# 规划模式只补充当前项，直接模式提示词保持不变。
			if planning:
				chooser = partial(
					chooser, subgoal=planning.local_context(include_navigation=provider == 'jev'), task_history=state['history']
				)
			# DeepSeek 按条件分两次请求，每个阶段独立归档。
			if provider == 'deepseek':
				decision = chooser(state['page'], state['goal'], history, stage_call=stage_call)
			# Laya/Jev 仍保持单请求多问题，只消费选中操作的目标。
			else:
				decision = self.call_model(provider, chooser, state['page'], state['goal'], history, decision_round=rounds + 1)
			self.check_cancelled()
			state['decision'] = decision
			state['decisions'].append(
				{
					**state['decision'],
					'fingerprint': state['page']['fingerprint'],
					'elapsed_ms': round((time.perf_counter() - state['started_at']) * 1000),
					**(planning.metadata() if planning else {}),
				}
			)
			state['status'] = 'predicted'
		# 仅执行用户查看过或本轮自动产生的选择。
		elif name == 'act':
			decision, page = state['decision'], state['page']
			# 请求必须绑定当前观察，不能拿历史记录直接重放。
			if not decision or body.get('fingerprint') != page['fingerprint']:
				raise ValueError('Observe and choose before acting')
			# 先消费选择再调用 helper 或浏览器，防止请求重试导致重复点击。
			state['decision'] = None
			selected = decision['choice']
			if selected == 'COLLECT':
				if not state['browser'].fresh(page, read_only=True):
					raise StalePage('Page changed before collection. Choose again.')
				self.research.collect(page)
				return self.snapshot()
			# 子任务完成/困难只是请求交接，不在同一次 act 中调用云端。
			if planning and selected in {'STEP_DONE', 'UNABLE'}:
				if selected == 'STEP_DONE' and planning.action_pending():
					raise ValueError('This subgoal requires an actual interaction before completion review')
				if selected == 'STEP_DONE' and planning.completion_pending():
					raise ValueError('Planner rejected this condition on unchanged evidence; progress before review')
				# 这里只交接申请；Planner 会重新读取当前证据后判断完成，不因视频计时作废。
				if not state['browser'].fresh(page, read_only=True):
					raise StalePage('Page changed before subgoal handoff. Choose again.')
				planning.data['stale_count'] = 0
				planning.schedule(selected)
				return self.snapshot()
			# 直接模式保留原始终止语义。
			if selected in {'DONE', 'BLOCKED'}:
				# 终止前仍验证观察，独立基线随后核验事实。
				if not state['browser'].fresh(page):
					state['status'] = 'ready'
					raise StalePage('Page changed since the decision. Choose again.')
				state['status'] = 'finalizing' if selected == 'DONE' else 'blocked'
				state['plan_index'] = int(state['status'] == 'done')
				state['elapsed_ms'] = round((time.perf_counter() - state['started_at']) * 1000)
				return self.snapshot()
			action = next(a for a in page['actions'] if a['id'] == selected)
			# 动作上限最后一道保护在任何浏览器输入之前。
			if len(state['history']) >= MAX_STEPS:
				state['status'] = 'blocked'
				raise ValueError(f'Stopped at the {MAX_STEPS}-action demo budget')
			text, helper = None, None
			# 只有文本输入需要 helper，选择模型不能直接提供任意输入值。
			if action['kind'] == 'fill':
				# 不为已经失效的字段生成文本。
				if not state['browser'].fresh(page, action):
					raise StalePage('Page changed before text generation. Choose again.')
				context = field_context(state['goal'], action, page, state['history'])
				if state.get('task_time'):
					context['task_time'] = deepcopy(state['task_time'])
				# 当前目标及版本进入缓存键，防止跨子任务误用旧文本。
				if planning:
					context['subgoal'] = {**planning.local_context(), **planning.metadata()}
				# 只有完整 helper 输入完全相等时才复用失效重试值。
				if self.pending_text and self.pending_text[0] == context:
					_, text, helper = self.pending_text
				# 新上下文必须重新请求 helper。
				else:
					# 实际 helper 配置固定于启动时，缓存仍按完整输入严格匹配。
					helper_function = (
						partial(field_text, config=self.helper_config) if hasattr(self, 'helper_config') else field_text
					)
					text, helper = self.call_model(
						'text',
						helper_function,
						context,
						field=action['label'],
						decision_round=state.get('decision_rounds'),
						decision_call_id=next(
							(c['id'] for c in reversed(state.get('model_calls', [])) if c['kind'] in {'laya', 'deepseek', 'jev'}),
							None,
						),
					)
					if text is None:
						self.pending_text = None
						text_call = state['model_calls'][-1]
						refusal = {
							'operation': 'TYPE_TEXT',
							'field': context['field'],
							'node': action['node'],
							'source_url': page['url'],
							'source_document': document_key(page)[1],
							'decision_call_id': text_call['decision_call_id'],
							'text_call_id': text_call['id'],
							'executed': False,
							'reason': 'Text helper declined to supply a field value. Nothing was typed.',
						}
						self.emit('action_rejected', action=action['label'], kind=action['kind'], **refusal)
						self.check_cancelled()
						if planning:
							planning.schedule('UNABLE', text_refusal=refusal)
						else:
							state.update(status='blocked', stop_reason=refusal['reason'])
						return self.snapshot()
					self.pending_text = (context, text, helper)
					state['text_calls'].append({**helper, 'field': action['label'], 'value': text})
			# helper 返回后再检查停止，Browser.act 还会在真正输入前检查遮挡与新鲜度。
			self.check_cancelled()
			# 同一语义状态下用当前候选索引区分同名控件，不让 DOM 节点重建掩盖循环。
			repeat_key = (
				json.dumps(
					[
						semantic_key(page),
						decision['operation'],
						{'index': decision.get('target'), **{k: action.get(k) for k in ('kind', 'label', 'role', 'value')}},
						text,
					],
					sort_keys=True,
					ensure_ascii=False,
				)
				if planning
				else None
			)
			self.emit('action_started', action=action, text=text)
			try:
				state['browser'].act(action, page, text=text)
			except StalePage as error:
				previous = state.get('execution_error', {})
				rejected = dict(
					fingerprint=page['fingerprint'],
					node=action.get('node'),
					kind=action['kind'],
					action=action['label'],
					reason=str(error),
					executed=False,
				)
				same = all(previous.get(k) == v for k, v in rejected.items())
				rejected['repeats'] = previous.get('repeats', 0) + 1 if same else 1
				state['execution_error'] = rejected
				self.emit('action_rejected', **rejected)
				raise
			state.pop('execution_error', None)
			self.pending_text = None
			state['elapsed_ms'] = round((time.perf_counter() - state['started_at']) * 1000)
			# 执行先记账再观察；观察失败不得抹掉或重放刚刚发送的动作。
			state['history'].append(
				{
					'step': len(state['history']) + 1,
					'action': action['label'],
					'kind': action['kind'],
					'choice': selected,
					'node': action.get('node'),
					'role': action.get('role'),
					'previous_value': action.get('value'),
					'previous_checked': action.get('checked'),
					'source_url': page['url'],
					'source_document': document_key(page)[1],
					'context': deepcopy(action.get('context')),
					'probability': decision['probabilities'].get(selected),
					'confidence': decision['confidence'],
					'latency_ms': decision['latency_ms'],
					'text': text,
					'text_helper': helper['model'] if helper else None,
					'text_latency_ms': helper['latency_ms'] if helper else 0,
					'operation': decision['operation'],
					'target': decision['target'],
					'page_changed': None,
					'url': page['url'],
					'usage': decision['usage'],
					'executed_ms': round((time.perf_counter() - state['started_at']) * 1000),
					'elapsed_ms': state['elapsed_ms'],
					**({**planning.metadata(), 'repeat_key': repeat_key} if planning else {}),
				}
			)
			self.emit('action_executed', action=state['history'][-1])
			self.observe()
			state['elapsed_ms'] = round((time.perf_counter() - state['started_at']) * 1000)
			state['history'][-1].update(
				page_changed=state['page']['fingerprint'] != page['fingerprint'],
				url=state['page']['url'],
				elapsed_ms=state['elapsed_ms'],
			)
			# 录制只保存本次观察截图，不为补图重复浏览器动作。
			if state['record']:
				(self.record_dir / f'{state["elapsed_ms"]:06d}.jpg').write_bytes(base64.b64decode(state['page']['screenshot']))
			repeated = state['history'][-3:]
			# 规划模式由云端评估局部异常，旧模式保持原来的停止规则。
			if planning:
				state['status'] = 'ready'
				planning.data['stale_count'] = 0
				planning.guard()
				return self.snapshot()
			state['status'] = (
				'blocked'
				if len(repeated) == 3 and all(h['page_changed'] is False and h['kind'] != 'wait' for h in repeated)
				else 'ready'
			)
		# 未知命令明确拒绝，不推断用户希望执行的操作。
		else:
			raise ValueError('Unknown command')
		return self.snapshot()

	def run(self):
		"""同步迭代状态，云端交接同样向调用方产出一个状态。"""
		# 所有预算由统一命令边界执行，不能用新的迭代重新计数。
		while self.state['status'] not in {'done', 'blocked'}:
			yield self.command('tick')

	def close(self):
		"""只释放当前 Agent 持有的标签页。"""
		self.browser.close()

	def __enter__(self):
		"""允许库调用使用 with 管理受控标签页。"""
		return self

	def __exit__(self, *_args):
		"""异常退出也关闭标签页，不吞掉原始异常。"""
		self.close()
