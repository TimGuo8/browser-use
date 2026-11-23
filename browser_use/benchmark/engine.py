"""Browser Use execution behind the original synchronous Jev inspector contract."""

from __future__ import annotations

import asyncio
import base64
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._jev.agent import Agent as InspectorAgent
from ._jev.agent import RunCancelled
from ._jev.browser import READ_STATE, fingerprint
from ._jev.model import action_space
from ._jev.questions import MAX_STEPS

if TYPE_CHECKING:
	from browser_use.agent.service import Agent
	from browser_use.browser.session import BrowserSession, CDPSession
	from browser_use.browser.views import BrowserStateSummary

if os.name == 'nt':

	class _WindowsBrowserLoop(asyncio.ProactorEventLoop):
		"""Own the native watchdog's subprocess transports until their pipes close."""

		def __init__(self):
			super().__init__()
			self._child_transports = []

		async def subprocess_exec(self, *args, **kwargs):
			transport, protocol = await super().subprocess_exec(*args, **kwargs)
			self._child_transports.append(transport)
			return transport, protocol

		def close(self):
			# Browser Use retains only the browser PID. The asyncio pipe handles
			# must be released on their owning loop before closing its proactor.
			for transport in self._child_transports:
				transport.close()
			self._child_transports.clear()
			if not self.is_closed():
				self.run_until_complete(asyncio.sleep(0))
			super().close()


class _BrowserLoop:
	"""Keep CDP and its owning BrowserSession on one loop between HTTP requests."""

	def __init__(self):
		self.loop = _WindowsBrowserLoop() if os.name == 'nt' else asyncio.new_event_loop()
		self.thread = threading.Thread(target=self._serve, name='benchmark-browser', daemon=True)
		self.thread.start()

	def _serve(self):
		asyncio.set_event_loop(self.loop)
		try:
			self.loop.run_forever()
		finally:
			pending = asyncio.all_tasks(self.loop)
			for task in pending:
				task.cancel()
			if pending:
				self.loop.run_until_complete(asyncio.wait(pending, timeout=3))
			self.loop.run_until_complete(self.loop.shutdown_asyncgens())
			self.loop.close()

	def run(self, operation, *, cancel=None, timeout=90):
		async def bounded():
			if cancel is None:
				return await asyncio.wait_for(operation, timeout=timeout)
			task = asyncio.create_task(operation)
			try:
				async with asyncio.timeout(timeout):
					while not task.done():
						reason = cancel()
						if reason:
							raise RunCancelled(str(reason))
						await asyncio.wait({task}, timeout=0.12)
					return task.result()
			finally:
				if not task.done():
					task.cancel()
					await asyncio.gather(task, return_exceptions=True)

		return asyncio.run_coroutine_threadsafe(bounded(), self.loop).result()

	def close(self):
		self.loop.call_soon_threadsafe(self.loop.stop)
		self.thread.join()


class _BoundBrowser:
	"""Adapt Jev's read-only probe and allowlisted fixture setup to an owned CDP target."""

	def __init__(self, bridge):
		self.bridge = bridge
		self.native: BrowserSession | None = None
		self.bound: CDPSession | None = None
		self.target = self.session = None
		self._owned = {}

	async def bind(self):
		browser = self.native
		if browser is None or not (target := browser.agent_focus_target_id):
			raise RuntimeError('Browser Use has no active page')
		bound = await browser.get_or_create_cdp_session(target, focus=False)
		self.bound = bound
		self.target, self.session = bound.target_id, bound.session_id
		self._owned = {self.target: {'session': self.session}}

	async def send(self, method, params):
		browser = self.native
		bound = self.bound
		if browser is None or bound is None or browser.agent_focus_target_id != bound.target_id:
			raise RuntimeError('Browser target changed before sampling')
		active = await browser.get_or_create_cdp_session(bound.target_id, focus=False)
		if active.session_id != bound.session_id:
			raise RuntimeError('Browser session changed before sampling')
		group, name = method.split('.')
		command = getattr(getattr(bound.cdp_client.send, group), name)
		result = await command(params=params, session_id=bound.session_id)
		if browser.agent_focus_target_id != bound.target_id:
			raise RuntimeError('Browser target changed while sampling')
		return result

	def call(self, method, **params):
		timeout = params.pop('_response_timeout', 20)
		return self.bridge.run(self.send(method, params), timeout=timeout)

	def evaluate(self, expression, *, utf8=False):
		result = self.call('Runtime.evaluate', expression=expression, returnByValue=True, awaitPromise=True)
		if result.get('exceptionDetails'):
			raise RuntimeError(f'Browser evaluation failed: {result["exceptionDetails"].get("text", "")}')
		return result['result'].get('value')

	def observe(self, *, screenshot):
		self.bridge.run(self.bind())
		page = self.evaluate(READ_STATE)
		if not isinstance(page, dict):
			raise RuntimeError('Browser document is unavailable')
		page.update(target_id=self.target, observed_at=datetime.now(timezone.utc).isoformat())
		page['dom_fingerprint'] = page['fingerprint'] = fingerprint(page)
		if screenshot:
			page['screenshot'] = self.call('Page.captureScreenshot', format='jpeg', quality=72, captureBeyondViewport=False)[
				'data'
			]
		return page


class BrowserUseAgent(InspectorAgent):
	"""Expose native observation, prediction and execution as the inspector's original commands."""

	def __init__(
		self,
		url,
		goal,
		screenshots=False,
		on_event=None,
		should_cancel=None,
		prepare=None,
		system1_provider=None,
		execution_mode='direct',
		_settings=None,
		record_dir=None,
		**_,
	):
		if not _settings:
			from .models import runtime_settings

			_settings = runtime_settings(system1_provider, execution_mode)
		settings = _settings
		config = settings['system1']
		provider = config['provider']
		if execution_mode not in (None, 'direct'):
			raise ValueError('Browser Use currently supports direct execution')
		if provider != 'browser_use':
			raise ValueError(f'Unsupported Browser Use provider: {provider}')
		self.helper_config = settings['text']
		self._report_key = self.helper_config.get('api_key') if settings.get('model_profile') == 'final_report' else None
		self.on_event, self.should_cancel = on_event, should_cancel
		self.record_dir = Path(record_dir) if record_dir else None
		self.screenshots = screenshots or bool(self.record_dir)
		self.state: dict[str, Any] = dict(
			goal=goal,
			system1_provider=provider,
			system1_model=config['model'],
			system1_base_url=config.get('base_url', ''),
			execution_mode='direct',
			screenshot_mode=settings.get('screenshot_mode', False),
			screenshot_policy=settings.get('screenshot_policy', 'off'),
			model_profile=settings.get('model_profile', 'debug'),
			text_model=self.helper_config['model'],
			text_model_base_url=self.helper_config['base_url'],
			status='ready',
			page={},
			decision=None,
			decisions=[],
			history=[],
			model_calls=[],
			text_calls=[],
			decision_rounds=0,
			elapsed_ms=0,
			event_seq=0,
			observation_index=-1,
			plan=[goal],
			plan_index=0,
			record=bool(self.record_dir),
			free_step_supported=True,
		)
		self._started = time.monotonic()
		self._closed = False
		self._browser_start_dispatched = False
		self.check_cancelled()
		self._temp = tempfile.TemporaryDirectory(prefix='browser-use-benchmark-', ignore_cleanup_errors=True)
		self._bridge = _BrowserLoop()
		self.browser = _BoundBrowser(self._bridge)
		self.native: Agent | None = None
		self.summary: BrowserStateSummary | None = None
		try:
			# BrowserStartEvent launches and attaches CDP on the native event bus.
			# Let that operation settle before issuing kill() on the same session.
			self._bridge.run(self._start(url, settings), timeout=110)
			self.check_cancelled()
			if prepare:
				prepare(self.browser)
			self.check_cancelled()
			self.observe('initial')
		except BaseException:
			self.close()
			raise

	async def _start(self, url, settings):
		self.check_cancelled()
		from browser_use import Agent, BrowserSession, ChatDeepSeek

		self.check_cancelled()
		config = settings['system1']
		executable = settings.get('browser_executable') or os.getenv('BROWSER_USE_EXECUTABLE_PATH')
		profile = settings.get('user_data_dir') or str(Path(self._temp.name) / 'profile')
		options: dict[str, Any] = dict(
			user_data_dir=profile, keep_alive=True, enable_default_extensions=False, viewport={'width': 1366, 'height': 780}
		)
		if 'headless' in settings:
			options['headless'] = settings['headless']
		if executable:
			options['executable_path'] = executable
		browser = BrowserSession(**options)
		self.browser.native = browser
		execution = settings['execution_config']
		llm = ChatDeepSeek(
			model=execution['model'],
			api_key=execution['api_key'],
			base_url=execution['base_url'],
			timeout=execution['timeout'],
			max_tokens=execution['max_tokens'],
			thinking=False,
		)
		original = llm.ainvoke

		async def traced(messages, output_format=None, **kwargs):
			record = dict(
				id=len(self.state['model_calls']) + 1,
				kind=config['provider'],
				model=config['model'],
				status='pending',
				observation_index=self.state['observation_index'],
				request={'model': config['model'], 'messages': [m.model_dump(mode='json') for m in messages]},
			)
			self.state['model_calls'].append(record)
			self.emit('model_start', call=record)
			started = time.monotonic()
			try:
				reply = await original(messages, output_format=output_format, **kwargs)
				record.update(status='success', response=reply.model_dump(mode='json', exclude_none=True))
				return reply
			except BaseException as error:
				record.update(status='error', error=str(error) or type(error).__name__)
				raise
			finally:
				record['latency_ms'] = round((time.monotonic() - started) * 1000)
				self.emit('model_end', call=record)

		async def check_stop():
			self.check_cancelled()
			return False

		object.__setattr__(llm, 'ainvoke', traced)
		self.native = Agent(
			task=self.state['goal'],
			llm=llm,
			browser_session=browser,
			use_judge=False,
			directly_open_url=False,
			enable_signal_handler=False,
			register_external_agent_status_raise_error_callback=check_stop,
			use_vision=self.state['screenshot_mode'],
			max_actions_per_step=1,
			enable_planning=False,
			file_system_path=str(Path(self._temp.name) / 'agent-files'),
		)
		self.check_cancelled()
		self._browser_start_dispatched = True
		await browser.start()
		self.check_cancelled()
		await browser.navigate_to(url)
		self.check_cancelled()
		await self.browser.bind()

	def observe(self, phase='step'):
		self.state['page'] = self.browser.observe(screenshot=self.screenshots)
		self.emit('observation', phase=phase)
		if self.screenshots and not self.on_event:
			self.state['observations'][-1]['page']['screenshot'] = self.state['page']['screenshot']
		if self.record_dir:
			self.record_dir.mkdir(parents=True, exist_ok=True)
			if phase == 'initial':
				self._record_started = time.monotonic()
			elapsed = round((time.monotonic() - self._record_started) * 1000)
			frame = self.record_dir / f'{elapsed:06d}.jpg'
			frame.write_bytes(base64.b64decode(self.state['page']['screenshot'], validate=True))
		return self.state['page']

	async def _predict(self):
		from browser_use.agent.views import AgentStepInfo

		agent = self.native
		if agent is None:
			raise RuntimeError('Browser Use agent is unavailable')
		agent.step_start_time = time.time()
		summary = await agent._prepare_context(AgentStepInfo(agent.state.n_steps - 1, MAX_STEPS))
		self.summary = summary
		agent.state.last_model_output = agent.state.last_result = None
		await agent._get_next_action(summary)
		model_output = agent.state.last_model_output
		if model_output is None:
			raise RuntimeError('Browser Use did not return a decision')
		actions = model_output.action
		if len(actions) != 1:
			raise ValueError('Expected one Browser Use action per inspector step')
		raw = actions[0].model_dump(exclude_none=True)
		operation, args = next(iter(raw.items()))
		# Native indices and the inspector's indices enumerate different trees.
		# Match DOM identity before drawing a target on the inspector screenshot.
		target = ''
		index = args.get('index')
		node = summary.dom_state.selector_map.get(index) if isinstance(index, int) else None
		if node is not None:
			try:
				obj = await self.browser.send('DOM.resolveNode', {'backendNodeId': node.backend_node_id})
				object_id = obj['object']['objectId']
				try:
					result = await self.browser.send(
						'Runtime.callFunctionOn',
						{
							'objectId': object_id,
							'returnByValue': True,
							'functionDeclaration': 'function(){return window.__jevFast?.ids?.get(this) ?? null}',
						},
					)
				finally:
					await self.browser.send('Runtime.releaseObject', {'objectId': object_id})
				identity = result.get('result', {}).get('value')
				elements = action_space(self.state['page']['actions'], include_node_refs=True)[0]
				target = next((e['index'] for e in elements if e['node_ref'] == identity), '')
			except (RuntimeError, KeyError):
				pass  # Closed frame or remote child target cannot be annotated on this screenshot.
		return raw, operation, target

	async def _act(self):
		agent = self.native
		if agent is None:
			raise RuntimeError('Browser Use agent is unavailable')
		await agent._execute_actions()
		await agent._post_process()
		await agent._finalize(self.summary)
		return [r.model_dump(mode='json', exclude_none=True) for r in (agent.state.last_result or [])]

	def command(self, name, body=None):
		if name not in {'predict', 'act', 'tick'}:
			raise ValueError(f'Unknown Browser Use command: {name}')
		try:
			self.check_cancelled()
			if self.state['status'] in {'done', 'blocked'}:
				return self.snapshot()
			if name in {'predict', 'tick'} and self.state['decision'] is None:
				self._choose()
			if name in {'act', 'tick'}:
				self._execute((body or {}).get('fingerprint'))
		except RunCancelled:
			self.state.update(decision=None, status='ready')
			raise
		finally:
			self.state['elapsed_ms'] = round((time.monotonic() - self._started) * 1000)
		return self.snapshot()

	def _choose(self):
		if self.state['decision_rounds'] >= MAX_STEPS:
			raise RunCancelled('timeout')
		raw, operation, target = self._bridge.run(self._predict(), cancel=self.should_cancel, timeout=150)
		decision = dict(
			operation=operation,
			target=target,
			choice=operation,
			native_action=raw,
			provider=self.state['system1_provider'],
			fingerprint=self.state['page']['fingerprint'],
			observation_index=self.state['observation_index'],
			probabilities={},
		)
		self.state['decisions'].append(decision)
		self.state['decision_rounds'] += 1
		self.state.update(decision=decision, status='predicted')
		if self.state['model_calls']:
			self.state['model_calls'][-1].update(decision=decision, operation=operation, target=target)
		self.emit('decision', decision=decision)

	def _execute(self, expected_fingerprint):
		decision = self.state['decision']
		if decision is None:
			raise ValueError('Predict a Browser Use action first')
		self.check_cancelled()
		page = self.state['page']
		current = self.browser.observe(screenshot=False)
		if (
			current['target_id'] != page['target_id']
			or current.get('document_id') != page.get('document_id')
			or current['fingerprint'] != decision['fingerprint']
			or (expected_fingerprint is not None and expected_fingerprint != decision['fingerprint'])
		):
			self.state.update(decision=None, status='ready')
			self.observe('stale_retry')
			return
		operation = decision['operation']
		action = dict(
			kind=operation,
			operation=operation,
			target=decision['target'],
			label=decision['target'],
			native_action=decision['native_action'],
			source_url=page['url'],
			source_document=page.get('document_id'),
			step=len(self.state['history']) + 1,
		)
		self.emit('action_started', action=action)
		self.state.update(decision=None, status='ready')
		try:
			results = self._bridge.run(self._act(), cancel=self.should_cancel, timeout=150)
		except BaseException as error:
			self.emit('action_result', action=action, error=str(error) or type(error).__name__)
			raise
		result = results[-1] if results else {}
		action.update(result=results, elapsed_ms=round((time.monotonic() - self._started) * 1000))
		if operation != 'done':
			self.state['history'].append(action)
		self.emit('action_executed', action=action)
		if result.get('is_done'):
			answer = result.get('extracted_content') or result.get('long_term_memory') or ''
			self.state.update(
				status='blocked' if result.get('success') is False else 'done', plan_index=1, result={'markdown': answer}
			)
		elif not results:
			self.state.update(status='blocked', stop_reason='Browser Use returned no action result')
		self.emit('action_result', action=action, result=results)
		self.observe('step')

	def close(self):
		if self._closed:
			return
		self._closed = True
		try:
			if self.browser.native is not None:
				self._bridge.run(self._close_browser(), timeout=25)
		finally:
			self._bridge.close()
			self._temp.cleanup()

	async def _close_browser(self):
		try:
			if self.native is not None:
				await self.native.eventbus.stop(clear=True, timeout=1)
		finally:
			browser = self.browser.native
			if self._browser_start_dispatched and browser is not None:
				await browser.kill()
