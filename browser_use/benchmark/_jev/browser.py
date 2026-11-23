"""Observed actions through Browser Harness, with task-owned tab and node identity."""

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from browser_harness import _ipc
from browser_harness import helpers as harness
from browser_harness.admin import ensure_daemon
from browser_harness.helpers import cdp as _harness_cdp

# The reader preserves node identity and registers readState in the observed document.
READ_STATE = Path(__file__).with_name('snapshot.js').read_text(encoding='utf-8')
MARKER = f'(() => {{ const state={READ_STATE}; return state?.marker ?? null; }})()'
EDITOR_STATE = r"""(() => {
  const cache = window.__jevFast;
  if (!cache?.isExposed) return null;
  const editors = EDITOR_NODES.flatMap(node => {
    const e = cache.nodes.get(node);
    if (!e?.isContentEditable || !cache.clickPoint(e)) return [];
    const anchors = [];
    for (const anchor of e.querySelectorAll('a')) {
      if (!(anchor instanceof HTMLAnchorElement) || !cache.isExposed(anchor)) continue;
      const text = anchor.innerText.trim().replace(/\s+/g, ' ');
      if (!text) continue;
      const attributes = anchor.getAttributeNames()
        .filter(key => key === 'href' || (key.startsWith('data-') && key.length <= 64)).slice(0, 6);
      anchors.push({text: Array.from(text).slice(0, 120).join(''), attributes});
      if (anchors.length === 8) break;
    }
    const value = 'value' in e ? String(e.value) : e.innerText.trim();
    return [{node, value, anchors}];
  });
  return {page_key: cache.pageKey(), editors};
})()"""
SCROLL_KINDS = {'scroll', 'scroll_region_up', 'scroll_region_down'}
PAGE_RESPONSE_TIMEOUT = 15
SCROLL_POINT = """((node, delta) => {
  const cache=window.__jevFast, e=cache?.nodes.get(node), root=document.scrollingElement;
  if (!e?.isConnected) return null;
  const style=getComputedStyle(e).overflowY;
  const overflow=e===root && style==='visible' ? getComputedStyle(document.body).overflowY : style;
  if (e===root ? /hidden|clip/.test(overflow) : !/auto|scroll/.test(overflow)) return null;
  if (delta<0 ? e.scrollTop<=0 : e.scrollTop+e.clientHeight>=e.scrollHeight-2) return null;
  const point=cache.scrollPoint(e);
  return point ? {key:cache.scrollKey(e), point} : null;
})"""


class _Utf8Socket:
	"""仅调整同一 JSON 请求的编码，响应读取仍交给原 IPC 实现。"""

	def __init__(self, connection):
		self.connection = connection

	def sendall(self, data):
		request = json.loads(data)
		self.connection.sendall((json.dumps(request, ensure_ascii=False) + '\n').encode('utf-8'))

	def recv(self, size):
		return self.connection.recv(size)


def cdp(method, session_id=None, _response_timeout=harness.DEFAULT_IPC_RESPONSE_TIMEOUT_SECONDS, *, _utf8=False, **params):
	"""保留统一可替换的传输入口，仅显式请求使用 UTF-8 编码。"""
	if _utf8:
		# 沿用原连接、认证、响应解码及错误边界，只省去中文的 ASCII 转义膨胀。
		connection, token = _ipc.connect(harness.NAME, timeout=harness.IPC_CONNECT_TIMEOUT_SECONDS)
		try:
			connection.settimeout(_response_timeout)
			try:
				response = _ipc.request(
					_Utf8Socket(connection), token, {'method': method, 'params': params, 'session_id': session_id}
				)
			except TimeoutError as error:
				raise harness._IPCResponseTimeout(
					f'{method} timed out after {_response_timeout:g}s waiting for the daemon'
				) from error
		finally:
			connection.close()
		if 'error' in response:
			raise RuntimeError(response['error'])
		return response.get('result', {})
	return _harness_cdp(method, session_id=session_id, _response_timeout=_response_timeout, **params)


def _session_call(method, session, *, _utf8=False, **params):
	if method in {'Page.navigate', 'Runtime.evaluate'}:
		params['_response_timeout'] = PAGE_RESPONSE_TIMEOUT
	if _utf8:
		params['_utf8'] = True
	return cdp(method, session_id=session, **params)


class StalePage(ValueError):
	"""A decision no longer refers to the observed page."""


def _editable_guard(guard):
	description = guard[2]
	context = description.get('context', {})
	if not context.get('accessible_name'):
		return guard
	# Nearby display text is a fallback for an unnamed field. A named field still
	# needs its name, form, section and table binding to match the observation.
	description = {key: value for key, value in description.items() if key != 'label'}
	description['context'] = {key: value for key, value in context.items() if key != 'nearby_label'}
	return [*guard[:2], description, *guard[3:]]


def _scroll_point(evaluate, guard, delta):
	target = evaluate(f'{SCROLL_POINT}({guard[6]},{delta})')
	return target['point'] if target and target['key'] == guard else None


class Browser:
	def __init__(self, url):
		ensure_daemon()
		self._foreground_target = None
		# The local inspector remains the user's foreground while task tabs are controlled via CDP.
		for info in cdp('Target.getTargets').get('targetInfos', []):
			if self._is_inspector(info):
				self._foreground_target = info['targetId']
				break
		self.target = cdp('Target.createTarget', url='about:blank', background=True)['targetId']
		self._owned = {self.target: {'parent': None, 'session': None, 'node': -1}}
		self._follow = None
		try:
			self.session = cdp('Target.attachToTarget', targetId=self.target, flatten=True)['sessionId']
			self._owned[self.target]['session'] = self.session
			self._configure()
			self.call('Page.navigate', url=url)
			deadline = time.monotonic() + PAGE_RESPONSE_TIMEOUT
			while time.monotonic() < deadline:
				if self.evaluate('document.readyState') == 'complete':
					break
				time.sleep(0.02)
		except Exception:
			try:
				self.close()
			except Exception:
				pass
			raise

	def _configure(self):
		# The inspector scales the screenshot independently of the page's desktop layout.
		self.call('Emulation.setDeviceMetricsOverride', width=1366, height=780, deviceScaleFactor=1, mobile=False)
		# Keep owned background tabs rendering without activating the user's Chrome tab.
		self.call('Emulation.setFocusEmulationEnabled', enabled=True)

	@staticmethod
	def _is_inspector(info):
		try:
			url = urlsplit(info.get('url', ''))
			return (
				info.get('type') == 'page'
				and url.scheme == 'http'
				and url.hostname in {'127.0.0.1', 'localhost', '::1'}
				and url.port == int(os.getenv('TYPESAFE_DEMO_PORT', '8766'))
				and info.get('title', '').startswith('Jev Ultrafast')
			)
		except ValueError:
			return False

	def _restore_inspector(self, infos):
		target = getattr(self, '_foreground_target', None)
		if target in infos and self._is_inspector(infos[target]):
			try:
				cdp('Target.activateTarget', targetId=target)
			except RuntimeError:
				pass  # A user closing the inspector must not replay or fail a task action.

	def _activate(self, target):
		owned = self._owned[target]
		previous = self.target, self.session, getattr(self, 'after_input', None)
		try:
			if owned['session'] is None:
				owned['session'] = cdp('Target.attachToTarget', targetId=target, flatten=True)['sessionId']
			self.target, self.session = target, owned['session']
			self.after_input = None
			self._configure()
		except RuntimeError as error:
			self.target, self.session, self.after_input = previous
			if target not in self._sync_targets(follow=False):
				raise StalePage('Task-owned browser tab closed before switching. Observe again.') from error
			raise

	def _sync_targets(self, *, follow=True):
		"""Discover only observed opener descendants; retain parents after their tabs close."""
		infos = {t['targetId']: t for t in cdp('Target.getTargets').get('targetInfos', []) if t['type'] == 'page'}
		discovered = False
		while True:
			children = [t for t in infos.values() if t['targetId'] not in self._owned and t.get('openerId') in self._owned]
			if not children:
				break
			for info in children:
				discovered = True
				self._owned[info['targetId']] = {'parent': info['openerId'], 'session': None, 'node': -len(self._owned) - 1}
		if discovered:
			self._restore_inspector(infos)
		live = {target: infos[target] for target in self._owned if target in infos}
		if not follow:
			return live
		# A click gets one observation window, including the zero/multiple-child cases.
		pending, self._follow = self._follow, None
		if pending:
			parent, previous = pending
			opened = [t for t in live if t not in previous and self._owned[t]['parent'] == parent]
			if self.target == parent and len(opened) == 1:
				self._activate(opened[0])
		if self.target not in live:
			parent = self._owned.get(self.target, {}).get('parent')
			while parent and parent not in live:
				parent = self._owned[parent]['parent']
			if not parent:
				raise RuntimeError("The task's browser tab was closed")
			self._activate(parent)
		return live

	def _tab_actions(self, page, infos):
		page['dom_fingerprint'] = page['fingerprint']
		page['target_id'] = self.target
		page['tabs'] = [
			{'target_id': target, 'title': info.get('title', ''), 'url': info.get('url', '')} for target, info in infos.items()
		]
		# Append browser tabs after DOM nodes without changing existing element indices.
		index = len({a['node'] for a in page['actions'] if 'node' in a})
		for tab in page['tabs']:
			target = tab['target_id']
			if target == self.target:
				continue
			index += 1
			label = f'Browser tab · {tab["title"]} · {tab["url"]}'
			page['actions'].append(
				{
					'id': f'tab_{-self._owned[target]["node"]}',
					'node': self._owned[target]['node'],
					'kind': 'switch_tab',
					'role': 'tab',
					'label': label,
					'tab': tab,
				}
			)
			page['content'] = page.get('content', page.get('text', '')) + f'\n[{index} tab] {label}'
		page['fingerprint'] = fingerprint(page)
		return page

	def call(self, method, **params):
		return _session_call(method, self.session, **params)

	def evaluate(self, expression, *, utf8=False):
		params = {'expression': expression, 'returnByValue': True}
		if utf8:
			params['_utf8'] = True
		response = self.call('Runtime.evaluate', **params)
		if response.get('exceptionDetails'):
			raise StalePage('Document changed during evaluation')
		return response.get('result', {}).get('value')

	def observe(self, screenshot=True):
		infos = self._sync_targets() if hasattr(self, '_owned') else None
		if getattr(self, 'after_input', None):
			action, self.after_input = self.after_input, None
			# This is read-only and happens after execution was logged, even if navigation interrupts it.
			try:
				self.call(
					'Runtime.evaluate',
					expression="""(action => new Promise(resolve => {
                      const field=window.__jevFast?.nodes.get(action.node);
                      const autocomplete=action.kind==='fill' && field?.getAttribute('role')==='combobox';
                      let frames=0, stopped=false;
                      const finish=()=>{stopped=true;resolve()};
                      setTimeout(finish,autocomplete ? 200 : 50);
                      const ready=()=>{
                        if (stopped) return;
                        const ids=(field?.getAttribute('aria-controls')||field?.getAttribute('aria-owns')||'')
                          .split(/\\s+/).filter(Boolean);
                        const roots=ids.length ? ids.map(id=>document.getElementById(id)).filter(Boolean) : [document];
                        const options=roots.flatMap(root=>[...root.querySelectorAll('[role="option"]')]);
                        if (++frames>=2 && (!autocomplete || options.some(e=>{
                          const r=e.getBoundingClientRect();
                          return r.width && r.height && r.bottom>0 && r.top<innerHeight &&
                            e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
                        }))) finish();
                        else requestAnimationFrame(ready);
                      };
                      requestAnimationFrame(ready);
                    }))("""
					+ json.dumps(action)
					+ ')',
					awaitPromise=True,
					returnByValue=True,
				)
			except RuntimeError:
				pass
		for attempt in range(10):
			try:
				page = browser_operation(
					{
						'operation': 'observe',
						'session': self.session,
						'screenshot': screenshot,
					}
				)
				return self._tab_actions(page, infos) if infos is not None else page
			except StalePage:
				if attempt == 9:
					raise
				time.sleep(0.02)
		raise StalePage('Page did not settle')

	def read_editor_state(self, page, nodes):
		"""Read the Planner's observed rich-text fields from the same document and view."""
		current = self.evaluate(EDITOR_STATE.replace('EDITOR_NODES', json.dumps(nodes)))
		if current is None or current['page_key'] != page.get('page_key'):
			return None
		expected = {a['node']: a['value'] for a in page['actions'] if a['kind'] == 'fill' and a['node'] in nodes}
		editors = current['editors']
		if {editor['node']: editor['value'] for editor in editors} != expected:
			return None
		return editors

	def fresh(self, page, action=None, *, read_only=False):
		if hasattr(self, '_owned'):
			infos = self._sync_targets(follow=False)
			if page.get('target_id') != self.target or self.target not in infos:
				return False
			if action and action['kind'] == 'switch_tab':
				if action not in page['actions']:
					return False
				tab = action['tab']
				info = infos.get(tab['target_id'])
				if not info or any(info.get(k, '') != tab[k] for k in ('title', 'url')):
					return False
				read_only = True
		if read_only:
			# Snapshot capture and Planner handoffs do not execute browser input. Bind them to
			# the observed document/URL, viewport, fields and scrolls, not a changing video clock.
			expected = page.get('page_key')
			return (
				expected is not None
				and self.evaluate('(() => { const c=window.__jevFast; return c ? c.pageKey() : null; })()') == expected
			)
		if action is not None and action['kind'] in SCROLL_KINDS:
			expected = page['guards'].get(f'scroll:{action["id"]}')
			return expected is not None and _scroll_point(self.evaluate, expected, action['delta']) is not None
		if action is not None and action['kind'] in {'click', 'fill', 'select'}:
			node, kind = action['node'], action['kind']
			if type(node) is not int:
				return False
			expected = page['guards'].get(str(node))
			current = self.evaluate(
				f'(() => {{ const c=window.__jevFast; return c ? [c.pageKey(),c.guard(c.nodes.get({node}))] : null; }})()'
			)
			if expected is None or not current or current[1] is None:
				return False
			if kind == 'fill' or kind == 'click' and any(a['kind'] == 'fill' and a['node'] == node for a in page['actions']):
				return current[0][:6] == page['page_key'][:6] and _editable_guard(current[1]) == _editable_guard(expected)
			return current == [page['page_key'], expected]
		return self.evaluate(MARKER) == page['marker']

	def act(self, action, page, text=None):
		if hasattr(self, '_owned'):
			self._follow = None
		if not self.fresh(page, action):
			raise StalePage('Page changed since this decision. Observe again.')
		if action['kind'] == 'wait':
			time.sleep(0.1)
		if action['kind'] == 'switch_tab':
			self._activate(action['tab']['target_id'])
			return {'executed': action['id']}
		pending = (self.target, set(self._owned)) if action['kind'] == 'click' and hasattr(self, '_owned') else None
		request = {'operation': 'act', 'session': self.session, 'action': action, 'text': text}
		if action['kind'] in SCROLL_KINDS:
			request['scroll_guard'] = page['guards'][f'scroll:{action["id"]}']
		result = browser_operation(request)
		if pending:
			self._follow = pending
		self.after_input = action if action['kind'] != 'wait' else None
		return result

	def close(self):
		if not self.target:
			return
		if not hasattr(self, '_owned'):
			cdp('Target.closeTarget', targetId=self.target)
			self.target = None
			return
		self._follow = None
		try:
			self._sync_targets(follow=False)
		except Exception:
			pass  # Still release known tabs; the final query must confirm cleanup.
		attempted, failure = set(), None
		deadline = time.monotonic() + 2
		while True:
			for target in reversed(self._owned):
				if target in attempted:
					continue
				attempted.add(target)
				try:
					cdp('Target.closeTarget', targetId=target)
				except RuntimeError as error:
					failure = failure or error
			live = self._sync_targets(follow=False)
			if not live:
				self.target = None
				return
			if time.monotonic() >= deadline:
				raise RuntimeError(f'Task-owned browser tabs did not close: {", ".join(live)}') from failure
			if set(live).issubset(attempted):
				time.sleep(0.02)


def fingerprint(state):
	content = {k: state[k] for k in ('url', 'text', 'actions', 'scroll')}
	# 混合观察的正文顺序变化也应使旧决策失效；旧记录没有此字段时保持兼容。
	if 'content' in state:
		content['content'] = state['content']
	if 'links' in state:
		content['links'] = state['links']
	return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def browser_operation(request):
	operation = request['operation']
	session = request['session']

	def call(method, **params):
		return _session_call(method, session, **params)

	def evaluate(expression):
		result = call('Runtime.evaluate', expression=expression, returnByValue=True)
		if result.get('exceptionDetails'):
			if operation == 'act' and request['action']['kind'] == 'select':
				raise RuntimeError('Dropdown execution was interrupted; inspect before retrying.')
			raise StalePage('Document changed during evaluation')
		return result.get('result', {}).get('value')

	if operation == 'act':
		action = request['action']
		kind = action['kind']
		if kind in SCROLL_KINDS:
			point = _scroll_point(evaluate, request['scroll_guard'], action['delta'])
			if point is None:
				raise StalePage('Scroll target changed or is covered')
			call('Input.dispatchMouseEvent', type='mouseWheel', **point, deltaX=0, deltaY=action['delta'])
		elif kind != 'wait':
			if type(action['node']) is not int:
				raise ValueError('Invalid observed node')
			# Code-owned node IDs refer to actual observed elements, never model-generated selectors.
			target = evaluate(
				"""(action => {
              const e=window.__jevFast?.nodes.get(action.node);
              if (!e?.isConnected || e.matches(':disabled') || e.closest('[aria-disabled="true"],[inert]')) return null;
              if (action.kind==='fill' && (e.readOnly || e.getAttribute('aria-readonly')==='true')) return null;
              const point=window.__jevFast.clickPoint(e);
              if (!point) return null;
              const {x,y}=point;
              if (action.kind==='select') {
                if (e.tagName!=='SELECT' || ![...e.options].some(o=>o.value===action.value &&
                    !o.disabled && !o.closest('optgroup[disabled]'))) return null;
                e.value=action.value;
                e.dispatchEvent(new Event('input',{bubbles:true}));
                e.dispatchEvent(new Event('change',{bubbles:true}));
              }
              const editMode=action.kind!=='fill' ? null :
                ['INPUT','TEXTAREA'].includes(e.tagName) ? 'native' :
                e.isContentEditable ? 'contenteditable' : 'unsupported';
              return {x,y,editMode,labelFocus:editMode==='native' && point.surface!==e};
            })("""
				+ json.dumps(action)
				+ ')'
			)
			if target is None:
				if kind == 'select':
					raise RuntimeError('Dropdown execution was not confirmed; inspect before retrying.')
				raise StalePage('Target changed or is covered. Observe again.')
			if kind == 'fill' and target['editMode'] == 'unsupported':
				raise RuntimeError('Observed target does not support text replacement.')
			if kind != 'select':
				x, y = target['x'], target['y']
				for event in ('mousePressed', 'mouseReleased'):
					call(
						'Input.dispatchMouseEvent',
						type=event,
						x=x,
						y=y,
						button='left',
						clickCount=1,
					)
				if kind == 'fill':
					if target['editMode'] == 'native':
						if target.get('labelFocus'):
							# Check where the label click put focus before sending keyboard input.
							try:
								focused = evaluate(
									"""(node => {
                                  const e=window.__jevFast?.nodes.get(node);
                                  return !!e && e===document.activeElement && !e.readOnly &&
                                    !e.matches(':disabled') && e.getAttribute('aria-readonly')!=='true' &&
                                    !e.closest('[aria-disabled="true"],[inert],[aria-hidden="true"]');
                                })("""
									+ json.dumps(action['node'])
									+ ')'
								)
							except StalePage as error:
								raise RuntimeError('Document changed after label click; inspect before continuing.') from error
							if not focused:
								raise RuntimeError('Label click did not focus the editable field.')
						call(
							'Input.dispatchKeyEvent',
							type='keyDown',
							key='a',
							code='KeyA',
							modifiers=4 if sys.platform == 'darwin' else 2,
							commands=['selectAll'],
						)
						call(
							'Input.dispatchKeyEvent',
							type='keyUp',
							key='a',
							code='KeyA',
							modifiers=4 if sys.platform == 'darwin' else 2,
						)
					else:
						try:
							selected = evaluate(
								"""(node => {
                              const e=window.__jevFast?.nodes.get(node);
                              const editable=()=>e?.isConnected && e.isContentEditable && !e.readOnly &&
                                e.getAttribute('aria-readonly')!=='true' &&
                                !e.closest('[aria-disabled="true"],[inert]');
                              if (!editable()) return false;
                              // Finish synchronous focus handlers before installing a local selection.
                              e.focus({preventScroll:true});
                              if (!editable()) return false;
                              const range=document.createRange(), selection=getSelection();
                              range.selectNodeContents(e);
                              selection.removeAllRanges();
                              selection.addRange(range);
                              if (!e.isConnected || !e.isContentEditable || selection.rangeCount!==1) return false;
                              const actual=selection.getRangeAt(0), active=document.activeElement;
                              return !!active?.isContentEditable && active.contains(e) &&
                                e.contains(selection.anchorNode) && e.contains(selection.focusNode) &&
                                actual.startContainer===e && actual.startOffset===0 &&
                                actual.endContainer===e && actual.endOffset===e.childNodes.length;
                            })("""
								+ json.dumps(action['node'])
								+ ')'
							)
						except StalePage as error:
							raise RuntimeError('Text target changed after click; inspect before continuing.') from error
						if selected is not True:
							raise RuntimeError('Local text selection was not confirmed; inspect before continuing.')
					call('Input.insertText', text=request['text'])
					# Finish a plain single-line edit through native focus traversal. Recheck after
					# input: handlers may replace the field, move focus or expose suggestions.
					try:
						finish_edit = evaluate(
							"""(node => {
                          const e=window.__jevFast?.nodes.get(node);
                          if (!e?.isConnected || document.activeElement!==e || e.tagName!=='INPUT' ||
                              !['text','search','email','url','tel','number'].includes(e.type) ||
                              e.isContentEditable || e.readOnly || e.disabled ||
                              e.closest('[aria-disabled="true"],[inert],[aria-hidden="true"]') ||
                              !e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true})) return false;
                          if (e.hasAttribute('list') || e.closest('[role="combobox"]') ||
                              e.getAttribute('aria-multiline')==='true') return false;
                          const autocomplete=e.getAttribute('aria-autocomplete');
                          if (autocomplete && autocomplete!=='none') return false;
                          if (['true','listbox','tree','grid','menu'].includes(e.getAttribute('aria-haspopup')))
                            return false;
                          const ids=(e.getAttribute('aria-controls') || '')+' '+(e.getAttribute('aria-owns') || '');
                          if (ids.trim().split(/\\s+/).some(id =>
                              ['listbox','tree','grid','menu'].includes(document.getElementById(id)?.getAttribute('role'))))
                            return false;
                          return true;
                        })("""
							+ json.dumps(action['node'])
							+ ')'
						)
					except StalePage as error:
						# Text has already been inserted; this must not enter the stale-action retry path.
						raise RuntimeError(
							'Text entered but edit completion could not be inspected; observe before continuing.'
						) from error
					if finish_edit:
						for event in ('keyDown', 'keyUp'):
							call('Input.dispatchKeyEvent', type=event, key='Tab', code='Tab', windowsVirtualKeyCode=9)
		return {'executed': action['id']}

	info = evaluate(READ_STATE)
	if info is None:
		raise StalePage('Document is navigating')
	info['fingerprint'] = fingerprint(info)
	info['observed_at'] = datetime.now(timezone.utc).isoformat()
	if request.get('screenshot', True):
		info['screenshot'] = call(
			'Page.captureScreenshot',
			format='jpeg',
			quality=72,
			captureBeyondViewport=False,
			_response_timeout=PAGE_RESPONSE_TIMEOUT,
		)['data']
	return info
