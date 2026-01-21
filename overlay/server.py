def _deep_update(dst: dict, src: dict) -> dict:
    """Recursively update dst with src. Ignores None values.

    This prevents partial payloads from wiping nested dictionaries like 'stats' and 'game_state'.
    """
    if not isinstance(dst, dict) or not isinstance(src, dict):
        return dst
    for k, v in src.items():
        if v is None:
            continue
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst
"""
Web server for Pokémon Yellow RL overlay in OBS.
Serves HTML/CSS/JS overlay and provides a real WebSocket for live updates.
"""
import asyncio
import json
import os
import threading
import time
from aiohttp import web, WSMsgType

# Embedded HTML overlay (kept simple for the prototype)
OVERLAY_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Pokémon Yellow RL Overlay</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: 'Courier New', monospace; background: rgba(0,0,0,0.8); color: #00ff00; margin:0; padding:10px; font-size:14px; overflow:hidden }
        .container { display: grid; grid-template-columns: 1fr 1fr; gap:10px; max-width:800px }
        .panel { background: rgba(0,30,0,0.7); border:1px solid #00ff00; padding:10px; border-radius:5px }
        .panel h2 { margin-top:0; font-size:16px; color:#00ff00 }
        .commentary { font-style: italic; min-height:40px; padding:5px; background: rgba(0,20,0,0.5) }
        .stats-grid { display:grid; grid-template-columns: 1fr 1fr; gap:5px }
        .stat { display:flex; justify-content:space-between }
        .confidence-bar { height:10px; background:#333; border-radius:5px; margin-top:2px }
        .confidence-fill { height:100%; background:#00ff00; border-radius:5px }
        .action { display:flex; justify-content:space-between; margin:2px 0 }
    </style>
</head>
<body>
    <div class="container">
        <div class="panel">
            <h2>Game State</h2>
            <div id="location">Location: Pallet Town</div>
            <div id="position">Position: (0, 0)</div>
            <div id="status">Status: Normal</div>
        </div>
        <div class="panel">
            <h2>Statistics</h2>
            <div class="stats-grid">
                <div class="stat"><span>Tiles Visited:</span> <span id="tiles">0</span></div>
                <div class="stat"><span>Battles Won:</span> <span id="battles-won">0</span></div>
                <div class="stat"><span>Battles Lost:</span> <span id="battles-lost">0</span></div>
                <div class="stat"><span>Badges:</span> <span id="badges">0</span></div>
                <div class="stat"><span>Money:</span> <span id="money">0</span></div>
                <div class="stat"><span>Episode Steps:</span> <span id="steps">0</span></div>
                <div class="stat"><span>Global Steps:</span> <span id="global-steps">0</span></div>
                <div class="stat"><span>Phase:</span> <span id="phase">-</span></div>
                <div class="stat"><span>Phase Steps:</span> <span id="phase-steps">0</span></div>
            </div>
        </div>
        <div class="panel" style="grid-column: span 2;"><h2>Commentary</h2><div id="commentary" class="commentary">Welcome to Pokémon Yellow RL training!</div></div>
        <div class="panel" style="grid-column: span 2;"><h2>Action Confidence</h2>
            <div class="action"><span>Up:</span> <span id="conf-up">0%</span><div class="confidence-bar"><div id="conf-up-fill" class="confidence-fill" style="width: 0%"></div></div></div>
            <div class="action"><span>Down:</span> <span id="conf-down">0%</span><div class="confidence-bar"><div id="conf-down-fill" class="confidence-fill" style="width: 0%"></div></div></div>
            <div class="action"><span>Left:</span> <span id="conf-left">0%</span><div class="confidence-bar"><div id="conf-left-fill" class="confidence-fill" style="width: 0%"></div></div></div>
            <div class="action"><span>Right:</span> <span id="conf-right">0%</span><div class="confidence-bar"><div id="conf-right-fill" class="confidence-fill" style="width: 0%"></div></div></div>
            <div class="action"><span>A:</span> <span id="conf-a">0%</span><div class="confidence-bar"><div id="conf-a-fill" class="confidence-fill" style="width: 0%"></div></div></div>
            <div class="action"><span>B:</span> <span id="conf-b">0%</span><div class="confidence-bar"><div id="conf-b-fill" class="confidence-fill" style="width: 0%"></div></div></div>
        </div>
    </div>
    <script>
        const url = (location.protocol === 'https:') ? 'wss://' + location.host + '/ws' : 'ws://' + location.host + '/ws';
        const ws = new WebSocket(url);
        ws.onmessage = function(ev) {
            const data = JSON.parse(ev.data);
            document.getElementById('location').textContent = 'Location: ' + (data.game_state.location || 'Unknown');
            document.getElementById('position').textContent = 'Position: (' + (data.game_state.x||0) + ', ' + (data.game_state.y||0) + ')';
            document.getElementById('status').textContent = 'Status: ' + (data.game_state.in_battle ? 'In Battle' : data.game_state.in_menu ? 'In Menu' : 'Normal');
            document.getElementById('tiles').textContent = data.stats.tiles_visited||0;
            document.getElementById('battles-won').textContent = data.stats.battles_won||0;
            document.getElementById('battles-lost').textContent = data.stats.battles_lost||0;
            document.getElementById('badges').textContent = data.stats.badges||0;
            document.getElementById('money').textContent = data.stats.money||0;
            document.getElementById('steps').textContent = data.stats.episode_steps||0;
            document.getElementById('global-steps').textContent = data.stats.global_steps||0;
            document.getElementById('phase').textContent = (data.stats.phase === null || data.stats.phase === undefined) ? '-' : data.stats.phase;
            document.getElementById('phase-steps').textContent = data.stats.phase_steps||0;
            document.getElementById('commentary').textContent = data.commentary||'';
            const updateConfidence = (k,v)=>{ const percent = Math.round((v||0)*100); const el = document.getElementById('conf-'+k); const fill = document.getElementById('conf-'+k+'-fill'); if(el && fill){ el.textContent = percent + '%'; fill.style.width = percent + '%'; }};
            ['up','down','left','right','a','b'].forEach(k=>updateConfidence(k, data.action_confidence?.[k]));
        }
        ws.onopen = function(){ console.log('Overlay websocket connected') }
        ws.onclose = function(){ console.log('Overlay websocket closed') }
    </script>
</body>
</html>
"""


class OverlayServer:
    """A small overlay server using aiohttp that supports a real WebSocket.

    - GET / -> serves the overlay HTML
    - GET /data -> returns the latest JSON data
    - GET /ws -> websocket endpoint that broadcasts updates
    - POST /update -> accepts JSON payloads to update overlay data

    The server can run in a background thread and exposes a thread-safe update_data() method
    that other threads (trainer/showcase) can call.
    """

    def __init__(self, host='0.0.0.0', port=8080, loop=None):
        self.host = host
        self.port = int(port)
        self.app = web.Application()
        self.app.add_routes([
            web.get('/', self.handle_index),
            web.get('/data', self.handle_data),
            web.get('/ws', self.handle_ws),
            web.post('/update', self.handle_update),
        ])

        # Internal state
        self.overlay_data = {
            'game_state': {'map_id': 0, 'location': 'Pallet Town', 'x': 0, 'y': 0, 'in_battle': False, 'in_menu': False},
            'stats': {'tiles_visited': 0, 'battles_won': 0, 'battles_lost': 0, 'badges': 0, 'money': 0, 'episode_steps': 0},
            'commentary': 'Welcome to Pokémon Yellow RL training!',
            'action_confidence': {'up':0.0,'down':0.0,'left':0.0,'right':0.0,'a':0.0,'b':0.0},
            'timestamp': time.time()
        }

        # WebSocket clients
        self._clients = set()

        # Thread-safe queue for updates from other threads
        self._update_queue = None

        # Runner and site for aiohttp
        self.runner = web.AppRunner(self.app)
        self._server_thread = None
        self._loop = loop
        self._shutdown = threading.Event()

    async def handle_index(self, request):
        return web.Response(text=OVERLAY_HTML, content_type='text/html')

    async def handle_data(self, request):
        return web.json_response(self.overlay_data)

    async def handle_update(self, request):
        try:
            payload = await request.json()
        except Exception:
            return web.Response(status=400, text='invalid json')


        if self._update_queue is None:
            return web.Response(status=503, text="server not ready")
        await self._update_queue.put(payload)
        return web.Response(status=200, text='ok')

    async def handle_ws(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # send current state immediately
        await ws.send_json(self.overlay_data)

        self._clients.add(ws)
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    # Echo or accept simple commands if needed
                    # For now, we just echo back any text messages
                    await ws.send_str(msg.data)
                elif msg.type == WSMsgType.ERROR:
                    print('ws connection error', ws.exception())
                elif msg.type == WSMsgType.CLOSE:
                    print('ws connection closed')
        finally:
            self._clients.discard(ws)
        return ws

    async def _broadcast(self, data):
        if not self._clients:
            return
        payload = json.dumps(data)
        to_remove = []
        for ws in list(self._clients):
            try:
                await ws.send_str(payload)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                try:
                    await ws.close()
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                to_remove.append(ws)
        for ws in to_remove:
            self._clients.discard(ws)

    async def _update_worker(self):
        """Background task that consumes the update queue and broadcasts updates."""
        while True:
            payload = await self._update_queue.get()
            # merge payload into overlay_data using deep update
            try:
                _deep_update(self.overlay_data, payload)

                # Ensure required top-level keys always exist
                self.overlay_data.setdefault('game_state', {})
                self.overlay_data.setdefault('stats', {})
                self.overlay_data.setdefault('action_confidence', {})
                self.overlay_data.setdefault('commentary', '')

                # Ensure required nested keys exist (so the overlay UI never breaks)
                gs = self.overlay_data['game_state']
                gs.setdefault('location', 'Unknown')
                gs.setdefault('map_id', 0)
                gs.setdefault('x', 0)
                gs.setdefault('y', 0)
                gs.setdefault('in_battle', False)
                gs.setdefault('in_menu', False)

                st = self.overlay_data['stats']
                st.setdefault('tiles_visited', 0)
                st.setdefault('battles_won', 0)
                st.setdefault('battles_lost', 0)
                st.setdefault('badges', 0)
                st.setdefault('money', 0)
                st.setdefault('episode_steps', 0)
                # Newer optional stats (kept safe for both training + showcase payloads)
                st.setdefault('global_steps', 0)
                st.setdefault('phase', None)
                st.setdefault('phase_steps', 0)
                st.setdefault('total_reward', 0.0)

                ac = self.overlay_data['action_confidence']
                for k in ['up','down','left','right','a','b']:
                    ac.setdefault(k, 0.0)

                self.overlay_data['timestamp'] = time.time()
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

            await self._broadcast(self.overlay_data)

    def update_data(self, data: dict):
        if not self._loop or not self._loop.is_running() or self._update_queue is None:
            return  # server not started yet

        asyncio.run_coroutine_threadsafe(self._update_queue.put(data), self._loop)

    def start(self):
        """Start the aiohttp server in a background thread."""
        if self._server_thread and self._server_thread.is_alive():
            return

        def _run():
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)

            self._update_queue = asyncio.Queue()
            async def _runner():
                await self.runner.setup()
                site = web.TCPSite(self.runner, self.host, self.port)
                await site.start()
                # start update worker
                loop.create_task(self._update_worker())
                print(f'Overlay server running on http://{self.host}:{self.port}')
                while not self._shutdown.is_set():
                    await asyncio.sleep(0.5)

            try:
                loop.run_until_complete(_runner())
            except Exception as e:
                print('Overlay server error:', e)
            finally:
                loop.run_until_complete(self.runner.cleanup())
                loop.close()

        self._server_thread = threading.Thread(target=_run, daemon=True)
        self._server_thread.start()

    def stop(self):
        self._shutdown.set()
        if self._loop and self._loop.is_running():
            async def _shutdown_async():
                await self.runner.cleanup()
            asyncio.run_coroutine_threadsafe(_shutdown_async(), self._loop)
        if self._server_thread:
            self._server_thread.join(timeout=2)


def main():
    server = OverlayServer(host='127.0.0.1', port=8080)
    try:
        server.start()
        print('Press Ctrl+C to stop overlay server')
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('Stopping overlay server')
        server.stop()


if __name__ == '__main__':
    main()
