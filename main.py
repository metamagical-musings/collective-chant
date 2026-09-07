import sys, os, json, time, secrets, asyncio, uvicorn, hashlib, subprocess
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, BackgroundTasks, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks
from starlette.requests import Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
from streaming_form_data import StreamingFormDataParser
from streaming_form_data.targets import ValueTarget
from streaming_form_data.validators import MaxSizeValidator, ValidationError
from functools import lru_cache
import geoip2.database
import geoip2.errors

class Const:
    error_level = 3 # 0==FATAL, 1==COURSE, 2==FINE, 3==TIMER
    words_per_token = 3
    time_slot_duration = 60 # sec
    rec_slots_per_time_slot = 1.5 # float
    rec_slot_lifespan = 6 # number of time_slots
    rec_slot_duration = rec_slot_lifespan * time_slot_duration
    rec_tolerance_sec = 11 # sec
    uploads_dir = "uploads"
    mixes_dir = "mixes"
    mix_timeout = 300
    uploads_per_mix = 2
    max_upload_size = 10 * 1024 * 1024
    max_queue_size = 100
    ref_chant = "mantra.webm"
    word_list = "bip39_english.txt"
    silence_thresh = -60 # db
    min_silence_len = 200 # ms
    target_dbfs = -14.0 # db
    padding = 0.5 # sec
    audio_bitrate = "48k" # bps

class GetCurrent:
    @staticmethod
    def time_slot_index(now):
        return now // Const.time_slot_duration
    @staticmethod
    def time_slot_start(now):
        return Const.time_slot_duration * __class__.time_slot_index(now)
    @staticmethod
    def time_slot_remaining(now):
        return __class__.time_slot_start(now) + Const.time_slot_duration - now

class Mix:
    upload_count = 0
    lock = asyncio.Lock()
    @classmethod
    async def inc_upload_count(cls):
        async with cls.lock: cls.upload_count = cls.upload_count + 1
    @classmethod
    async def reset_upload_count(cls):
        async with cls.lock: cls.upload_count = 0
        log("Broadcasting upload counter reset", 2, "[Mix]")
        await socket.broadcast("json", { "type": "upload_count", "upload_count": 0 })
    @staticmethod
    def atomic_write_binary(filename, data):
        dir_path = Path(Const.uploads_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / filename
        temp_path = file_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'wb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, file_path)
        except Exception as e:
            if temp_path.exists(): temp_path.unlink(missing_ok=True)
            log(f"Error saving upload {filename}", 0, "[Mix]")
    @staticmethod
    def launch_mix_process(mix_id):
        cmd = ["nice", "-n", "19", "ionice", "-c", "3", sys.executable, "mix_worker.py", str(mix_id), Const.uploads_dir, Const.mixes_dir, "mix.webm"]
        log(f"Launching mix process: {' '.join(cmd)}", 2, "[Mix]")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        log(f"Mix process started with PID: {process.pid}", 1,  "[Mix]")
        return
    @staticmethod
    async def wait_for_mix_completion(mix_id):
        log(f"Waiting for mix {mix_id} to complete...", 2, "[Mix]")
        mix_file = os.path.join(Const.mixes_dir, f"{mix_id}.webm")
        contributors_file = os.path.join(Const.mixes_dir, f"{mix_id}.json")
        start_time = time.time()
        while time.time() - start_time < Const.mix_timeout:
            if os.path.exists(contributors_file) and os.path.exists(mix_file):
                log(f"Mix contributors file found: {contributors_file}", 1, "[Mix]")
                log(f"Mix audio file found: {mix_file}", 1, "[Mix]")
                return (contributors_file, mix_file)
            await asyncio.sleep(1)
        raise TimoutError(f"Mix {mix_id} did not complete within {Const.mix_timeout} seconds.")
    @staticmethod
    async def broadcast(contributors_path, mix_path):
        with open(mix_path, 'rb') as g: bytes = g.read()
        await socket.broadcast("bytes", bytes)
        with open(contributors_path, 'r') as f:
            contributors = json.loads(f.read())
            contributors = [{token: IP.get_country_code(ip)} for d in contributors for token, ip in d.items()]
        await socket.broadcast("json", {"type": "mix_contributors", "contributors": contributors})

class Chant:
    ref_chant = None
    @staticmethod
    def load_ref_chant():
        with open(f"static/{Const.ref_chant}", "rb") as f: Chant.ref_chant = f.read()

class WS:
    def __init__(self):
        self.client_counter = 0
        self.client_lock = asyncio.Lock()
        self.clients = {} # client_id -> {"ws": websocket, "q": queue}
    async def broadcast(self, mode, packet, client_id=None): # mode: "json" or "bytes"
        sent = 0; failed = 0
        if client_id is not None: client_list = [(client_id, socket.clients[client_id])]
        else: client_list = list(socket.clients.items())
        for client_id, client_data in client_list:
            try:
                await client_data["q"].put({ "mode": mode, "packet": packet })
                sent += 1
            except Exception as e:
                failed += 1
                log(f"Failed to broadcast to client #{client_id}: {e}", 1, "[WS]")
                await client_data["ws"].close()
#        if sent > 0 or failed > 0: log(f"Broadcast: {sent} sent, {failed} failed", 2, "[WS]")

class TokenState:
    def __init__(self):
        self.tokens = {}
        self.lock = asyncio.Lock()
    async def _cleanup_expired(self):
        now = int(time.time())
        for token in [token for token, data in self.tokens.items() if data["expires_at"] < now]:
            del self.tokens[token]
    async def add_token(self, token, expires_at):
        async with self.lock:
            await self._cleanup_expired()
            self.tokens[token] = {"granted": False, "expires_at": expires_at}
    async def grant_token(self, token):
        async with self.lock:
            await self._cleanup_expired()
            if token in self.tokens:
                self.tokens[token]["granted"] = True
                return True
            return False
    async def validate_token(self, token):
        async with self.lock:
            await self._cleanup_expired()
            if token in self.tokens:
                data = self.tokens[token]
                now = int(time.time())
                if data["granted"] and now < data["expires_at"]:
                    del self.tokens[token]
                    return True
            return False
    async def get_token_count(self): # current session only
        async with self.lock:
            await self._cleanup_expired()
            return len(self.tokens)
    async def get_granted_token_count(self): # current session only
        async with self.lock:
            await self._cleanup_expired()
            return sum(1 for data in self.tokens.values() if data["granted"])

class BIP39WordList:
    def __init__(self):
        self.words = []
        self.path = Path(f"static/{Const.word_list}")
        with open(self.path, "r", encoding="utf-8") as f: self.words = [line.strip() for line in f if line.strip()]

    def generate_token(self):
        return "-".join(secrets.SystemRandom().choices(self.words, k=Const.words_per_token))

class IP:
    DB_PATH = Path("static/GeoLite2-Country.mmdb")
    @staticmethod
    @lru_cache(maxsize=1)
    def _get_reader():
        if not IP.DB_PATH.exists(): raise FileNotFoundError(f"GeoIP database not found at {IP.DB_PATH}.")
        return geoip2.database.Reader(str(IP.DB_PATH))
    @staticmethod
    def get_client_ip(request: Request):
        if cf_ip := request.headers.get("cf-connecting-ip"): return cf_ip.strip()
        if xff := request.headers.get("x-forwarded-for"): return xff.split(",")[0].strip()
        if real_ip := request.headers.get("x-real-ip"): return real_ip.strip()
        return request.client.host if request.client else "127.0.0.1"
    @staticmethod
    def get_country_code(ip):
        if not ip or ip in ("127.0.0.1", "::1", "0.0.0.0"): return "xx"
        try:
            reader = IP._get_reader()
            response = reader.country(ip)
            return response.country.iso_code
        except (geoip2.errors.AddressNotFoundError, ValueError): return "xx"
        except Exception as e:
            # Log this in production if needed
            return "xx"

@asynccontextmanager
async def lifespan(app: FastAPI):
    log("Server started. Waiting for connections...", 1, "[App]")
    app.state.background_tasks = [
        asyncio.create_task(mix_worker(), name="mix_worker"),
        asyncio.create_task(token_worker(), name="token_worker"),
    ]
    log("Lifespan startup successful", 1, "[App]")
    yield  # The app runs here
    for task in app.state.background_tasks:
        if task.done(): continue
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass
        except Exception as e: log(f"Error while cancelling background task: {e}", 0, "[App]")

app = FastAPI(title="Universe", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_methods=["*"], allow_headers=["*"], allow_credentials=True, # remove null for production
    allow_origins=["null", "http://localhost", "http://127.0.0.1", "http://192.168.18.56", "http://192.168.18.*"])
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
Chant.load_ref_chant()
wordlist = BIP39WordList()
socket = WS()
token_store = TokenState()
Path(Const.uploads_dir).mkdir(parents=True, exist_ok=True)

async def mix_worker():
    log(f"Starting background worker...", 1, "[Mix]")
    while True:
        count = Mix.upload_count
#        log(f"Broadcasting upload count: {count}", 2, "[Mix]")
        await socket.broadcast("json", { "type": "upload_count", "upload_count": count })
        if count >= Const.uploads_per_mix:
            log("Initiating mix...", 1, "[Mix]")
            await Mix.reset_upload_count()
            mix_id = int(time.time())
            try:
                Mix.launch_mix_process(mix_id)
                contributors_path, mix_path = await Mix.wait_for_mix_completion(mix_id)
                await Mix.broadcast(contributors_path, mix_path)
            except TimeoutError: log(f"Error: Mix {mix_id} timed out", 1, "[Mix]")
            except Exception as e: log(f"Error in mix_worker loop: {e}", 1, "[Mix]")
        await asyncio.sleep(5)

async def token_worker():
    def get_prev_allocations(index):
        return round(index * Const.rec_slots_per_time_slot)
    log(f"Starting background worker...", 1, "[Token]")
    next_time_slot_index = None
    while True:
        now = int(time.time())
        time_slot_index = GetCurrent.time_slot_index(now)
        if next_time_slot_index == None: next_time_slot_index = time_slot_index + 1
        if time_slot_index != next_time_slot_index:
            sleep = GetCurrent.time_slot_remaining(now)
            log(f"Sleeping for {sleep} sec...", 2, "[Token]")
            await asyncio.sleep(sleep)
            continue
        next_time_slot_index = time_slot_index + 1
        log(f"New time slot #{time_slot_index}", 2, "[Token]")
        new_allocations = get_prev_allocations(time_slot_index + 1) - get_prev_allocations(time_slot_index)
        new_tokens = []
        expires_at = GetCurrent.time_slot_start(now) + Const.rec_slot_duration
        for i in range(new_allocations):
            token = wordlist.generate_token()
            await token_store.add_token(token, expires_at)
            new_tokens.append(token)
            log(f"New token added: {token}", 2, "[Token]")
        log(f"Unexpired token count: {await token_store.get_token_count()}", 2, "[Token]")
        client_list = list(socket.clients)
        if client_list:
            if len(client_list) <= new_allocations: recipients = client_list
            else: recipients = secrets.SystemRandom().sample(client_list, new_allocations)
            for i, client in enumerate(recipients):
                if i < len(new_tokens):
                    token = new_tokens[i]
                    client_dict = socket.clients.get(client)
                    if client_dict == None: continue
                    queue = client_dict.get("q")
                    try:
                        queue.put_nowait({"mode": "json", "packet": { "type": "rec_slot_grant", "token": token, "expires_at": expires_at}})
                        log(f"Granting new token {token} to client #{client}", 2, "[Token]")
                        await token_store.grant_token(token)
                    except asyncio.QueueFull: pass

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    async def handle_client_message(client_id, data, queue):
        msg_type = data.get("type") or data.get("action")
        log(f"Client #{client_id} → type: {msg_type} | payload: {data}", 3, "[WS]")
        if msg_type == "ping": await queue.put({ "mode": "json", "packet": { "type": "pong", "timestamp": asyncio.get_event_loop().time() }})
        else: await queue.put({ "mode": "json", "packet": { "type": "echo", "original": data, "message": "Message received" }}) # debugging
    client_id = None; queue = None
    tasks = []
    try:
        await websocket.accept()
        async with socket.client_lock:
            socket.client_counter += 1
            client_id = socket.client_counter
        queue = asyncio.Queue(maxsize=Const.max_queue_size)
        socket.clients[client_id] = {"ws": websocket, "q": queue}
        log(f"Client #{client_id} connected.", 2, "[WS]")
        now = time.time() # float
        await websocket.send_json({"type": "sync", "client_id": client_id, "now": now})
        await socket.broadcast("bytes", Chant.ref_chant, client_id)
        async def sender():
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=1.0)
                    if msg.get("mode") == "json": await websocket.send_json(msg["packet"])
                    else:
                        await websocket.send_bytes(msg.get("packet", b""))
                except asyncio.TimeoutError: continue
                except Exception as e: break
        async def receiver():
            while True:
                try:
                    data = await websocket.receive_json()
                    log(f"Client #{client_id} sent: {json.dumps(data, indent=2)}", 2, "[WS]")
                    await handle_client_message(client_id, data, queue)
                except WebSocketDisconnect: raise
                except Exception as e:
                    log(f"Receiver error for client {client_id}: {e}", 1, "[WS]")
                    break
        async with asyncio.TaskGroup() as tg:
            tasks.append(tg.create_task(sender()))
            tasks.append(tg.create_task(receiver()))
    except* WebSocketDisconnect: log(f"Client #{client_id} disconnected.", 1, "[WS]") # Note: except* (exception group)
    except* Exception as eg: log(f"Error with client #{client_id}: {eg}", 1, "[WS]")
    finally:
        if client_id and client_id in socket.clients: socket.clients.pop(client_id, None)
        for t in tasks:
            if not t.done(): t.cancel()

@app.post("/upload")
async def upload_endpoint(request: Request):
    def convert_to_mb(size):
        return f"{size // (1024 * 1024)} MB"
    log("Handling upload request...", 1, "[Upload]")
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > Const.max_upload_size + (512 * 1024):  # small overhead
        return { "status": f"Upload request ({convert_to_mb(int(content_length))}) exceeds maximum allowed: {convert_to_mb(Const.max_upload_size)}." }
    log("WebM content length OK", 2, "[Upload]")
    parser = StreamingFormDataParser(request.headers)
    file_target = ValueTarget(validator=MaxSizeValidator(Const.max_upload_size))
    token_target = ValueTarget()
    parser.register("file", file_target)
    parser.register("token", token_target)
    token = None
    webm_checked = False
    try:
        async for chunk in request.stream():
            parser.data_received(chunk)
            if token is None:
                token = token_target.value.decode()
                if token is None or not await token_store.validate_token(token):
                    return { "status": "Token missing, invalid, expired or already used." }
                log(f"Valid token: {token}", 2, "[Upload]")
            if not webm_checked:
                bytes = file_target.value
                if len(bytes) >= 4 and bytes.startswith(b"\x1A\x45\xDF\xA3"):
                    webm_checked = True
                    log("WebM magic bytes found", 2, "[Upload]")
                    continue
                return { "status": "Invalid audio file: not WebM." }
    except ValidationError as e:
        return { "status": f"Upload request exceeds maximum allowed: {convert_to_mb(Const.max_upload_size)}." }
    log(f"Size within limit: {convert_to_mb(Const.max_upload_size)}", 2, "[Upload]")
    if not file_target.value: return { "status": "Incomplete audio upload." }
    file_bytes = file_target.value
    rec_size = len(file_bytes)
    log(f"Actual audio rec_size OK: {rec_size}", 2, "[Upload]")
    md5_hash = hashlib.md5(file_bytes).hexdigest()
    log(f"Audio hash: {md5_hash}", 2 , "[Upload]")
    existing_files = list(Path(Const.uploads_dir).glob(f"{md5_hash}.*.webm"))
    if existing_files: return { "status": "Audio file previously uploaded." }
    log("Non-duplicate audio file", 2 , "[Upload]")
    ip = IP.get_client_ip(request)
    filename = f"{md5_hash}.{token}.{ip}.webm"
    Mix.atomic_write_binary(filename, file_bytes)
    log(f"Audio file saved to: {filename}", 2, "[Upload]")
    await Mix.inc_upload_count()
    return { "status": "Upload successful." }

@app.get("/status")
async def status_endpoint():
    now = int(time.time())
    return {
        "current_time": now,
        "time_slot_duration": Const.time_slot_duration,
        "current_time_slot_index": GetCurrent.time_slot_index(now),
        "current_time_slot_remaining": GetCurrent.time_slot_remaining(now),
        "rec_slot_duration": Const.rec_slot_duration,
        "uploads_per_mix": Const.uploads_per_mix,
        "upload_count": Mix.upload_count,
        "token_count": await token_store.get_token_count(),
        "granted_token_count": await token_store.get_granted_token_count(),
    }

@app.get("/flag/{country_code}")
def get_flag(country_code: str):
    file = Path(f"static/flags/{country_code.lower()}.png")
    if not file.exists(): file = Path("static/flags/xx.png")
    return FileResponse(file)

@app.get("/")
async def root(request: Request):
    context = {
        "request": request,
        "time_slot_duration": Const.time_slot_duration,
        "rec_slot_duration": Const.rec_slot_duration,
        "rec_tolerance_sec": Const.rec_tolerance_sec,
        "uploads_per_mix": Const.uploads_per_mix,
        "max_upload_size": Const.max_upload_size,
    }
    return templates.TemplateResponse(request, "index.html", context)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")

def log(message, level, worker=""):
    if level > Const.error_level: return
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {worker} {message}")
    if level == 0: sys.exit(1)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=443,
        ssl_keyfile="fastapi.key",
        ssl_certfile="fastapi.crt"
    )
