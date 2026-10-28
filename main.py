import sys, os, json, time, secrets, asyncio, uvicorn, hashlib
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
    error_level = 2 # 0==FATAL, 1==COURSE, 2==FINE
    words_per_token = 3
    time_slot_duration_sec = 60
    rec_slots_per_time_slot = 1.5 # float
    rec_slot_lifespan = 6 # number of time_slots
    rec_slot_duration_sec = rec_slot_lifespan * time_slot_duration_sec
    mix_timeout_sec = 300
    uploads_per_mix = 2
    max_upload_size = 10 * 1024 * 1024
    max_queue_size = 100
    uploads_dir = "uploads"
    mixes_dir = "mixes"
    archive_dir = "mixes/archive"
    mix_id_path = "./mix_id.json"
    word_list = "static/bip39_english.txt"
    flags_dir = "static/flags"
    countries_db = "static/GeoLite2-Country.mmdb"
    favicon = "static/favicon.ico"
    ref_audio_path = "reference/chant.webm"
    ref_ticks_path = "reference/chant_ticks.webm"
    ref_stanzas_path = "reference/chant_stanzas.txt"

class Util:
    @staticmethod
    def convert_to_mb(size):
        return f"{size // (1024 * 1024)} MB"
    @staticmethod
    def atomic_write_binary(file_path, data):
        temp_path = file_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'wb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, file_path)
        except Exception as e:
            if temp_path.exists(): temp_path.unlink(missing_ok=True)
            log(f"Error saving upload {filename}", 1, "[Mix]")

class GetCurrent:
    @staticmethod
    def time_slot_index(now):
        return now // Const.time_slot_duration_sec
    @staticmethod
    def time_slot_start(now):
        return Const.time_slot_duration_sec * __class__.time_slot_index(now)
    @staticmethod
    def time_slot_remaining(now):
        return __class__.time_slot_start(now) + Const.time_slot_duration_sec - now

class Mix:
    upload_count = 0
    mix_id = 0
    epoch = 0
    mix_id_path = Path(Const.mix_id_path)
    mixes_dir_path = Path(Const.mixes_dir)
    archive_dir_path = Path(Const.archive_dir)
    lock = asyncio.Lock()
    @classmethod
    def save_mix_id(cls):
        cls.mix_id_path.write_text(json.dumps({"mix_id": cls.mix_id, "epoch": cls.epoch}, indent=2), encoding="utf-8")
    @classmethod
    def load_mix_id(cls):
        if not cls.mix_id_path.exists():
            cls.epoch = int(time.time())
            cls.save_mix_id()
        else:
            data = json.loads(cls.mix_id_path.read_text(encoding="utf-8"))
            cls.mix_id = data["mix_id"]
            cls.epoch = data["epoch"]
    @classmethod
    async def inc_mix_id(cls):
        async with cls.lock:
            cls.mix_id += 1
            cls.save_mix_id()
    @classmethod
    async def inc_upload_count(cls):
        async with cls.lock: cls.upload_count += 1
    @classmethod
    async def mix_initiation(cls):
        log("Broadcasting mix {Mix.mix_id} initiation", 2, "[Mix]")
        await WS.broadcast("json", { "type": "mix_initiation", "mix_id": cls.mix_id })
    @classmethod
    async def reset_upload_count(cls):
        async with cls.lock: cls.upload_count = 0
        log("Broadcasting upload counter reset", 2, "[Mix]")
        await WS.broadcast("json", { "type": "upload_count", "upload_count": 0 })
    @classmethod
    async def launch_mix_process(cls):
        async with cls.lock:
            mix_path = cls.mixes_dir_path / f"{cls.mix_id}.mix"
            mix_path.touch()
        log(f"Mix {cls.mix_id} process started.", 1,  "[Mix]")
    @classmethod
    async def wait_for_mix_completion(cls):
        log(f"Waiting for mix {cls.mix_id} to complete...", 2, "[Mix]")
        mix_path = cls.mixes_dir_path / f"{cls.mix_id}.webm"
        contributors_path = cls.mixes_dir_path / f"{cls.mix_id}.json"
        start_time = time.time()
        while time.time() - start_time < Const.mix_timeout_sec:
            if contributors_path.exists() and mix_path.exists():
                log(f"Mix contributors file found: {contributors_path}", 1, "[Mix]")
                log(f"Mix audio file found: {mix_path}", 1, "[Mix]")
                return (contributors_path, mix_path)
            await asyncio.sleep(1)
        raise TimoutError(f"Mix {cls.mix_id} did not complete within {Const.mix_timeout_sec} seconds.")
    @classmethod
    async def archive_mix(cls, contributors_path, mix_path):
        async with cls.lock:
            mix_path.rename(cls.archive_dir_path / mix_path.name)
            contributors_path.rename(cls.archive_dir_path / contributors_path.name)
        log(f"Mix {cls.mix_id} moved to archive.", 1,  "[Mix]")
    @classmethod
    async def broadcast(cls, contributors_path, mix_path):
        with open(mix_path, 'rb') as g: bytes = g.read()
        with open(contributors_path, 'r') as f:
            contributors = json.loads(f.read())
            contributors = [{token: IP.get_country_code(ip)} for d in contributors for token, ip in d.items()]
        await WS.broadcast("json", {"type": "mix_contributors", "contributors": contributors, "mix_id": cls.mix_id})
        await WS.broadcast("bytes", bytes)

class Chant:
    ref_audio = None
    ref_ticks = None
    ref_stanzas = None
    @classmethod
    def load_references(cls):
        def get_refs(ref_paths):
            refs = []
            for ref_path in ref_paths:
                path = Path(ref_path)
                if not path.is_file(): log(f"Fatal: reference file not found: {path.name}", 0, "[App]")
                if path.suffix.lower() == ".txt": refs.append(path.read_text(encoding="utf-8"))
                else: refs.append(path.read_bytes())
            return tuple(refs)
        cls.ref_audio, cls.ref_ticks, cls.ref_stanzas = get_refs([Const.ref_audio_path, Const.ref_ticks_path, Const.ref_stanzas_path])

class WS:
    client_counter = 0
    client_lock = asyncio.Lock()
    clients = {} # client_id -> {"ws": websocket, "q": queue}
    @classmethod
    async def broadcast(cls, mode, packet, client_id=None): # mode: "json" or "bytes"
        sent = 0; failed = 0
        if client_id is not None: client_list = [(client_id, WS.clients[client_id])]
        else: client_list = list(cls.clients.items())
        for client_id, client_data in client_list:
            try:
                await client_data["q"].put({ "mode": mode, "packet": packet })
                sent += 1
            except Exception as e:
                failed += 1
                log(f"Failed to broadcast to client #{client_id}: {e}", 1, "[WS]")
                await client_data["ws"].close()
#        if sent > 0 or failed > 0: log(f"Broadcast: {sent} sent, {failed} failed", 2, "[WS]")

class TokenStore:
    tokens = {}
    lock = asyncio.Lock()
    @classmethod
    async def _cleanup_expired(cls):
        now = int(time.time())
        for token in [token for token, data in cls.tokens.items() if data["expires_at"] < now]:
            del cls.tokens[token]
    @classmethod
    async def add_token(cls, token, expires_at):
        async with cls.lock:
            await cls._cleanup_expired()
            cls.tokens[token] = {"granted": False, "expires_at": expires_at}
    @classmethod
    async def grant_token(cls, token):
        async with cls.lock:
            await cls._cleanup_expired()
            if token in cls.tokens:
                cls.tokens[token]["granted"] = True
                return True
            return False
    @classmethod
    async def validate_token(cls, token):
        async with cls.lock:
            await cls._cleanup_expired()
            if token in cls.tokens:
                data = cls.tokens[token]
                now = int(time.time())
                if data["granted"] and now < data["expires_at"]:
                    del cls.tokens[token]
                    return True
            return False
    @classmethod
    async def get_token_count(cls): # current session only
        async with cls.lock:
            await cls._cleanup_expired()
            return len(cls.tokens)
    @classmethod
    async def get_granted_token_count(cls): # current session only
        async with cls.lock:
            await cls._cleanup_expired()
            return sum(1 for data in cls.tokens.values() if data["granted"])

class BIP39WordList:
    words = []
    @classmethod
    def load_words(cls):
        with open(Const.word_list, "r", encoding="utf-8") as f: cls.words = [line.strip() for line in f if line.strip()]
    @classmethod
    def generate_token(cls):
        return "-".join(secrets.SystemRandom().choices(cls.words, k=Const.words_per_token))

class IP:
    db_path = Path(Const.countries_db)
    @staticmethod
    @lru_cache(maxsize=1)
    def _get_reader():
        if not IP.db_path.exists(): raise FileNotFoundError(f"GeoIP database not found at {IP.db_path}.")
        return geoip2.database.Reader(str(IP.db_path))
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

def log(message, level, worker=""):
    if level > Const.error_level: return
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {worker} {message}")
    if level == 0: sys.exit(1)

def create_app():
    templates = Jinja2Templates(directory="templates")
    async def mix_worker():
        log(f"Starting background worker...", 1, "[Mix]")
        while True:
            count = Mix.upload_count
    #        log(f"Broadcasting upload count: {count}", 2, "[Mix]")
            await WS.broadcast("json", { "type": "upload_count", "upload_count": count })
            if count >= Const.uploads_per_mix:
                await Mix.inc_mix_id()
                await Mix.mix_initiation()
                try:
                    await Mix.launch_mix_process()
                    contributors_path, mix_path = await Mix.wait_for_mix_completion()
                    await Mix.reset_upload_count()
                    await Mix.broadcast(contributors_path, mix_path)
                    await Mix.archive_mix(contributors_path, mix_path)
                except TimeoutError: log(f"Error: Mix {Mix.mix_id} timed out", 1, "[Mix]")
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
            expires_at = GetCurrent.time_slot_start(now) + Const.rec_slot_duration_sec
            for i in range(new_allocations):
                token = BIP39WordList.generate_token()
                await TokenStore.add_token(token, expires_at)
                new_tokens.append(token)
                log(f"New token added: {token}", 2, "[Token]")
            log(f"Unexpired token count: {await TokenStore.get_token_count()}", 2, "[Token]")
            client_list = list(WS.clients)
            if client_list:
                if len(client_list) <= new_allocations: recipients = client_list
                else: recipients = secrets.SystemRandom().sample(client_list, new_allocations)
                for i, client in enumerate(recipients):
                    if i < len(new_tokens):
                        token = new_tokens[i]
                        client_dict = WS.clients.get(client)
                        if client_dict == None: continue
                        queue = client_dict.get("q")
                        try:
                            queue.put_nowait({"mode": "json", "packet": { "type": "rec_slot_grant", "token": token, "expires_at": expires_at}})
                            log(f"Granting new token {token} to client #{client}", 2, "[Token]")
                            await TokenStore.grant_token(token)
                        except asyncio.QueueFull: pass
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        Chant.load_references()
        BIP39WordList.load_words()
        Mix.load_mix_id()
        log("Server started. Waiting for connections...", 1, "[App]")
        app.state.background_tasks = [
            asyncio.create_task(mix_worker(), name="mix_worker"),
            asyncio.create_task(token_worker(), name="token_worker"),
        ]
        log("Lifespan started", 1, "[App]")
        yield  # The app runs here
        for task in app.state.background_tasks:
            if task.done(): continue
            task.cancel()
            try: await task
            except asyncio.CancelledError: pass
            except Exception as e:
                log(f"Error while cancelling background task: {e}", 1, "[App]")
                raise
    app = FastAPI(title="Universe", version="1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_methods=["*"], allow_headers=["*"], allow_credentials=True, # remove null for production
        allow_origins=["null", "http://localhost", "http://127.0.0.1", "http://192.168.18.56", "http://192.168.18.*"])
    app.mount("/static", StaticFiles(directory="static"), name="static")
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
            async with WS.client_lock:
                WS.client_counter += 1
                client_id = WS.client_counter
            queue = asyncio.Queue(maxsize=Const.max_queue_size)
            WS.clients[client_id] = {"ws": websocket, "q": queue}
            log(f"Client #{client_id} connected.", 2, "[WS]")
            now = time.time() # float
            await websocket.send_json({"type": "sync", "client_id": client_id, "now": now})
            await WS.broadcast("bytes", Chant.ref_audio, client_id)
            async def sender():
                while True:
                    try:
                        msg = await asyncio.wait_for(queue.get(), timeout=1.0)
                        if msg.get("mode") == "json": await websocket.send_json(msg["packet"])
                        else: await websocket.send_bytes(msg.get("packet", b""))
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
            if client_id and client_id in WS.clients: WS.clients.pop(client_id, None)
            for t in tasks:
                if not t.done(): t.cancel()
    @app.post("/upload")
    async def upload_endpoint(request: Request):
        log("Handling upload request...", 1, "[Upload]")
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > Const.max_upload_size + (512 * 1024):  # small overhead
            return { "status": f"Upload request ({Util.convert_to_mb(int(content_length))}) exceeds maximum allowed: {Util.convert_to_mb(Const.max_upload_size)}." }
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
                    if token is None or not await TokenStore.validate_token(token):
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
            return { "status": f"Upload request exceeds maximum allowed: {Util.convert_to_mb(Const.max_upload_size)}." }
        log(f"Size within limit: {Util.convert_to_mb(Const.max_upload_size)}", 2, "[Upload]")
        if not file_target.value: return { "status": "Incomplete audio upload." }
        file_bytes = file_target.value
        rec_size = len(file_bytes)
        log(f"Actual audio rec_size OK: {rec_size}", 2, "[Upload]")
        md5_hash = hashlib.md5(file_bytes).hexdigest()
        log(f"Audio hash: {md5_hash}", 2 , "[Upload]")
        uploads_dir_path = Path(Const.uploads_dir)
        existing_files = list(uploads_dir_path.glob(f"{md5_hash}.*.webm"))
        if existing_files: return { "status": "Audio file previously uploaded." }
        log("Non-duplicate audio file", 2 , "[Upload]")
        ip = IP.get_client_ip(request)
        filename = f"{md5_hash}.{token}.{ip}.webm"
        Util.atomic_write_binary(uploads_dir_path/filename, file_bytes)
        log(f"Audio file saved to: {filename}", 2, "[Upload]")
        await Mix.inc_upload_count()
        return { "status": "Upload successful." }
    @app.get("/status")
    async def status_endpoint():
        now = int(time.time())
        return {
            "current_time": now,
            "time_slot_duration_sec": Const.time_slot_duration_sec,
            "current_time_slot_index": GetCurrent.time_slot_index(now),
            "current_time_slot_remaining": GetCurrent.time_slot_remaining(now),
            "rec_slot_duration_sec": Const.rec_slot_duration_sec,
            "uploads_per_mix": Const.uploads_per_mix,
            "upload_count": Mix.upload_count,
            "token_count": await TokenStore.get_token_count(),
            "granted_token_count": await TokenStore.get_granted_token_count(),
        }
    @app.get("/flag/{country_code}")
    def get_flag(country_code):
        flags_dir_path = Path(Const.flags_dir)
        flags_path = flags_dir_path / Path(f"{country_code.lower()}.png")
        if not flags_path.exists(): flags_path = flags_dir_path / Path("xx.png")
        return FileResponse(flags_path)
    @app.get("/")
    async def root(request: Request):
        context = {
            "request": request,
            "time_slot_duration_sec": Const.time_slot_duration_sec,
            "rec_slot_duration_sec": Const.rec_slot_duration_sec,
            "uploads_per_mix": Const.uploads_per_mix,
            "max_upload_size": Const.max_upload_size,
            "last_mix_id": Mix.mix_id,
            "mix_epoch": Mix.epoch,
        }
        return templates.TemplateResponse(request, "index.html", context)
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return FileResponse(Const.favicon)
    return app

def main():
    def check_dir(path):
        path = Path(path)
        if not path.is_dir(): log(f"Fatal: directory not found: {path.name}", 0, "[App]")
    check_dir(Const.uploads_dir)
    check_dir(Const.mixes_dir)
    check_dir(Const.archive_dir)
    app = create_app()
    uvicorn.run("main:create_app", factory=True, host="0.0.0.0", port=443, ssl_keyfile="fastapi.key", ssl_certfile="fastapi.crt")

if __name__ == "__main__": main()