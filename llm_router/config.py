"""YAML configuration loading and validation."""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml

MODEL_TYPES = {"llm", "embedding"}
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


@dataclass
class ModelConfig:
    name: str                       # name exposed to clients
    type: str = "llm"               # "llm" or "embedding"
    backend_name: Optional[str] = None  # model name on the backend side (default = name)
    max_concurrent: Optional[int] = None  # per-model concurrency limit on this machine

    @property
    def upstream_name(self) -> str:
        return self.backend_name or self.name


@dataclass
class MachineConfig:
    name: str
    url: str                        # OpenAI base URL, e.g. http://192.168.1.10:8080/v1
    performance: float = 1.0        # relative score: higher = faster
    max_concurrent: int = 1         # number of simultaneous requests accepted
    api_key: Optional[str] = None
    enabled: bool = True
    supports_stream_options: bool = True  # whether stream_options.include_usage is accepted
    models: list[ModelConfig] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["models"] = [{k: v for k, v in m.items() if v is not None} for m in d["models"]]
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class ApiKey:
    name: str
    key: str


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    # Accepted keys: list of strings or {name, key} dicts. Env var LLM_ROUTER_API_KEYS (comma-separated)
    api_keys: list = field(default_factory=list)
    allow_no_auth: bool = False           # start without a key (not recommended if exposed)
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    forwarded_allow_ips: Optional[str] = None  # trusted reverse-proxy IP (X-Forwarded-For)
    default_priority: str = "medium"
    max_queue_size: int = 1000
    queue_timeout: float = 300.0          # max seconds waiting in the queue
    request_timeout: float = 600.0        # max seconds for a backend request
    connect_timeout: float = 5.0
    health_check_interval: float = 15.0   # 0 = disabled
    aging_seconds: float = 0.0            # +1 priority level every N seconds of waiting (0 = off)
    internal_streaming: bool = True       # internally stream non-streaming requests to measure prefill
    max_retries: int = 1                  # retries if a machine is unreachable
    db_path: str = "router_metrics.db"
    log_file: Optional[str] = "router_requests.jsonl"
    machines_file: Optional[str] = "machines.yaml"  # pool modified from the console (takes precedence at startup)

    def parsed_keys(self) -> list[ApiKey]:
        keys: list[ApiKey] = []
        for i, k in enumerate(self.api_keys or []):
            if isinstance(k, dict):
                keys.append(ApiKey(name=str(k.get("name") or f"key{i + 1}"), key=str(k["key"])))
            elif k:
                keys.append(ApiKey(name=f"key{i + 1}", key=str(k)))
        env = os.environ.get("LLM_ROUTER_API_KEYS", "")
        for i, k in enumerate(x.strip() for x in env.split(",") if x.strip()):
            keys.append(ApiKey(name=f"env{i + 1}", key=k))
        for k in keys:
            if len(k.key) < 16:
                raise ValueError(f"API key '{k.name}' too short (16 characters minimum)")
            if "REMPLACEZ" in k.key.upper():
                raise ValueError(f"API key '{k.name}': replace the example key "
                                 "(python -m llm_router --gen-key)")
        return keys


@dataclass
class AdminConfig:
    enabled: bool = True
    host: str = "0.0.0.0"        # ideally the machine's LAN IP, e.g. 192.168.1.5
    port: int = 8001
    # Only these ranges can access the console (no authentication)
    allowed_networks: list[str] = field(default_factory=lambda: [
        "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "::1/128", "fc00::/7", "fe80::/10",
    ])


@dataclass
class Config:
    server: ServerConfig
    machines: list[MachineConfig]
    admin: AdminConfig = field(default_factory=AdminConfig)
    path: Optional[Path] = None

    def resolve(self, p: Optional[str]) -> Optional[Path]:
        """Path relative to the configuration file's directory."""
        if not p:
            return None
        pp = Path(p)
        if pp.is_absolute() or self.path is None:
            return pp
        return self.path.parent / pp


def model_from_dict(raw: dict | str) -> ModelConfig:
    if isinstance(raw, str):
        raw = {"name": raw}
    raw = {k: v for k, v in raw.items() if v not in (None, "")}
    m = ModelConfig(**raw)
    if not NAME_RE.match(m.name or ""):
        raise ValueError(f"Invalid model name: '{m.name}'")
    if m.type not in MODEL_TYPES:
        raise ValueError(f"Model {m.name}: invalid type '{m.type}' (llm or embedding)")
    if m.max_concurrent is not None:
        m.max_concurrent = int(m.max_concurrent)
        if m.max_concurrent < 1:
            raise ValueError(f"Model {m.name}: max_concurrent must be >= 1")
    return m


def machine_from_dict(raw: dict) -> MachineConfig:
    raw = dict(raw)
    models_raw = raw.pop("models", None) or []
    raw.pop("has_api_key", None)
    known = set(MachineConfig.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown fields: {', '.join(sorted(unknown))}")
    mc = MachineConfig(**raw, models=[model_from_dict(m) for m in models_raw])
    if not NAME_RE.match(mc.name or ""):
        raise ValueError(f"Invalid machine name: '{mc.name}'")
    mc.url = (mc.url or "").strip().rstrip("/")
    if not re.match(r"^https?://[^/\s]+", mc.url):
        raise ValueError(f"Invalid URL for {mc.name}: '{mc.url}' (expected http(s)://host:port/v1)")
    mc.performance = float(mc.performance)
    mc.max_concurrent = int(mc.max_concurrent)
    if mc.performance <= 0:
        raise ValueError(f"[{mc.name}] performance must be > 0")
    if mc.max_concurrent < 1:
        raise ValueError(f"[{mc.name}] max_concurrent must be >= 1")
    names = [m.name for m in mc.models]
    if len(names) != len(set(names)):
        raise ValueError(f"[{mc.name}] model declared twice")
    mc.api_key = mc.api_key or None
    return mc


def check_model_types(machines: list[MachineConfig]) -> None:
    """A model name must have the same type across all machines."""
    types: dict[str, tuple[str, str]] = {}
    for mc in machines:
        for m in mc.models:
            t, where = types.setdefault(m.name, (m.type, mc.name))
            if t != m.type:
                raise ValueError(f"Model {m.name} is '{t}' on {where} but '{m.type}' on {mc.name}")


def parse_machines(raw_list: list) -> list[MachineConfig]:
    machines = [machine_from_dict(r) for r in raw_list or []]
    names = [m.name for m in machines]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"Duplicate machine(s): {', '.join(dup)}")
    check_model_types(machines)
    return machines


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    server = ServerConfig(**(data.get("server") or {}))
    if server.default_priority.lower() not in ("low", "medium", "high"):
        raise ValueError("server.default_priority doit être low, medium ou high")
    server.parsed_keys()  # validation
    admin = AdminConfig(**(data.get("admin") or {}))
    cfg = Config(server=server, machines=[], admin=admin, path=path)

    mf = cfg.resolve(server.machines_file)
    if mf is not None and mf.exists():
        raw = yaml.safe_load(mf.read_text(encoding="utf-8")) or {}
        cfg.machines = parse_machines(raw.get("machines") or [])
    else:
        cfg.machines = parse_machines(data.get("machines") or [])
    return cfg


def save_machines(path: Path, machines: list[MachineConfig]) -> None:
    """Atomic write of the machine pool."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    header = ("# Machine pool managed by the router admin console.\n"
              "# This file takes precedence over the 'machines' section of config.yaml.\n")
    tmp.write_text(header + yaml.safe_dump({"machines": [m.to_dict() for m in machines]},
                                           allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.replace(tmp, path)
