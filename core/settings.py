"""API 配置:从哪读、往哪存、怎么测通。

**配置只有一项**:视觉模型(读比例尺上的 "40 μm")。三条来源,优先级从高到低:

    1. `.env.local`  —— 界面里「⚙ API 设置」保存的(已在 .gitignore,不会进 git)
    2. 进程环境变量  —— 临时调试用 `set VISION_API_KEY=...`
    3. 项目根 `.env` —— 手工维护的模板(从 .env.example 复制)
    4. 内置默认

界面值压在最上面是有意的:点了「保存」就必须立刻生效,不能被环境里的旧值盖掉。
代价是"我明明 set 了环境变量怎么没用" —— 用**来源可见**化解:状态行会写明
`来源:界面保存 / 环境变量 / .env`,并且提供「清除界面配置」一键回落。

变量名 `VISION_*`。旧名 `DEEPSEEK_*` 仍会被识别(作**回退**),老配置不用改。

本模块**不 import integrations.vision**(会成环):保存后由调用方显式调
`vision.invalidate()` 清读数缓存,见 views/settings_dialog.py。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx

from .config import BASE_DIR

ENV_LOCAL = BASE_DIR / '.env.local'   # 界面写的(不进 git)
ENV_FILE = BASE_DIR / '.env'          # 手工维护的

# 新名 -> 旧名(旧名只作回退,见模块 docstring)
_FIELDS = ('VISION_API_KEY', 'VISION_BASE_URL', 'VISION_MODEL')
_LEGACY = {
    'VISION_API_KEY': 'DEEPSEEK_API_KEY',
    'VISION_BASE_URL': 'DEEPSEEK_BASE_URL',
    'VISION_MODEL': 'DEEPSEEK_VISION_MODEL',
}
_DEFAULTS = {
    # base_url / model 给的是**最常见的组合**(DeepSeek),这样只填一个 key 就能跑
    # ——"快速导入 api 就能工作"靠的就是它。用别家的话在界面里挑预设,或者直接改。
    'VISION_API_KEY': '',
    'VISION_BASE_URL': 'https://api.deepseek.com',
    'VISION_MODEL': 'deepseek-v4-flash-vision-exp',
}
_TIMEOUT_PROBE = 8.0    # GET /models 探活
_TIMEOUT_PING = 20.0    # chat ping(推理模型慢,给宽点)


@dataclass(frozen=True)
class VisionConfig:
    """一次读取拿到的配置快照。"""
    api_key: str = ''
    base_url: str = ''
    model: str = ''
    source: str = 'default'      # gui / env / dotenv / default

    @property
    def configured(self) -> bool:
        """能不能拿去发请求。

        地址和模型有默认值,所以一般是"填了 key 就能用";本地 ollama / vLLM
        通常不校验 key,那种端点(host 是 localhost/127.0.0.1)不看 key。
        """
        if not (self.base_url and self.model):
            return False
        return bool(self.api_key) or _is_local(self.base_url)

    @property
    def masked_key(self) -> str:
        k = self.api_key or ''
        if not k:
            return '(空)'
        return f'{k[:4]}…{k[-4:]}' if len(k) > 12 else '…'


_cache: Optional[VisionConfig] = None


# ---------- 读 ----------

def _parse_env_file(path: Path) -> dict:
    """读一个 KEY=VALUE 文件;不存在/损坏一律返回空 dict(绝不抛)。

    值两端可以用单/双引号包起来(写入时含空格的值就是这么存的)。
    这里自己写十几行,是为了不依赖 python-dotenv —— 它只是 nicegui 的
    传递依赖,哪天上游不带了就断。
    """
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
        return {}
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
            value = value[1:-1]
        out[key.strip()] = value
    return out


def _lookup(name: str, legacy: str, dotenv: dict) -> tuple[str, str]:
    """在「环境变量 / .env」两层里找一个变量,返回 (值, 来源)。

    层与层之间:环境变量 > .env(命令行临时 set 一把要能压过文件里的旧值)。
    同一层里:新名 > 旧名(VISION_* 是正式名,DEEPSEEK_* 只为兼容老配置)。
    都没找到返回 ('', 'default')。界面写的 .env.local 不在这里管,由 load() 最后覆盖。
    """
    for src, env in (('env', os.environ), ('dotenv', dotenv)):
        for key in (name, legacy):
            value = (env.get(key) or '').strip()
            if value:
                return value, src
    return '', 'default'


def load(refresh: bool = False) -> VisionConfig:
    """当前生效的配置。结果缓存;refresh=True 强制重读磁盘。"""
    global _cache
    if _cache is not None and not refresh:
        return _cache

    dotenv = _parse_env_file(ENV_FILE)     # 界面写的 .env.local 不混进来,单独处理
    merged = dict(_DEFAULTS)
    source = 'default'
    for name in _FIELDS:
        value, src = _lookup(name, _LEGACY[name], dotenv)
        if value:
            merged[name] = value
            if source == 'default':
                source = src       # 以第一个有值的来源为准,状态行只需要一句话

    # .env.local(界面保存)排最后、直接盖过前面所有来源
    local = _parse_env_file(ENV_LOCAL)
    for name in _FIELDS:
        if (local.get(name) or '').strip():
            merged[name] = local[name].strip()
            source = 'gui'

    _cache = VisionConfig(api_key=merged['VISION_API_KEY'],
                          base_url=merged['VISION_BASE_URL'],
                          model=merged['VISION_MODEL'],
                          source=source)
    return _cache


def get_vision() -> VisionConfig:
    """给调用方用的读入口(走缓存,O(1))。"""
    return load()


def _invalidate_cache() -> None:
    global _cache
    _cache = None


# ---------- 写 ----------

_NEEDS_QUOTE = re.compile(r'[\s#"\']')


def _quote(value: str) -> str:
    """值里有空格/#/引号就用双引号包起来,内部的双引号换成单引号。"""
    v = (value or '').replace('\r', '').replace('\n', '').strip()
    if not _NEEDS_QUOTE.search(v):
        return v
    return '"' + v.replace('"', "'") + '"'


def save_vision(api_key: str, base_url: str, model: str) -> Path:
    """把界面里填的三项写进 .env.local(先 .tmp 再 replace,原子写)。

    写完清掉内存缓存 —— 下一次 get_vision() 就是新值,不用重启服务。
    返回值是写入的文件路径,供界面提示用户。
    """
    lines = [
        '# 由「API 设置」界面写入 —— 手改这个文件也行,但如果界面上又点了一次保存会被覆盖。',
        '# 这三项不进 git(.gitignore 已排除)。',
        f'VISION_API_KEY={_quote(api_key)}',
        f'VISION_BASE_URL={_quote(base_url)}',
        f'VISION_MODEL={_quote(model)}',
        '',
    ]
    tmp = ENV_LOCAL.with_name(ENV_LOCAL.name + '.tmp')
    tmp.write_text('\n'.join(lines), encoding='utf-8')
    tmp.replace(ENV_LOCAL)
    try:
        os.chmod(ENV_LOCAL, 0o600)   # Linux/mac 收紧权限;Windows 上是 no-op
    except OSError:
        pass
    _invalidate_cache()
    return ENV_LOCAL


def clear_gui() -> None:
    """删掉界面保存的配置,回落到环境变量 / .env。文件本来就不存在也不报错。"""
    ENV_LOCAL.unlink(missing_ok=True)
    _invalidate_cache()


# ---------- URL ----------

def chat_url(base_url: str) -> str:
    """base_url -> 完整的 chat/completions 地址。

    允许用户**直接把整条 endpoint 粘进来**(很多人手上就是那一条),
    已经以 /chat/completions 结尾就不再拼一次。
    """
    b = (base_url or '').strip().rstrip('/')
    if not b:
        return ''
    if b.endswith('/chat/completions'):
        return b
    return f'{b}/chat/completions'


def models_url(base_url: str) -> str:
    """base_url -> /models 地址(探活用)。同样容忍粘进来的是 chat endpoint。"""
    b = (base_url or '').strip().rstrip('/')
    if b.endswith('/chat/completions'):
        b = b[: -len('/chat/completions')]
    if not b:
        return ''
    return f'{b}/models'


def _host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except ValueError:
        return ''


def _is_local(url: str) -> bool:
    """是不是本机跑的推理服务(ollama / vLLM / LM Studio)。那种端点不需要 key。"""
    host = _host(url)
    return host.startswith(('localhost', '127.0.0.1', '0.0.0.0', '[::1]'))


# ---------- 探活 ----------

def _headers(cfg: VisionConfig) -> dict:
    return {'Authorization': f'Bearer {cfg.api_key}'} if cfg.api_key else {}


def _describe_http_error(r: httpx.Response) -> str:
    body = (r.text or '').strip().replace('\n', ' ')[:200]
    if r.status_code == 401:
        return 'API key 无效或已过期(401)'
    if r.status_code == 403:
        return '这个 key 没有访问权限(403)'
    if r.status_code == 404:
        return '地址不对(404):确认 base_url 需不需要带 /v1'
    if r.status_code == 429:
        return '被限流了(429),稍后再试'
    if r.status_code >= 500:
        return f'服务端出错({r.status_code}):{body}'
    return f'HTTP {r.status_code}:{body}'


def _chat_ping(cfg: VisionConfig) -> tuple[bool, str]:
    """退路:有些端点(老版本 ollama、自建网关)没有 /models,直接发一次最小 chat。"""
    try:
        r = httpx.post(chat_url(cfg.base_url),
                       json={'model': cfg.model,
                             'messages': [{'role': 'user', 'content': 'ping'}],
                             'max_tokens': 1},
                       headers=_headers(cfg), timeout=_TIMEOUT_PING)
    except Exception as ex:      # noqa: BLE001
        return False, f'连不上 {chat_url(cfg.base_url)}:{type(ex).__name__}'
    if r.status_code == 200:
        return True, '连接正常(chat 接口可用)'
    if r.status_code in (404, 405):
        return False, '地址不对:确认 base_url 是否需要带 /v1'
    return False, _describe_http_error(r)


def test_connection(cfg: VisionConfig) -> tuple[bool, str]:
    """测一下配置能不能用,返回 (是否成功, 给人看的一句话)。

    先 `GET {base}/models`(OpenAI / DeepSeek / 通义 / vLLM / davinci 版 ollama
    都实现了),404/405 再退回发一次最小 chat 请求。

    **纯函数、不碰界面** —— 界面那边用 run.io_bound 调,别阻塞事件循环。
    """
    if not cfg.base_url:
        return False, '先填 base_url(例如 https://api.deepseek.com)'
    url = models_url(cfg.base_url)
    try:
        r = httpx.get(url, headers=_headers(cfg), timeout=_TIMEOUT_PROBE)
    except Exception as ex:      # noqa: BLE001
        host = _host(cfg.base_url)
        hint = '本地服务没启动?(ollama:先跑 ollama serve)' if host.startswith(
            ('localhost', '127.0.0.1')) else '检查网络 / 代理 / 地址拼写'
        return False, f'连不上 {host or url}:{type(ex).__name__} —— {hint}'

    if r.status_code in (404, 405):
        return _chat_ping(cfg)
    if r.status_code != 200:
        return False, _describe_http_error(r)

    try:
        ids = [str(m.get('id')) for m in (r.json().get('data') or []) if m.get('id')]
    except Exception:            # noqa: BLE001
        ids = []
    if cfg.model and ids and cfg.model not in ids:
        shown = '、'.join(ids[:8])
        more = f' 等 {len(ids)} 个' if len(ids) > 8 else ''
        return False, f'地址通了,但模型「{cfg.model}」不在可用列表里。可用:{shown}{more}'
    if ids:
        return True, f'连接正常 · 该端点有 {len(ids)} 个模型'
    return True, '连接正常'


def status() -> dict:
    """给状态行用的一小撮信息。"""
    cfg = load(refresh=True)
    return {
        'configured': cfg.configured,
        'source': cfg.source,
        'model': cfg.model,
        'base_url': cfg.base_url,
        'masked_key': cfg.masked_key,
    }


_SOURCE_LABEL = {
    'gui': '界面保存',
    'env': '环境变量',
    'dotenv': '.env 文件',
    'default': '默认(未配置)',
}


def source_label(source: str) -> str:
    return _SOURCE_LABEL.get(source, source)
