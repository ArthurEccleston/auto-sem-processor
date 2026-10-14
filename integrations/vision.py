"""用视觉大模型读 SEM 比例尺上的数值(如 "40 μm")。

**为什么不用本地 OCR**:本地 OCR 要装 Tesseract(Windows 上还得配 PATH),
而比例尺上的 "μm" 是特殊字符,老式 OCR 常认成 um / u / m。交给视觉模型
既不用装东西,又认得准,还能顺便读 nm 这类高倍率单位。

**接口是通用的**:任何 OpenAI 格式的 `/chat/completions` 都能用 ——
DeepSeek、通义、智谱、OpenAI、以及本地的 ollama / vLLM。配置见 core/settings.py。

**设计原则:绝不抛异常**(照搬 Agent_material 那边 deep_research.py 的思路)。
没配 key、断网、超时、模型返回看不懂的内容 —— 一律返回 None,由调用方
安静地退回手填。这个功能只是个便利,**绝不能让它成为新的失败源**。

**只裁比例尺那一小块发给模型**(约 350×60),整张 SEM 图不上传:既省 token,
也让模型只专注读那几个字。
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import cv2
import httpx

from core import settings
from core.sem import load_rgb

_TIMEOUT = 20.0
_PAD_X = 30      # 横线左右各留一点,别把两端的刻度线切掉

# 连接级能力缓存:base_url 的 host -> 这个端点认不认 `thinking` 字段。
# `thinking` 是 DeepSeek 的扩展字段,别的端点收到会直接 400 ——
# 所以默认只对 DeepSeek 官方带,被拒过就永久记住别再带了。
_THINKING_OK: dict[str, bool] = {}
_THINKING_HOSTS = ('api.deepseek.com',)


def _pad_y(h: int) -> tuple[int, int]:
    """上下各留多少(px),按**图高比例**走 —— 不要用固定值。

    比例尺数值的位置依仪器而定:FEI 夹在横线中间,ZEISS 放在**横线上方**。
    实测 2048×1536 的图横线在 y=1485、数字笔画顶端到 y=1445,固定留 30 只能
    够到 1455 —— 正好从数字中间切开,模型读出来是乱码或 null。
    """
    return max(30, int(h * 0.030)), max(24, int(h * 0.020))

_PROMPT = (
    '这是一张 SEM 图像里比例尺的局部截图。请只回答横线上标注的数值和单位。'
    '只输出一个 JSON,形如 {"value": 40, "unit": "μm"} —— '
    '不要输出任何解释文字。看不清就输出 {"value": null}。'
)

_UNIT_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(μ?m|um|nm|mm|µm)', re.I)


def _mtime(path: str) -> float:
    """图片修改时间,用作缓存键的一部分(图换了就重新读,不缓存旧的)。"""
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


def _crop_png(image_path: str, bar: dict) -> Optional[bytes]:
    """按检测到的横线位置裁出比例尺局部,编码成 PNG 字节。"""
    img = load_rgb(image_path)
    h, w = img.shape[:2]
    x1 = max(0, int(bar['x1']) - _PAD_X)
    x2 = min(w, int(bar['x2']) + _PAD_X)
    pad_up, pad_down = _pad_y(h)
    y1 = max(0, int(bar['y']) - pad_up)
    y2 = min(h, int(bar['y']) + pad_down)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    ok, buf = cv2.imencode('.png', img[y1:y2, x1:x2])
    return buf.tobytes() if ok else None


def _parse(text: str) -> Optional[dict]:
    """从模型回答里抠出 {"value": 40, "unit": "μm"}。

    模型不一定老实只给 JSON(可能裹 ```json 代码块、或前后带一句话),
    所以先剥代码块、再抓第一个 {...};实在解析不出来,就从纯文本里
    直接找 "40 μm" 这种模式兜底。
    """
    if not text:
        return None
    s = text.strip()
    s = re.sub(r'^```(?:json)?|```$', '', s, flags=re.M).strip()

    m = re.search(r'\{.*?\}', s, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            obj = None
        if isinstance(obj, dict) and obj.get('value') is not None:
            try:
                value = float(obj['value'])
            except (TypeError, ValueError):
                value = None
            if value is not None:
                unit = str(obj.get('unit') or 'μm').strip() or 'μm'
                return {'value': value, 'unit': unit}

    m2 = _UNIT_RE.search(s)          # 兜底:模型直接回了 "40 μm"
    if m2:
        return {'value': float(m2.group(1)), 'unit': m2.group(2)}
    return None


def _host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except ValueError:
        return ''


def _wants_thinking(base_url: str) -> bool:
    """这个端点要不要带 `thinking` 字段(DeepSeek 专有)。

    没记录过的端点:只有 DeepSeek 官方先带,其余一律不带 —— 少一次 400 往返。
    """
    h = _host(base_url)
    known = _THINKING_OK.get(h)
    if known is not None:
        return known
    return h.endswith(_THINKING_HOSTS)


def _build_payload(cfg, prompt: str, b64: str) -> dict:
    """请求体只放 OpenAI 标准字段(谁都认),扩展字段按端点能力加。"""
    payload = {
        'model': cfg.model,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url',
                 'image_url': {'url': f'data:image/png;base64,{b64}'}},
            ],
        }],
        'temperature': 0,          # 读数要的是确定性,不要发挥
        # ⚠ max_tokens 别给小。推理模型思考过程同样吃输出预算 ——
        # 给 64 时它思考完就没额度了,content 直接返回空串(实测踩过,HTTP 还是 200)。
        'max_tokens': 512,
    }
    if _wants_thinking(cfg.base_url):
        # 读几个字用不上推理,关掉又快又省(实测 completion_tokens 56 → 12)
        payload['thinking'] = {'type': 'disabled'}
    return payload


def _post(cfg, payload: dict) -> dict:
    """发一次请求;因 `thinking` 被拒就摘掉它重试一次,并记住这个端点不支持。

    只加标准字段 + 这一层降级,已经够覆盖主流端点;刻意**不加**
    response_format(很多第三方网关不认,而 _parse 的兜底解析本来就够稳)。
    """
    url, headers = settings.chat_url(cfg.base_url), {}
    if cfg.api_key:
        headers['Authorization'] = f'Bearer {cfg.api_key}'

    resp = httpx.post(url, json=payload, timeout=_TIMEOUT, headers=headers)
    if resp.status_code in (400, 404, 422) and 'thinking' in payload:
        body = (resp.text or '').lower()
        if any(k in body for k in ('thinking', 'unknown', 'unsupported',
                                   'unexpected', 'extra', 'invalid')):
            _THINKING_OK[_host(cfg.base_url)] = False
            payload.pop('thinking')
            resp = httpx.post(url, json=payload, timeout=_TIMEOUT, headers=headers)
    resp.raise_for_status()
    content = resp.json()['choices'][0]['message']['content']
    return _parse(content if isinstance(content, str) else str(content))


def _cfg_fingerprint(cfg) -> str:
    """配置指纹 —— 换了模型/地址/key,缓存键就变,不会读到上一套配置的结果。"""
    raw = f'{cfg.base_url}|{cfg.model}|{(cfg.api_key or "")[-6:]}'
    return hashlib.md5(raw.encode('utf-8')).hexdigest()[:8]


@lru_cache(maxsize=128)
def _read_cached(path: str, mtime: float, x1: int, x2: int, y: int,
                 cfg_fp: str) -> Optional[dict]:
    """带缓存的实际读取。缓存键含**图片修改时间**和**配置指纹**。

    ⚠ **失败绝不能进缓存**:lru_cache 会把返回值(包括 None)原样记住。网络抖动、
    限流(429)、超时只要赶上一次,这张图在**本次进程**里就再也不会真去请求了 ——
    表现是"刚才还好好的,忽然全都读不出来",重跑没用,只有重启服务才恢复
    (实测踩过)。所以异常一律往外抛,由 read_scale_value 兜住但不写缓存。
    """
    cfg = settings.get_vision()
    png = _crop_png(path, {'x1': x1, 'x2': x2, 'y': y})
    if png is None:
        return None
    b64 = base64.b64encode(png).decode('ascii')
    return _post(cfg, _build_payload(cfg, _PROMPT, b64))


def invalidate() -> None:
    """配置变了之后调用:清掉读数缓存和端点能力缓存,让新配置立刻生效。

    由界面在「保存」时显式调(见 views/settings_dialog.py)——
    core/settings.py 不能自己 import 本模块,会成环。
    """
    _read_cached.cache_clear()
    _THINKING_OK.clear()


def available() -> bool:
    """是否配好了。页面可以据此提示"配置后能自动读数值"。"""
    return settings.get_vision().configured


def read_scale_value(image_path: str, bar: dict) -> Optional[dict]:
    """读比例尺横线上的数值,返回 {'value': 40.0, 'unit': 'μm'} 或 None。

    bar 是 core.sem.detect_scale_bar_in_strip() 的返回值(用它的
    x1/x2/y 定位要裁哪一块)。任何失败都返回 None,调用方退回手填即可。
    """
    cfg = settings.get_vision()
    if not cfg.configured:
        return None
    try:
        return _read_cached(str(image_path), _mtime(str(image_path)),
                            int(bar['x1']), int(bar['x2']), int(bar['y']),
                            _cfg_fingerprint(cfg))
    except Exception as ex:    # noqa: BLE001 —— 绝不外抛(见模块 docstring)
        # 但要留一行痕迹:否则批量里几十张全静悄悄"读不出来",分不清到底是
        # 没认到比例尺,还是网络/限流/配置挂了 —— 这两种的处理方式完全不同。
        print(f'[vision] 读比例尺数值失败: {type(ex).__name__}: {ex}')
        return None
