"""SEM 工作台的落盘状态。

**为什么需要它**:导入列表和处理结果本来只是 `sem_page()` 里的一个局部 dict,
而页面每次连接/刷新都会从头执行一遍函数 —— 按一下 F5,字典就变回空的,
表现就是"辛苦拖进来的图和刚生成的结果全没了"(哪怕文件其实好好躺在磁盘上)。

所以把工作台落盘:刷新、重启服务都还在,要清由用户手动点「清空工作台」。

**路径约定**:一律存**相对项目根目录**的路径,整个文件夹搬走也不会失效。

**读失败绝不抛异常**:工作台读不出来最多是回到空台面,不能因此让整个页面打不开。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .config import BASE_DIR, DATA_DIR

WORKSPACE_PATH = DATA_DIR / 'sem_workspace.json'


def _empty() -> dict:
    """空工作台。字段要和 views/sem.py 里那个 state 对得上,少一个就是 KeyError。"""
    return {
        'path': None, 'size': None, 'mode': 'crop', 'crop': None,
        'drag_mode': None, 'drag_anchor': None, 'drag_origin': None,
        'scale_pts': [], 'scale_hi': None, 'crop_k': None, 'crop_aspect': 0.75,
        'sources': [], 'active': None, 'results': [],
    }


def _rel(p) -> str:
    """磁盘路径 -> 入库字符串(相对项目根目录;空值原样返回空串)。"""
    if not p:
        return ''
    try:
        return Path(p).resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        # 项目目录之外的图(拖进来之前就在别处):只能存绝对路径,
        # 换台机器会失效 —— 但总比直接丢掉这条记录强。
        return str(Path(p))


def _abs(s) -> Optional[Path]:
    """入库字符串 -> 磁盘路径。"""
    if not s:
        return None
    p = Path(s)
    return p if p.is_absolute() else BASE_DIR / p


def load() -> dict:
    """读工作台。文件不存在、内容损坏、磁盘上的图被删过 —— 都能安全降级。"""
    state = _empty()
    try:
        raw = json.loads(WORKSPACE_PATH.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return state
    if not isinstance(raw, dict):
        return state

    # 逐条还原,磁盘上已经没有的直接丢掉 —— 避免读出指向空气的"幽灵条目"
    for item in raw.get('sources') or []:
        p = _abs(item.get('path'))
        if p is None or not p.exists():
            continue
        thumb = _abs(item.get('thumb'))
        thumb = thumb if (thumb and thumb.exists()) else p
        state['sources'].append({
            'no': len(state['sources']) + 1,      # 重新编号,避免删除后号段有洞
            'path': p, 'thumb': thumb,
            'name': item.get('name') or p.name,
            'sel': bool(item.get('sel', True)),
        })
    for item in raw.get('results') or []:
        p = _abs(item.get('path'))
        if p is not None and p.exists():
            state['results'].append({'path': p, 'caption': item.get('caption', '')})

    # 当前加载的那张:按路径反查下标(不能直接用存的 active —— 上面过滤后索引会错位)
    cur = _abs(raw.get('path'))
    if cur is not None:
        for i, s in enumerate(state['sources']):
            if str(s['path']) == str(cur):
                state['active'] = i
                state['path'] = str(s['path'])
                break

    if raw.get('mode') in ('crop', 'scale'):
        state['mode'] = raw['mode']

    # 构图参数。坏值不能让页面打不开(本模块的既有原则):读不出来就沿用
    # _empty() 里的默认值。
    try:
        state['crop_k'] = float(raw['crop_k']) if raw.get('crop_k') else None
    except (TypeError, ValueError):
        pass
    try:
        state['crop_aspect'] = float(raw.get('crop_aspect') or 0.75)
    except (TypeError, ValueError):
        pass
    return state


def save(state: dict) -> None:
    """把工作台写回磁盘。

    先写 .tmp 再 replace:中途崩了也不会留下一个只写了一半的坏文件,
    下次启动读到的要么是旧的完整内容、要么是新的完整内容。
    """
    payload = {
        'sources': [{'no': s.get('no'), 'name': s.get('name'),
                     'path': _rel(s.get('path')), 'thumb': _rel(s.get('thumb')),
                     # 勾选状态(批量只处理勾上的那些)。老工作台没这个字段,
                     # 读的时候按 True 兜底 —— 默认全选。
                     'sel': s.get('sel', True)}
                    for s in state.get('sources') or []],
        'results': [{'path': _rel(r.get('path')), 'caption': r.get('caption', '')}
                    for r in state.get('results') or []],
        'path': _rel(state.get('path')),
        'mode': state.get('mode', 'crop'),
        # 构图参数(裁剪框宽 ÷ 新比例尺像素长、高宽比)也存下来:刷新页面后
        # 不存的话"调好框 → 刷新一下 → 回来"就丢了,批量会被直接拒绝。
        # crop 的像素坐标**不存** —— 它属于"当时加载的那张图",跨会话还原没意义。
        'crop_k': state.get('crop_k'),
        'crop_aspect': state.get('crop_aspect', 0.75),
    }
    tmp = WORKSPACE_PATH.with_name(WORKSPACE_PATH.name + '.tmp')
    try:
        WORKSPACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding='utf-8')
        tmp.replace(WORKSPACE_PATH)
    except OSError as ex:      # 存不下不是致命错误:页面照常用,只是下次要重来
        print(f'[workspace] 工作台保存失败: {ex}')
        tmp.unlink(missing_ok=True)


def clear() -> None:
    """清掉工作台记录。

    **不碰磁盘上的图** —— 那些还能从 data/ 目录里找回来;
    真正想删文件请用页面上的「清空列表」。
    """
    WORKSPACE_PATH.unlink(missing_ok=True)
