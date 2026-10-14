"""SEM 比例尺分析(交互式,固定比例裁剪框)。

工作流:
1. 拖入一张或多张原图 → 下方出现横向预览条,点哪张就加载哪张
2. 选「常用比例」→ 图上生成虚线框;拖角的把手缩放(保持比例)、拖框内移动
3. 「量比例尺」模式点两端量长度(换图会自动识别,配了 API 还能自动读数值)
4. 填数值和样式 → 生成。结果累积在下方结果区,不会覆盖前一张
5. 勾选若干张 → 「批量处理选中」,每张各自识别比例尺、套同一个构图
6. 结果区的图按顺序拼成整合图,直接下载

比例尺上的数值(如 "40 μm")交给视觉模型读,配置见页面右上角「⚙ API 设置」;
没配也能用 —— 像素长度是本地算的,数值手填即可。
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from nicegui import run, ui
from PIL import Image

from core import workspace
from core.config import SEM_GRID_DIR, SEM_OUT_DIR, SEM_SRC_DIR, ensure_dirs
from core.sem import (compose_grid, compose_image, detect_scale_bar_in_strip,
                      load_gray, load_rgb, make_thumbnail)
from integrations import vision
from .settings_dialog import open_settings_dialog, refresh_api_status

ALLOWED_EXT = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


def _safe_stem(name: str, limit: int = 40) -> str:
    """把拖入的文件名清成可安全落盘的短名(保留原名方便在文件夹里认)。"""
    stem = Path(name).stem[:limit]
    return re.sub(r'[\\/:*?"<>|\s]+', '_', stem) or 'img'


def _store_source(name: str, data: bytes) -> Path:
    """原图落盘。按内容哈希命名:同一张图重复拖入不会存成两份。"""
    ext = Path(name).suffix.lower() or '.png'
    digest = hashlib.md5(data).hexdigest()[:12]
    path = SEM_SRC_DIR / f'{digest}_{_safe_stem(name)}{ext}'
    if not path.exists():
        path.write_bytes(data)
    return path


def _thumb_for(path: Path) -> Path:
    """预览用缩略图的路径(和原图同目录,文件名带 _thumb)。"""
    return path.with_name(f'{path.stem}_thumb.png')


def _ensure_thumb(path: Path) -> Path:
    """保证预览缩略图存在;转码失败(比如损坏文件)就退回原图,不阻断导入。"""
    thumb = _thumb_for(path)
    if thumb.exists():
        return thumb
    try:
        make_thumbnail(str(path), str(thumb))
        return thumb
    except Exception:  # noqa: BLE001
        return path


def _purge_sources(sources: list) -> int:
    """删掉这些导入项对应的原图副本、预览缩略图、工作图,返回实际删除的文件数。

    只删当前列表里引用到的文件,不动 sem_src 下的其它内容。被占用删不掉
    就跳过(不影响列表清空)。
    """
    targets: set[Path] = set()
    for s in sources:
        targets.add(s['path'])
        if s.get('thumb'):
            targets.add(s['thumb'])   # 转码失败时 thumb 就是原图本身,靠 set 去重
        targets.add(_work_path(s['path']))
    removed = 0
    for p in targets:
        try:
            if p.exists():
                p.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def _int_or(v, default: int) -> int:
    """ui.number 的值可能是 None / float / 空串,统一转成 int。"""
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _float_or(v, default: float) -> float:
    """同 _int_or,但要小数(字号/线粗是按图宽百分比给的,得保留小数)。"""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _work_path(src: Path) -> Path:
    """处理区显示用的工作图(原图转码成 8 位 PNG,浏览器才认)。

    用原图的内容哈希命名 —— 同一张图永远同一个文件名,不同图绝不重名。

    ⚠ 别改回自增计数器:计数器写在页面函数里,每次浏览器连接/刷新都从 0 重来,
    于是又重新生成 _work_1.png 这种**上一轮用过的**名字。浏览器一看 URL 没变
    就直接拿缓存里的旧图,点预览就会显示出别的图(而且多开标签页会互相覆盖)。
    """
    digest = src.name.split('_', 1)[0]
    return SEM_SRC_DIR / f'_work_{digest}.png'


def _out_path(img) -> Path:
    """处理结果图的路径,用图像内容哈希命名。

    和 work/thumb 同一个道理:自增计数器写在页面函数里,刷新页面后从 1 重来,
    会把上一轮的 sem_out_001.png 直接覆盖(旧结果真的丢了),浏览器也会因为
    URL 没变而显示缓存里的旧图 —— 结果区看起来就是"乱的"。
    用内容哈希:同一张结果永远同一个文件名,不同结果绝不重名,也不会覆盖。
    """
    digest = hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest()[:12]
    return SEM_OUT_DIR / f'sem_out_{digest}.png'


# ---------- 批量处理用的纯函数 ----------
#
# 下面这几个都不碰 state 和界面,只是"算" —— 单张路径(set_crop_by_k)和批量
# 路径各写一份几何,迟早会在某个边界上算出不一样的框,所以抽出来共用。

CROP_MIN_SIDE = 40      # 裁剪框一边的最小像素


def _centered_crop(iw: int, ih: int, px, k, aspect: float,
                   fallback_frac: float = 0.7) -> tuple:
    """按「新比例尺像素长的 k 倍」算一个居中的裁剪框,返回 (x1, y1, x2, y2)。

    k 为 None(还没量过比例尺)就按图宽的 fallback_frac 起一个默认框。
    结果一定落在图幅内 —— 批量时每张图尺寸不同,框是按各自尺寸现算的,
    所以不会出现"框超出图幅"这种需要调用方再钳一次的情况。
    """
    bw = int(k * px) if (k and px) else int(iw * fallback_frac)
    bw = min(max(CROP_MIN_SIDE, bw), iw)
    bh = int(bw * aspect)
    if bh > ih:
        bh = ih
        bw = min(iw, max(CROP_MIN_SIDE, int(bh / max(aspect, 1e-6))))
    bh = max(1, bh)
    x1, y1 = (iw - bw) // 2, (ih - bh) // 2
    return x1, y1, x1 + bw, y1 + bh


def _crop_metrics(crop, px, stored_k, stored_aspect) -> tuple:
    """取「当前构图」的两个参数:(倍数 k, 高宽比 aspect)。

    k 优先用记住的值;没记住但框还活着(用户拖过框,只是拖完才填新数值、
    没触发 remember_crop)就现算一次 —— 免得批量因为「差一次拖动」直接拒绝。
    """
    k = stored_k
    aspect = stored_aspect or 0.75
    if crop:
        bw, bh = crop[2] - crop[0], crop[3] - crop[1]
        if bw > 0 and bh > 0:
            aspect = bh / bw
            if k is None and px:
                k = bw / px
    return k, aspect


# 角标「序号模板」白名单:**整串**必须长成这四种样子才递增。
# 判据卡这么死是必须的 —— 标签栏里还会写 20000×(倍率)、"1 μm"、"5V" 这类
# 参数文本,当成序号递增就是往图上印错数据,几十张一起印错还很难发现。
# 单字母只认一个字符:(SE)/(BSE) 这类模式标签必须原样不动。
_LABEL_NUM_RE = (
    re.compile(r'^\(([A-Za-z])\)$'),
    re.compile(r'^\((\d+)\)$'),
    re.compile(r'^([A-Za-z])$'),
    re.compile(r'^(\d+)$'),
)


def _alpha_ord(s: str) -> int:
    """字母序号 -> 0 基下标:'a'=0、'z'=25、'aa'=26(和 Excel 列名同一套)。"""
    n = 0
    for ch in s.lower():
        n = n * 26 + (ord(ch) - 96)
    return n - 1


def _ord_alpha(i: int, upper: bool) -> str:
    """0 基下标 -> 字母:'z' 后面接 'aa',不回头、不跳号。"""
    s, i = '', i + 1
    while i > 0:
        i, r = divmod(i - 1, 26)
        s = chr(97 + r) + s
    return s.upper() if upper else s


def _nth_label(template: str, i: int) -> str:
    """把角标模板往后推 i 个序号;不像序号的文本原样返回(空值还是空值)。

    数字模板保留原来的补零宽度,字母模板保留大小写。i=0 就返回模板本身。
    """
    t = (template or '').strip()
    if not t or i <= 0:
        return t
    m = _LABEL_NUM_RE[0].match(t)
    if m:
        v = m.group(1)
        return f'({_ord_alpha(_alpha_ord(v) + i, v.isupper())})'
    m = _LABEL_NUM_RE[1].match(t)
    if m:
        return f'({int(m.group(1)) + i:0{len(m.group(1))}d})'
    m = _LABEL_NUM_RE[2].match(t)
    if m:
        v = m.group(1)
        return _ord_alpha(_alpha_ord(v) + i, v.isupper())
    m = _LABEL_NUM_RE[3].match(t)
    if m:
        return f'{int(m.group(1)) + i:0{len(m.group(1))}d}'
    return t


# 比例尺数值的单位换算(以 μm 为基准)。高倍率图常用 nm,识别出的数值若不与
# 用户填的单位统一就直接相乘,结果就是错的。
_UNIT_TO_UM = {'μm': 1.0, 'um': 1.0, 'nm': 1e-3, 'mm': 1e3}


def _to_unit(value: float, from_u: str, to_u: str):
    """把数值从 from_u 换算到 to_u;任一端单位认不出就返回 None。"""
    def _norm(u: str) -> str:
        return (u or '').strip().lower().replace('µ', 'μ')
    f, t = _UNIT_TO_UM.get(_norm(from_u)), _UNIT_TO_UM.get(_norm(to_u))
    if not f or not t:
        return None
    return value * f / t


# 常用裁剪比例
RATIO_OPTIONS = {'free': '自由', '1:1': '1:1', '4:3': '4:3', '3:2': '3:2',
                 '16:9': '16:9', '9:16': '9:16'}
RATIO_VALUES = {'free': None, '1:1': 1.0, '4:3': 4 / 3, '3:2': 3 / 2,
                '16:9': 16 / 9, '9:16': 9 / 16}

HANDLE = 14  # 拖拽把手检测半径


@ui.page('/')
def sem_page():
    ui.page_title('AutoSEM Processor')
    # 页面外壳:一张通栏容器装所有卡片。`.page-in` 让卡片从下往上滑入(样式在 theme.py)。
    with ui.column().classes('w-full max-w-7xl mx-auto p-6 gap-6 page-in'):
        # 工作台从磁盘恢复(见 core/workspace.py):页面函数每次访问都会重新执行
        # (刷新、新开标签页),写死的局部 dict 会让"刷一下就全没了"。
        # 落盘后刷新、重启服务都还在,要清由用户手动点「清空工作台」。
        state = workspace.load()

        # ---------- 标题行:名字 + API 配置入口 ----------
        with ui.row().classes('w-full items-center gap-3'):
            ui.label('🔬 AutoSEM Processor').classes('text-2xl font-bold')
            ui.label('拖入电镜原图 → 识别比例尺 → 裁剪 → 批量出图') \
                .classes('text-sm text-slate-500')
            ui.space()
            api_status = ui.label('').classes('text-sm')
            ui.button('⚙ API 设置', icon='settings',
                      on_click=lambda: open_settings_dialog(api_status)) \
                .props('outline color=primary').classes('rounded-btn')
        refresh_api_status(api_status)

        # ---------- 顶部:拖入区 + 预览条 ----------
        with ui.card().classes('w-full'):
            ui.label('导入原图').classes('text-xl font-bold')

            with ui.row().classes('w-full justify-center'):
                # ⚠ reset() 只能在"全部文件传完"后调(on_multi_upload),绝不能放在
                # 单个文件的 on_upload 里:Quasar 的 reset() 内部会 abort() 掉所有
                # 还在进行中的上传 —— 一次拖 10 张时,第 1 张传完就 reset,其余全被
                # 取消,表现就是"无论选多少张都只能传上去几张"。
                # 给了 on_multi_upload 后 NiceGUI 会开启 batch 模式,所有文件走同一个
                # 请求,天然没有并发取消的问题。
                ui.upload(label='把图片拖到这里(可一次拖多张),或点击选择',
                          multiple=True, auto_upload=True,
                          on_upload=lambda e: handle_upload(e),
                          on_multi_upload=lambda e: e.sender.reset()) \
                    .props(f'accept="{",".join(ALLOWED_EXT)}"') \
                    .classes('sem-upload')

            with ui.row().classes('w-full items-center gap-2 mt-1'):
                ui.label('已导入').classes('text-sm font-semibold text-gray-700')
                src_count = ui.label('0 张').classes('text-sm text-slate-400')
                sel_count = ui.label('已选 0 张').classes('text-sm text-primary')
                cur_label = ui.label('').classes('text-sm text-primary font-semibold')
                ui.space()
                # 批量只处理勾选的那些;这两颗是快捷开关,免得几十张里逐张点
                ui.button('全选', icon='check_box',
                          on_click=lambda: select_all(True)) \
                    .props('dense flat color=grey').classes('rounded-btn')
                ui.button('清空选择', icon='check_box_outline_blank',
                          on_click=lambda: select_all(False)) \
                    .props('dense flat color=grey').classes('rounded-btn')
                ui.button('清空工作台', icon='clear_all',
                          on_click=lambda: clear_workspace()) \
                    .props('dense flat color=grey').classes('rounded-btn')
                ui.button('清空列表', icon='delete_sweep',
                          on_click=lambda: clear_sources()) \
                    .props('dense flat color=grey').classes('rounded-btn')

            # 预览条:一行放不下就自动换到下一行(Quasar 的 .row 自带 flex-wrap,
            # 不用再加滚动条那套)
            sources_row = ui.row().classes('w-full gap-3 items-start')

        # ---------- 主工作区 ----------
        with ui.row().classes('w-full items-start gap-6'):
            # 左:图片操作区(占据主要空间)
            with ui.column().classes('flex-1 min-w-0 gap-4'):
                with ui.card().classes('w-full'):
                    with ui.row().classes('w-full items-center gap-4'):
                        # 两个**独立按钮**,互斥点亮:当前模式实心,另一个描边压暗。
                        # 谁被点亮只由 state['mode'] 决定,统一在 paint_mode_buttons()
                        # 里上色 —— 从任何地方改模式都不会出现"两个都亮/都不亮"。
                        crop_btn = ui.button('✂ 裁剪',
                                             on_click=lambda: set_mode('crop')) \
                            .classes('rounded-btn')
                        scale_btn = ui.button('📏 量比例尺',
                                              on_click=lambda: set_mode('scale')) \
                            .classes('rounded-btn')
                        ui.space()
                        ui.label('常用比例:').classes('text-sm text-slate-500')
                        ratio_bar = ui.radio(RATIO_OPTIONS, value='4:3').props('inline')

                    hint = ui.label(
                        '「裁剪」:拖框的角/边缩放(保持比例),拖框内移动位置;'
                        '选「自由」可拖任意框。') \
                        .classes('text-xs text-slate-400 mt-1')

                    ui.separator()
                    img_view = ui.interactive_image(
                        events=['click', 'mousemove', 'mousedown', 'mouseup']) \
                        .classes('w-full rounded bg-slate-50') \
                        .on_mouse(lambda e: handle_mouse(e))

            # 右:参数面板(分组)
            with ui.card().classes('w-96 shrink-0 gap-3'):
                ui.label('✂ 裁剪框').classes('font-semibold text-gray-800')
                with ui.row().classes('w-full gap-2 items-center'):
                    crop_w_input = ui.number('框宽 (px)', value=0, min=0, step=10,
                                             format='%d').classes('flex-1')
                    crop_h_input = ui.number('框高 (px)', value=0, min=0, step=10,
                                             format='%d').classes('flex-1')
                crop_size_hint = ui.label('填数字可直接设定大小;固定比例时改一边另一边自动跟着算。')\
                    .classes('text-xs text-slate-400')

                ui.separator()
                ui.label('📏 比例尺测量').classes('font-semibold text-gray-800')
                len_input = ui.input('原比例尺像素长度 (px)', placeholder='在图上量,或直接填') \
                    .classes('w-full')
                orig_val = ui.input('原比例尺数值', placeholder='如 3').classes('w-full')
                new_val = ui.input('新比例尺数值', placeholder='如 2').classes('w-full')
                unit_input = ui.input('单位', value='μm').classes('w-full')

                ui.separator()
                ui.label('🎨 比例尺样式').classes('font-semibold text-gray-800')
                pos_sel = ui.select({'bottom-left': '左下角', 'bottom-right': '右下角'},
                                    label='位置', value='bottom-left').classes('w-full')
                color_sel = ui.select({'white': '白色', 'black': '黑色'},
                                      label='颜色', value='white').classes('w-full')
                with ui.row().classes('w-full gap-2 items-center'):
                    thickness_input = ui.number('线粗 (%图宽)', value=0.4, min=0.05,
                                                max=3.0, step=0.05, format='%.2f') \
                        .classes('flex-1')
                    bar_font_input = ui.number('比例尺字号 (%图宽)', value=3.0, min=0.5,
                                               max=15.0, step=0.25, format='%.2f') \
                        .classes('flex-1')

                ui.separator()
                ui.label('🏷️ 角落标签').classes('font-semibold text-gray-800')
                label_tl = ui.input('左上角标签', placeholder='如 (a)').classes('w-full')
                label_tr = ui.input('右上角标签', placeholder='如 20000×').classes('w-full')
                label_font_input = ui.number('标签字号 (%图宽)', value=0, min=0, max=15.0,
                                             step=0.25, format='%.2f').classes('w-full')

                ui.separator()
                generate_btn = ui.button('生成图片', icon='auto_fix_high',
                                         color='primary') \
                    .classes('w-full rounded-btn')
                # 批量:把上面这一整列参数套到导入列表里的每一张上。outline 是次要
                # 动作,不和实心的「生成图片」抢。batch_hint 兼作进度行。
                batch_btn = ui.button('批量处理选中', icon='auto_awesome_motion') \
                    .props('outline color=primary').classes('w-full rounded-btn')
                batch_hint = ui.label(
                    '只处理缩略图上勾选的图;每张各自识别比例尺,认不出的跳过并在'
                    '结束时列出。裁剪沿用当前记住的倍数,角标按输入框里的模板顺序递增。') \
                    .classes('text-xs text-slate-400')

        # ---------- 结果区:处理完的图累积在这里(不覆盖前一张) ----------
        with ui.card().classes('w-full'):
            with ui.row().classes('w-full items-center gap-2'):
                ui.label('处理结果').classes('text-lg font-bold')
                result_count = ui.label('0 张').classes('text-sm text-slate-400')
                ui.space()
                ui.button('清空显示', icon='delete_sweep',
                          on_click=lambda: clear_results()) \
                    .props('dense flat color=grey').classes('rounded-btn')
            results_row = ui.row().classes('w-full flex-wrap gap-4 items-start mt-1')

        # ---------- 整合图:结果区的图拼成一张,直接下载 ----------
        grid_state = {'path': None}
        with ui.card().classes('w-full'):
            ui.label('SEM 整合图').classes('text-lg font-bold')
            ui.label('把「处理结果」区的图按顺序拼成一张网格图'
                     '(摆放顺序 = 卡片当前的先后顺序,可拖动卡片调整)。') \
                .classes('text-xs text-slate-400')
            with ui.row().classes('w-full items-center gap-3 mt-2 flex-wrap'):
                cols_input = ui.number('每行几张', value=3, min=1, max=12, step=1,
                                       format='%d') \
                    .classes('w-28')
                rows_input = ui.number('行数', value=0, min=0, max=30, step=1,
                                       format='%d') \
                    .classes('w-28')
                ui.button('组合成整合图', icon='grid_view', color='primary',
                          on_click=lambda: make_grid()).classes('rounded-btn')
                download_btn = ui.button('下载图片', icon='download',
                                         on_click=lambda: download_grid()) \
                    .classes('rounded-btn')
                download_btn.disable()   # 还没生成过,先灰着
            grid_hint = ui.label('').classes('text-xs text-slate-500 mt-1')
            grid_box = ui.column().classes('w-full mt-2')

        # ---------- 裁剪框交互 ----------
        def _scale_hi_svg() -> str:
            """自动识别到的比例尺横线,画成绿色实线 + 两端圆点。

            单独抽出来,是因为它必须和裁剪框**拼进同一个 SVG**:
            set_content 每次是整体替换,分两次画的话,用户拖一下裁剪框
            就会把识别标注覆盖没了。
            """
            hi = state.get('scale_hi')
            if not hi:
                return ''
            x1, y, x2 = hi[0], hi[1], hi[2]
            return (f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" '
                    f'stroke="#16a34a" stroke-width="3"/>'
                    f'<circle cx="{x1}" cy="{y}" r="5" fill="none" '
                    f'stroke="#16a34a" stroke-width="2"/>'
                    f'<circle cx="{x2}" cy="{y}" r="5" fill="none" '
                    f'stroke="#16a34a" stroke-width="2"/>')

        def draw_crop_box():
            c = state['crop']
            if not c:
                img_view.set_content(_scale_hi_svg())
                sync_crop_inputs()   # 框没了,输入框也要跟着清零
                return
            x1, y1, x2, y2 = c
            handles = ''
            for hx, hy in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
                handles += (f'<rect x="{hx-4}" y="{hy-4}" width="8" height="8" '
                            f'fill="white" stroke="#3b82f6" stroke-width="2"/>')
            svg = (f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" '
                   f'fill="rgba(59,130,246,0.08)" stroke="#3b82f6" stroke-width="2" '
                   f'stroke-dasharray="6,4"/>{handles}')
            img_view.set_content(svg + _scale_hi_svg())
            sync_crop_inputs()

        # 框尺寸输入与框本身是双向的:填数字改框、拖框回填数字。
        # 回填时置位这个标志,否则"填数字→改框→回填→再触发"会自己绕圈。
        _syncing = {'flag': False}

        def sync_crop_inputs():
            """把当前框的像素尺寸回填到输入框。"""
            c = state['crop']
            _syncing['flag'] = True
            try:
                crop_w_input.value = (c[2] - c[0]) if c else 0
                crop_h_input.value = (c[3] - c[1]) if c else 0
            finally:
                _syncing['flag'] = False

        def apply_crop_size(changed: str):
            """用户填了框宽/框高:据此重设裁剪框(位置保持原中心)。

            固定比例时改一边、另一边按比例自动算;「自由」模式下两边各管各的。
            """
            if _syncing['flag'] or not state['size'] or not state['crop']:
                return
            iw, ih = state['size']
            c = state['crop']
            w0, h0 = c[2] - c[0], c[3] - c[1]
            ratio = RATIO_VALUES[ratio_bar.value]
            w = _int_or(crop_w_input.value, w0)
            h = _int_or(crop_h_input.value, h0)
            if ratio is not None:
                if changed == 'w':
                    h = int(w / ratio)
                else:
                    w = int(h * ratio)
            w = max(40, min(w, iw))
            h = max(40, min(h, ih))
            cx, cy = (c[0] + c[2]) // 2, (c[1] + c[3]) // 2
            x1 = max(0, min(cx - w // 2, iw - w))
            y1 = max(0, min(cy - h // 2, ih - h))
            state['crop'] = (x1, y1, x1 + w, y1 + h)
            state['crop_aspect'] = h / w
            draw_crop_box()
            remember_crop()

        def current_new_px():
            """按当前输入算出"新比例尺"的像素长度;输入不全时返回 None。

            这就是生成时真正画出来的那条横线长度,裁剪框的大小以它为基准。
            """
            try:
                measured = float(len_input.value)
                orig_v = float(orig_val.value)
                new_v = float(new_val.value)
            except (TypeError, ValueError):
                return None
            if measured <= 0 or orig_v <= 0 or new_v <= 0:
                return None
            return measured * new_v / orig_v

        def remember_crop():
            """把当前框记成"比例尺长度的倍数"和"高宽比",供换图/改数值时复用。"""
            c = state['crop']
            if not c:
                return
            bw, bh = c[2] - c[0], c[3] - c[1]
            if bw > 0 and bh > 0:
                state['crop_aspect'] = bh / bw
                px = current_new_px()
                if px:
                    state['crop_k'] = bw / px

        def set_crop_by_k(keep_center=True):
            """按记住的倍数重建裁剪框并居中放好。

            没记过倍数(还没量比例尺)就先按图宽的 70% 起步;
            记住过就一直沿用,直到用户下次拖动 —— 这就是"一次设好、后面每张都一致"。
            几何算法在 _centered_crop 里(批量处理走的是同一个)。
            """
            if not state['size']:
                return
            iw, ih = state['size']
            state['crop'] = _centered_crop(iw, ih, current_new_px(),
                                           state['crop_k'], state['crop_aspect'])
            draw_crop_box()

        def create_ratio_box():
            """选比例/换图后重建裁剪框。

            固定比例:更新高宽比,大小仍按记住的倍数走(不重置成默认大小)。
            「自由」:保持现有框不动,让用户随便拖。
            """
            ratio = RATIO_VALUES[ratio_bar.value]
            if ratio is not None:
                state['crop_aspect'] = 1.0 / ratio
            if ratio is None and state['crop'] is None:
                img_view.set_content('')
                return
            set_crop_by_k()

        def on_scale_input_change():
            """比例尺长度一变,裁剪框按同比例缩放 —— 生成图里比例尺占比恒定。"""
            if state['crop'] and state['crop_k']:
                set_crop_by_k()

        def hit_test(x, y):
            c = state['crop']
            if not c:
                return None
            x1, y1, x2, y2 = c
            if abs(x - x1) < HANDLE and abs(y - y1) < HANDLE:
                return 'tl'
            if abs(x - x2) < HANDLE and abs(y - y1) < HANDLE:
                return 'tr'
            if abs(x - x1) < HANDLE and abs(y - y2) < HANDLE:
                return 'bl'
            if abs(x - x2) < HANDLE and abs(y - y2) < HANDLE:
                return 'br'
            if abs(x - x1) < HANDLE and y1 + 12 < y < y2 - 12:
                return 'l'
            if abs(x - x2) < HANDLE and y1 + 12 < y < y2 - 12:
                return 'r'
            if abs(y - y1) < HANDLE and x1 + 12 < x < x2 - 12:
                return 't'
            if abs(y - y2) < HANDLE and x1 + 12 < x < x2 - 12:
                return 'b'
            if x1 < x < x2 and y1 < y < y2:
                return 'move'
            return None

        def _grow(ax, ay, x, y, ratio, iw, ih):
            """从锚点(ax,ay)按比例向目标点(x,y)生长出一个保持比例的框。"""
            dx, dy = x - ax, y - ay
            if ratio is None:
                return (max(0, min(ax, x)), max(0, min(ay, y)),
                        min(iw, max(ax, x)), min(ih, max(ay, y)))
            if abs(dx) >= abs(dy) * ratio:
                w = abs(dx)
                h = w / ratio
            else:
                h = abs(dy)
                w = h * ratio
            if w < 20 or h < 20:
                return state['crop']
            x1 = ax if dx >= 0 else ax - w
            x2 = ax + w if dx >= 0 else ax
            y1 = ay if dy >= 0 else ay - h
            y2 = ay + h if dy >= 0 else ay
            return (int(max(0, x1)), int(max(0, y1)),
                    int(min(iw, x2)), int(min(ih, y2)))

        def apply_drag(x, y):
            mode = state['drag_mode']
            if mode is None or not state['drag_origin']:
                return
            ox1, oy1, ox2, oy2 = state['drag_origin']
            ratio = RATIO_VALUES[ratio_bar.value]
            iw, ih = state['size']
            if mode == 'move':
                dx, dy = x - state['drag_anchor'][0], y - state['drag_anchor'][1]
                w, h = ox2 - ox1, oy2 - oy1
                nx1 = max(0, min(ox1 + dx, iw - w))
                ny1 = max(0, min(oy1 + dy, ih - h))
                state['crop'] = (nx1, ny1, nx1 + w, ny1 + h)
            elif mode in ('tl', 'tr', 'bl', 'br'):
                anchor = {'tl': (ox2, oy2), 'tr': (ox1, oy2),
                          'bl': (ox2, oy1), 'br': (ox1, oy1)}[mode]
                state['crop'] = _grow(anchor[0], anchor[1], x, y, ratio, iw, ih)
            elif mode == 'r':
                x2 = min(max(x, ox1 + 20), iw)
                h = int((x2 - ox1) / ratio) if ratio else (oy2 - oy1)
                state['crop'] = (ox1, oy1, x2, min(oy1 + h, ih))
            elif mode == 'l':
                x1 = min(max(x, 0), ox2 - 20)
                h = int((ox2 - x1) / ratio) if ratio else (oy2 - oy1)
                state['crop'] = (x1, oy1, ox2, min(oy1 + h, ih))
            elif mode == 'b':
                y2 = min(max(y, oy1 + 20), ih)
                w = int((y2 - oy1) * ratio) if ratio else (ox2 - ox1)
                state['crop'] = (ox1, oy1, min(ox1 + w, iw), y2)
            elif mode == 't':
                y1 = min(max(y, 0), oy2 - 20)
                w = int((oy2 - y1) * ratio) if ratio else (ox2 - ox1)
                state['crop'] = (ox1, y1, min(ox1 + w, iw), oy2)
            draw_crop_box()

        def handle_crop(e, x, y):
            if e.type == 'mousedown' and e.button == 0:
                state['drag_mode'] = hit_test(x, y)
                state['drag_anchor'] = (x, y)
                state['drag_origin'] = state['crop']
                # 没有框且是自由模式:从当前点拉新框
                if state['drag_mode'] is None and state['crop'] is None \
                        and RATIO_VALUES[ratio_bar.value] is None:
                    state['crop'] = (x, y, x, y)
                    state['drag_origin'] = (x, y, x, y)
                    state['drag_mode'] = 'br'
                draw_crop_box()
            elif e.type == 'mousemove' and state.get('drag_mode'):
                apply_drag(x, y)
            elif e.type == 'mouseup' and state.get('drag_mode'):
                state['drag_mode'] = None
                c = state['crop']
                if c and (c[2] - c[0]) < 10 or (c and (c[3] - c[1]) < 10):
                    state['crop'] = None
                    img_view.set_content('')
                else:
                    remember_crop()   # 拖完就记住这次的大小,之后每张图都沿用

        # ---------- 比例尺交互 ----------
        def _draw_preview_scale(x1, y1, x2):
            svg = (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y1}" '
                   f'stroke="#3b82f6" stroke-width="3" stroke-dasharray="6,4"/>'
                   f'<circle cx="{x1}" cy="{y1}" r="6" fill="none" stroke="#3b82f6" stroke-width="2"/>')
            img_view.set_content(svg)

        def _draw_final_scale(x1, y1, x2, y2):
            svg = (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                   f'stroke="red" stroke-width="3"/>'
                   f'<circle cx="{x1}" cy="{y1}" r="6" fill="none" stroke="red" stroke-width="2"/>'
                   f'<circle cx="{x2}" cy="{y2}" r="6" fill="none" stroke="red" stroke-width="2"/>')
            img_view.set_content(svg)

        def handle_scale(e, x, y):
            pts = state['scale_pts']
            if e.type == 'mousemove':
                if len(pts) == 1:
                    _draw_preview_scale(pts[0][0], pts[0][1], x)
                return
            if e.type != 'click':
                return
            if len(pts) == 2:
                pts.clear()
            if len(pts) == 0:
                pts.append((x, y))
                len_input.value = ''
                _draw_preview_scale(x, y, x)
                return
            x1, y1 = pts[0]
            length = int(abs(x - x1))
            pts.append((x, y1))
            _draw_final_scale(x1, y1, x, y1)
            len_input.value = str(length)
            ui.notify(f'比例尺长度: {length}px (水平)', type='positive', position='top')

        def handle_mouse(e):
            x, y = int(e.image_x), int(e.image_y)
            if state['mode'] == 'crop':
                handle_crop(e, x, y)
            else:
                handle_scale(e, x, y)

        def paint_mode_buttons():
            """给两个模式按钮上色:当前模式实心点亮,另一个描边压暗。

            两个按钮天然互斥 —— 点亮谁完全由 state['mode'] 一处决定,
            不会出现两个都亮或都不亮。
            """
            for key, btn in (('crop', crop_btn), ('scale', scale_btn)):
                on = state['mode'] == key
                # NiceGUI 的 Props 是「先 remove 再 add」(见 props.py 的 __call__),
                # 所以一句就能把旧状态清干净再重新上色,只触发一次 update()。
                btn.props(add='color=primary' if on else 'color=grey-8 outline',
                          remove='color flat outline')

        def set_mode(mode: str):
            """切换「裁剪 / 量比例尺」模式(点按钮进来)。"""
            state['mode'] = mode
            paint_mode_buttons()
            if mode == 'crop':
                hint.set_text('「裁剪」模式:拖框的角/边缩放(保持比例),拖框内移动位置;'
                              '或选「自由」拖出任意框。')
                draw_crop_box()
            else:
                hint.set_text('「量比例尺」模式:点击左端 → 蓝色虚线锁定水平跟随 → 点右端量出长度。')

        # ---------- 预览条 ----------
        def selected_items() -> list:
            """当前勾选的导入项。老工作台没有 sel 字段,按"勾上"兜底。"""
            return [s for s in state['sources'] if s.get('sel', True)]

        def refresh_sel_ui() -> None:
            """同步「已选 N 张」文字和批量按钮上写的张数。"""
            n = len(selected_items())
            sel_count.set_text(f'已选 {n} 张')
            batch_btn.set_text(f'批量处理选中（{n} 张）')

        def set_selected(i: int, val: bool) -> None:
            """勾上/取消一张(供批量挑选)。

            这里**不重画整条预览条**:只勾选框本身变了,重画会把横向滚动
            位置弹回最左边,几十张图时要重新找位置。
            """
            if 0 <= i < len(state['sources']):
                state['sources'][i]['sel'] = val
                workspace.save(state)
                refresh_sel_ui()

        def select_all(val: bool) -> None:
            for s in state['sources']:
                s['sel'] = val
            render_sources()      # 勾选框长在卡片里,得重画才会跟着变

        def render_sources():
            """重画预览条:每张带编号和勾选框,当前加载的那张高亮。"""
            workspace.save(state)   # 列表有变化就顺手落盘(放在第一行:函数中途有 return)
            sources_row.clear()
            src_count.set_text(f'{len(state["sources"])} 张')
            if state['active'] is not None and 0 <= state['active'] < len(state['sources']):
                cur = state['sources'][state['active']]
                cur_label.set_text(f'· 当前处理 #{cur["no"]}  {cur["name"]}')
            else:
                cur_label.set_text('')
            refresh_sel_ui()
            with sources_row:
                if not state['sources']:
                    ui.label('还没有导入图片 —— 把图拖到上面的框里,可以一次拖多张。') \
                        .classes('text-sm text-slate-400 py-6')
                    return
                for i, s in enumerate(state['sources']):
                    picked = state['active'] == i
                    border = 'border-primary' if picked else 'border-slate-200'
                    text = 'text-primary font-semibold' if picked else 'text-slate-500'
                    with ui.column().classes('shrink-0 items-center gap-1 w-32 '
                                             'cursor-pointer') \
                            .on('click', lambda i=i: do_load(i)):
                        with ui.element('div').classes('relative'):
                            # 预览走 8 位缩略图:16 位 tif/bmp 原图在浏览器里是空白
                            ui.image(str(s.get('thumb') or s['path'])).classes(
                                f'w-32 h-24 object-cover rounded border-2 {border}')
                            # 编号角标:文件名相似或被截断时,靠它认图
                            ui.label(f'#{s["no"]}').classes(
                                'absolute top-1 left-1 px-1.5 rounded text-xs '
                                'font-bold bg-black/60 text-white leading-5')
                            # 勾选框:挑哪几张进批量。外面包一层吃点击事件 ——
                            # 不挡住的话,勾一下会顺着冒泡把这张图加载到处理区
                            # (缩略图整块都是"点一下加载"的热区)。
                            with ui.element('div').classes(
                                    'absolute top-0 right-0 p-1 bg-white/85 '
                                    'rounded-bl') \
                                    .on('click',
                                        js_handler='(e) => e.stopPropagation()'):
                                ui.checkbox(value=bool(s.get('sel', True))) \
                                    .props('dense size=sm color=primary') \
                                    .on_value_change(
                                        lambda e, i=i: set_selected(i, bool(e.value)))
                        ui.label(s['name']).classes(
                            f'text-xs w-32 truncate text-center {text}')

        # ---------- 加载 ----------
        async def auto_detect_scale(path: str) -> None:
            """换图后自动认这张图的比例尺:像素长度本地算,数值交给视觉模型读。

            ⚠ 两条路失败都**只是不改动任何东西**,绝不抛异常 —— 尤其不能清掉
            用户已经手填的值。认不出来的图,照旧手动量、手动填。
            """
            try:
                gray = await run.io_bound(load_gray, path)
                bar = await run.io_bound(detect_scale_bar_in_strip, gray)
            except Exception:   # noqa: BLE001
                bar = None
            if not bar:
                ui.notify('这张图没自动认到比例尺,请照旧手动量取',
                          type='warning', position='top')
                return

            state['scale_hi'] = (bar['x1'], bar['y'], bar['x2'], bar['y'])
            draw_crop_box()             # 先把绿线画到图上,方便一眼核对对不对
            len_input.value = str(bar['length'])
            ui.notify(f'自动识别比例尺: {bar["length"]}px',
                      type='positive', position='top')

            if not vision.available():
                return                  # 没配 API:像素长度照样有了,数值手填
            val = await run.io_bound(vision.read_scale_value, path, bar)
            if val:
                orig_val.value = f"{val['value']:g}"
                unit_input.value = val['unit']
                ui.notify(f"读出数值: {val['value']:g} {val['unit']}",
                          type='positive', position='top')

        async def do_load(idx: int) -> bool:
            """加载预览条里第 idx 张到处理区,并自动识别它的比例尺。"""
            if not (0 <= idx < len(state['sources'])):
                return False
            src = state['sources'][idx]
            path = str(src['path'])
            try:
                rgb = load_rgb(path)
                ensure_dirs()
                # 工作图按原图内容哈希命名(见 _work_path):文件名稳定且唯一,
                # 既不撞名、也不会被浏览器缓存串味;已经转码过就直接复用。
                work_path = _work_path(src['path'])
                if not work_path.exists():
                    Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)) \
                        .save(str(work_path))
                img_view.set_source(str(work_path))
                state['path'] = path
                state['size'] = (rgb.shape[1], rgb.shape[0])
                state['active'] = idx
                state['scale_pts'] = []
                state['crop'] = None
                state['scale_hi'] = None   # 换图了,上一张的识别标注先撤掉
                # 单位/样式/角落标签/新比例尺数值一律保留,换图不用重填;
                # 只有「原比例尺像素长度」和「原数值」会被下面的自动识别覆盖
                # —— 每张图的比例尺本来就不一样,这两个值必须跟着图走。
                render_sources()  # 刷新高亮
                create_ratio_box()  # 按当前比例生成默认框
            except Exception as ex:  # noqa: BLE001
                ui.notify(f'加载失败: {ex}', type='negative')
                return False
            await auto_detect_scale(path)
            return True

        def clear_sources():
            """清空导入列表,并把这次导入的原图副本 / 缩略图 / 工作图从磁盘删掉。"""
            n = len(state['sources'])
            removed = _purge_sources(state['sources'])
            state['sources'].clear()
            state['active'] = None
            state['path'] = None
            state['size'] = None
            state['crop'] = None
            img_view.set_source('')
            render_sources()
            ui.notify(f'已清空 {n} 张导入的图,并删除 {removed} 个缓存文件',
                      type='positive')

        def clear_workspace():
            """清空工作台(导入列表 + 结果区),但**磁盘上的图一张都不删**。

            这是它和「清空列表」的分工:前者只是把台面收拾干净,图还躺在
            data/ 目录里,想接着用重新拖进来即可;后者会连缓存文件一起删掉。
            """
            n_src, n_res = len(state['sources']), len(state['results'])
            state['sources'] = []
            state['results'] = []
            state['active'] = None
            state['path'] = None
            state['size'] = None
            state['crop'] = None
            state['scale_pts'] = []
            state['scale_hi'] = None
            img_view.set_source('')
            render_sources()    # 这两个 render 里会把空工作台落盘
            render_results()
            ui.notify(f'工作台已清空(导入 {n_src} 张 / 结果 {n_res} 张);'
                      f'图都没删,还在 data/ 目录里', type='positive')

        async def handle_upload(e):
            """单个文件传到服务端后的处理(要 await do_load 做自动识别,故为 async)。

            这里**不要**调 e.sender.reset() —— 会把同批次还没传完的文件全 abort 掉
            (原因见上面 upload 组件的注释),收尾统一交给 on_multi_upload。
            另外一次拖很多张时这里逐张触发,成功提示会刷屏,所以不弹提示:
            预览条本身就是反馈,只有格式不对才提示。
            """
            ext = Path(e.name).suffix.lower() or '.png'
            if ext not in ALLOWED_EXT:
                ui.notify(f'不支持的文件类型: {ext}', type='warning')
                return
            ensure_dirs()
            path = _store_source(e.name, e.content.read())
            # 这张已经在列表里 → 只切过去,不重复添加
            for i, s in enumerate(state['sources']):
                if s['path'] == path:
                    await do_load(i)
                    return
            state['sources'].append(
                {'no': len(state['sources']) + 1, 'path': path,
                 'thumb': _ensure_thumb(path), 'name': e.name})
            await do_load(len(state['sources']) - 1)

        # ---------- 结果区 ----------
        def _result_card(path: Path, caption: str):
            # rid-xxx 这个类名是给拖拽用的:前端脚本从 class 里认出这张图是谁,
            # 拖完把 from/to 两个 id 回传给 move_result。
            with ui.card().classes(f'w-72 gap-2 p-3 result-card rid-{path.stem}'):
                ui.image(str(path)).classes('w-full rounded')
                ui.label(caption).classes('text-xs text-slate-600')
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label(path.name).classes('text-xs font-mono text-slate-400')
                    with ui.row().classes('gap-0'):
                        # filename= 不能省:不传时浏览器对 PNG 往往是"新标签页打开",
                        # 而不是弹下载框
                        ui.button(icon='download',
                                  on_click=lambda p=path: ui.download(
                                      str(p), filename=p.name)) \
                            .props('dense flat').classes('rounded-btn')
                        ui.button(icon='delete', color='negative',
                                  on_click=lambda p=path: confirm_delete_result(p)) \
                            .props('dense flat').classes('rounded-btn')

        def move_result(rid_from: str, rid_to: str) -> None:
            """把 from 那张挪到 to 的位置(整合图就按这个顺序拼)。"""
            items = state['results']
            fi = next((i for i, r in enumerate(items) if r['path'].stem == rid_from), None)
            ti = next((i for i, r in enumerate(items) if r['path'].stem == rid_to), None)
            if fi is None or ti is None or fi == ti:
                return
            items.insert(ti, items.pop(fi))
            render_results()
            ui.notify(f'已挪到第 {ti + 1} 位', position='top')

        def render_results():
            """按 state['results'] 重画结果区(删掉某张后也走这里)。"""
            workspace.save(state)   # 同 render_sources:渲染前先把状态落盘
            results_row.clear()
            result_count.set_text(f'{len(state["results"])} 张')
            with results_row:
                if not state['results']:
                    ui.label('还没有处理结果 —— 上面量好比例尺、填好参数,点「生成图片」。') \
                        .classes('text-sm text-slate-400 py-4')
                for item in state['results']:
                    _result_card(item['path'], item['caption'])
            update_grid_hint()   # 结果变了,整合图的"几行几列"提示跟着更新

        def add_result_card(path: Path, caption: str) -> bool:
            """把一张处理结果追加到结果区(不覆盖之前的)。内容重复就不重复添加。"""
            if any(r['path'] == path for r in state['results']):
                return False
            state['results'].append({'path': path, 'caption': caption})
            render_results()
            return True

        def confirm_delete_result(path: Path):
            with ui.dialog() as dlg, ui.card():
                ui.label('删除这张处理结果?').classes('text-lg font-bold')
                ui.label(path.name).classes('text-xs font-mono text-slate-500 mt-1')
                ui.label('图片文件会一并删掉,不能恢复(需要的话重新生成一张即可)。') \
                    .classes('text-sm text-slate-600 mt-1')
                with ui.row().classes('w-full justify-end gap-2 mt-3'):
                    ui.button('取消', on_click=dlg.close).classes('rounded-btn')
                    ui.button('删除', color='negative', icon='delete',
                              on_click=lambda: (delete_result(path), dlg.close())) \
                        .classes('rounded-btn')
            dlg.open()

        def delete_result(path: Path):
            """删掉一张结果:卡片 + state + 磁盘文件。"""
            state['results'] = [r for r in state['results'] if r['path'] != path]
            try:
                if path.exists():
                    path.unlink()
            except OSError as ex:  # noqa: BLE001
                ui.notify(f'文件没删掉(可能被占用): {ex}', type='warning')
                render_results()
                return
            render_results()
            ui.notify('已删除这张结果', type='positive')

        def clear_results():
            """只清空界面显示:文件留在磁盘上。"""
            state['results'].clear()
            render_results()
            ui.notify('已清空显示(文件仍保留在 data/sem_out)', type='positive')

        # ---------- 整合图 ----------
        def _grid_dims() -> tuple:
            """读行列输入框,返回 (列数, 行数);行数给 None 表示按图片数量自动。"""
            cols = max(1, min(12, _int_or(cols_input.value, 3)))
            rows_in = max(0, min(30, _int_or(rows_input.value, 0)))
            return cols, (rows_in or None)

        def update_grid_hint():
            """在按钮下面提示:当前几张图、会排成几行几列。"""
            n = len(state['results'])
            cols, rows = _grid_dims()
            if n == 0:
                grid_hint.set_text('结果区还没有图 —— 先在上面处理几张。')
                return
            if rows:
                cap = cols * rows
                if n > cap:
                    extra = f';放不下,只取前 {cap} 张'
                elif n < cap:
                    extra = f';会空 {cap - n} 格'
                else:
                    extra = ''
                grid_hint.set_text(f'当前 {n} 张 → 排成 {cols} 列 × {rows} 行{extra}')
            else:
                auto = (n + cols - 1) // cols
                grid_hint.set_text(f'当前 {n} 张 → 排成 {cols} 列 × {auto} 行(行数自动)')

        def make_grid():
            """把结果区的图按指定的行×列拼成整合图,落盘后即可下载。"""
            if not state['results']:
                ui.notify('结果区还没有图,先处理几张再组合', type='warning')
                return
            try:
                cols, rows = _grid_dims()
                paths = [r['path'] for r in state['results']]
                if rows and len(paths) > cols * rows:
                    ui.notify(f'{cols} 列 × {rows} 行只放得下 {cols * rows} 张,'
                              f'只取前 {cols * rows} 张', type='warning')
                    paths = paths[:cols * rows]
                grid = compose_grid(paths, cols=cols, rows=rows)
                ensure_dirs()
                # 文件名全 ASCII:下载头里带中文容易乱码;每次都新建一个,
                # 同一批图换个行列数再拼一版也不会互相覆盖。
                stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                out_path = SEM_GRID_DIR / f'grid_{stamp}.png'
                Image.fromarray(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)).save(str(out_path))

                grid_state['path'] = out_path
                real_rows = rows or ((len(paths) + cols - 1) // cols)
                grid_box.clear()
                with grid_box:
                    ui.image(str(out_path)).classes('w-full max-w-3xl rounded mt-1')
                    ui.label(f'{len(paths)} 张 · {cols} 列 × {real_rows} 行 · {out_path.name}') \
                        .classes('text-xs font-mono text-slate-500')
                download_btn.enable()
                ui.notify('整合图已生成,点「下载图片」保存到本机', type='positive')
            except Exception as ex:  # noqa: BLE001
                ui.notify(f'生成整合图失败: {ex}', type='negative')

        def download_grid():
            """下载最近一次生成的整合图(没生成过时按钮是灰的,走不到这儿)。"""
            p = grid_state['path']
            if not p or not Path(p).exists():
                ui.notify('整合图文件不在了,请重新点一次「组合成整合图」',
                          type='warning')
                download_btn.disable()
                return
            ui.download(str(p), filename=Path(p).name)

        # ---------- 生成 ----------
        def generate():
            if not state['path']:
                ui.notify('请先加载图片', type='warning')
                return
            try:
                measured = float(len_input.value)
                orig_v = float(orig_val.value)
                new_v = float(new_val.value)
                unit_s = unit_input.value.strip() or 'μm'
                if measured <= 0 or orig_v <= 0 or new_v <= 0:
                    ui.notify('请先量好原比例尺长度,并填原/新数值', type='warning')
                    return

                img = load_rgb(state['path'])
                if state['crop']:
                    x1, y1, x2, y2 = state['crop']
                    x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])
                    cropped = img[y1:y2, x1:x2]
                    if cropped.shape[0] < 10 or cropped.shape[1] < 10:
                        ui.notify('裁剪区域太小', type='warning')
                        return
                else:
                    cropped = img

                new_px = int(measured * new_v / orig_v)
                result = compose_image(
                    cropped, new_px, new_v, unit_s,
                    position=pos_sel.value, color=color_sel.value,
                    label_tl=label_tl.value.strip(), label_tr=label_tr.value.strip(),
                    thickness_pct=_float_or(thickness_input.value, 0.4),
                    bar_font_pct=_float_or(bar_font_input.value, 3.0),
                    label_font_pct=_float_or(label_font_input.value, 0.0) or None,
                )

                # 结果图按内容哈希命名(见 _out_path):不重名、不覆盖、缓存不串味
                out_path = _out_path(result)
                if not out_path.exists():
                    Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)) \
                        .save(str(out_path))
                added = add_result_card(
                    out_path,
                    f'新比例尺 {new_v:g} {unit_s}（{new_px}px）· '
                    f'{pos_sel.value} · {color_sel.value}')
                ui.notify('生成成功,已加到下方结果区' if added
                          else '这张和已有结果完全相同,没有重复添加',
                          type='positive' if added else 'info')
            except Exception as ex:  # noqa: BLE001
                ui.notify(f'生成失败: {ex}', type='negative')

        # ---------- 批量处理(全部已导入) ----------
        _batch_busy = {'flag': False}

        def _append_result(path: Path, caption: str) -> bool:
            """只往 state 里追加、不立刻重画。

            逐张走 add_result_card 会把结果区整体重画 N 次(第 N 次里已经躺着
            N 张卡片),几十张就是平方级的 DOM 重建;批完统一刷一次就够。
            """
            if any(r['path'] == path for r in state['results']):
                return False
            state['results'].append({'path': path, 'caption': caption})
            return True

        async def batch_generate() -> None:
            """给**勾选的**每一张各跑一遍:各自识别比例尺 → 套同一构图 → 生成。

            全程只读参数、只写 state['results'] —— 当前正在编辑的那张图
            (当前图/裁剪框/输入框)一律不碰,批完页面还是原样。
            """
            if _batch_busy['flag']:
                return
            items = selected_items()            # 快照:批到一半清空列表也不会炸
            if not items:
                if not state['sources']:
                    ui.notify('还没有导入图片 —— 先把图拖进来', type='warning')
                else:
                    ui.notify('一张都没勾选 —— 在缩略图右上角勾出要处理的图,'
                              '或点「全选」', type='warning')
                return
            new_v = _float_or(new_val.value, 0)
            if new_v <= 0:
                ui.notify('先填「新比例尺数值」—— 每张图的像素长度都按它折算',
                          type='warning')
                return
            k, aspect = _crop_metrics(state['crop'], current_new_px(),
                                      state['crop_k'], state['crop_aspect'])
            if not k:
                ui.notify('先在任意一张图上量一次比例尺、把裁剪框调好(倍数会被记住),'
                          '批量才能给每张图套同一个构图', type='warning')
                return
            if not vision.available():
                ui.notify('批量要逐张自动读比例尺数值,请先在右上角「⚙ API 设置」里'
                          '配好模型;没配的话只能逐张手动处理', type='warning')
                return

            unit_s = (unit_input.value or '').strip() or 'μm'
            tpl_tl = (label_tl.value or '').strip()
            tpl_tr = (label_tr.value or '').strip()
            style = dict(position=pos_sel.value, color=color_sel.value,
                         thickness_pct=_float_or(thickness_input.value, 0.4),
                         bar_font_pct=_float_or(bar_font_input.value, 3.0),
                         label_font_pct=_float_or(label_font_input.value, 0.0) or None)

            ensure_dirs()
            # ⚠ 置位之前**一个 await 都不能有**:连点两下会各排一个后台任务,
            #   谁先跑到这里谁赢,中间不能让出事件循环。按钮 disable 的 DOM 生效
            #   要等一个 websocket 往返,挡不住快速双击,真正的保险是这个标志位。
            _batch_busy['flag'] = True
            batch_btn.disable()
            generate_btn.disable()
            n, ok, dup, skipped = len(items), 0, 0, []
            batch_hint.set_text(f'处理中 0/{n} …')
            try:
                for i, s in enumerate(items):
                    name, path = s['name'], str(s['path'])
                    batch_hint.set_text(f'处理中 {i + 1}/{n} · #{s["no"]} · 识别中…')
                    try:
                        gray = await run.io_bound(load_gray, path)
                        bar = await run.io_bound(detect_scale_bar_in_strip, gray)
                        if not bar:
                            skipped.append((name, '没认到比例尺'))
                            continue
                        batch_hint.set_text(f'处理中 {i + 1}/{n} · #{s["no"]} · 读数值…')
                        val = await run.io_bound(vision.read_scale_value, path, bar)
                        if not val:
                            skipped.append((name, '数值没读出来'))
                            continue
                        # 识别出的数值先换算到用户填的单位 —— 高倍率图上的 nm 不
                        # 换算就直接乘,算出来的像素长度是错的
                        orig_v = _to_unit(float(val['value']),
                                          val.get('unit', unit_s), unit_s)
                        if orig_v is None:
                            skipped.append((name, '单位认不出'))
                            continue
                        if orig_v <= 0 or bar['length'] <= 0:
                            skipped.append((name, '识别出的值不合理'))
                            continue
                        new_px = int(bar['length'] * new_v / orig_v)
                        if new_px < 5:
                            skipped.append((name, f'算出来比例尺只有 {new_px}px'))
                            continue
                        rgb = await run.io_bound(load_rgb, path)
                        ih, iw = rgb.shape[:2]
                        if iw < CROP_MIN_SIDE or ih < CROP_MIN_SIDE:
                            skipped.append((name, '图太小'))
                            continue
                        x1, y1, x2, y2 = _centered_crop(iw, ih, new_px, k, aspect)
                        cropped = rgb[y1:y2, x1:x2]
                        if cropped.shape[0] < 10 or cropped.shape[1] < 10:
                            skipped.append((name, '裁剪区域太小'))
                            continue
                        if new_px > cropped.shape[1] * 0.9:
                            skipped.append((name, '比例尺比裁剪框还宽'))
                            continue
                        result = compose_image(
                            cropped, new_px, new_v, unit_s,
                            label_tl=_nth_label(tpl_tl, ok),
                            label_tr=_nth_label(tpl_tr, ok), **style)
                        out_path = _out_path(result)
                        if not out_path.exists():
                            Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)) \
                                .save(str(out_path))
                        if not _append_result(
                                out_path,
                                f'批量 #{s["no"]} {name} · 原 {orig_v:g} {unit_s} → '
                                f'新 {new_v:g} {unit_s}（{new_px}px）'):
                            dup += 1      # 内容一模一样:卡片不重复加,但图是好的
                        ok += 1
                    except Exception as ex:          # noqa: BLE001
                        skipped.append((name, f'处理失败: {ex}'))
            finally:
                _batch_busy['flag'] = False
                batch_btn.enable()
                generate_btn.enable()

            workspace.save(state)   # 先落盘:后面渲染要是出岔子,结果也丢不了
            try:
                render_results()
            except Exception as ex:              # noqa: BLE001
                print(f'[sem] 批量结果刷新失败(页面可能已经关了): {ex}')

            short = f'批量完成:成功 {ok} 张,跳过 {len(skipped)} 张'
            if dup:
                short += f',其中 {dup} 张与已有结果完全相同'
            batch_hint.set_text(short)
            if skipped:
                detail = '、'.join(f'{nm}({why})' for nm, why in skipped)
                ui.notify(f'{short}\n跳过的:{detail}', type='warning', position='top',
                          multi_line=True, close_button=True, timeout=0)
            else:
                ui.notify(short, type='positive', position='top')

        paint_mode_buttons()   # 初始点亮「裁剪」
        ratio_bar.on_value_change(lambda e: create_ratio_box())
        # 比例尺长度一动,裁剪框跟着按比例缩放(见 on_scale_input_change)
        for _w in (len_input, orig_val, new_val):
            _w.on_value_change(lambda e: on_scale_input_change())
        crop_w_input.on_value_change(lambda e: apply_crop_size('w'))
        crop_h_input.on_value_change(lambda e: apply_crop_size('h'))

        # 结果卡片拖拽排序。NiceGUI 2.24 没有 ui.sortable,自己用 HTML5 dnd 实现:
        # 卡片上带了 rid-xxx 类名,脚本从 class 里认身份,拖完把两个 id 回传。
        # 用 MutationObserver 盯住 DOM —— 结果区每次重画都会换掉整批卡片,
        # 不重新挂监听的话新卡片就拖不动了。
        ui.add_body_html('''
<script>
(function () {
  function initResultDrag() {
    document.querySelectorAll('.result-card').forEach(function (el) {
      if (el.dataset.dragInit) return;
      el.dataset.dragInit = '1';
      el.setAttribute('draggable', 'true');
      el.style.cursor = 'grab';
      el.addEventListener('dragstart', function () {
        window.__dragRid = (el.className.match(/rid-([A-Za-z0-9_]+)/) || [])[1];
        el.style.opacity = '0.35';
      });
      el.addEventListener('dragend', function () { el.style.opacity = '1'; });
      el.addEventListener('dragover', function (ev) { ev.preventDefault(); });
      el.addEventListener('drop', function (ev) {
        ev.preventDefault();
        el.style.opacity = '1';
        var to = (el.className.match(/rid-([A-Za-z0-9_]+)/) || [])[1];
        if (window.__dragRid && to && window.__dragRid !== to) {
          emitEvent('reorder', {from: window.__dragRid, to: to});
        }
      });
    });
  }
  new MutationObserver(function () { setTimeout(initResultDrag, 60); })
    .observe(document.body, {childList: true, subtree: true});
  initResultDrag();
})();
</script>''')
        ui.on('reorder', lambda e: move_result(
            (e.args or {}).get('from', ''), (e.args or {}).get('to', '')))
        generate_btn.on_click(generate)
        batch_btn.on_click(batch_generate)   # 直接传协程函数,别用 lambda 包
        cols_input.on_value_change(lambda e: update_grid_hint())
        rows_input.on_value_change(lambda e: update_grid_hint())
        render_sources()   # 初始:显示"还没有导入图片"占位
        render_results()   # 初始:显示"还没有处理结果"占位 + 刷新行列提示
