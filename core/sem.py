"""SEM 图像分析与处理。

功能:
- 正确读取 8/16 位图像(16 位做百分位归一化;浏览器只认 8 位,预览另存缩略图)
- 在底部元数据栏里检测比例尺横线(认 FEI / ZEISS 两种仪器形态)
- 在裁剪图上绘制白色比例尺(横线 + 数值)和左上/右上角标
- 把多张图拼成一张网格整合图

依赖:opencv-python-headless、pillow、numpy。**不需要 Tesseract** ——
比例尺上的数值交给视觉模型读(见 integrations/vision.py),本地只算像素几何。
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------- 读取 ----------

def _normalize16(img: np.ndarray) -> np.ndarray:
    """16 位图像按百分位归一化到 8 位,保证深色 SEM 图和白线都可见。"""
    lo, hi = np.percentile(img, (0.5, 99.5))
    img8 = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255)
    return img8.astype(np.uint8)


def load_gray(path: str) -> np.ndarray:
    """读取灰度图(支持中文路径、8/16 位)。"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'无法读取图片: {path}')
    if img.dtype != np.uint8:
        img = _normalize16(img)
    return img


def load_rgb(path: str) -> np.ndarray:
    """读取彩色图(支持中文路径、8/16 位)。"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'无法读取图片: {path}')
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def make_thumbnail(src: str, dst: str, max_w: int = 320) -> None:
    """生成一张浏览器能直接显示的 8 位 PNG 缩略图。

    预览条如果直接指向原图,16 位 TIFF/BMP 在浏览器里就是一片空白
    (浏览器只认 8 位 PNG/JPEG/WebP)。这里统一转码一份小图专供预览,
    原图原样保留,不影响后续处理链路。
    """
    rgb = load_rgb(src)
    h, w = rgb.shape[:2]
    if w > max_w:
        rgb = cv2.resize(rgb, (max_w, max(1, round(h * max_w / w))),
                         interpolation=cv2.INTER_AREA)
    Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)).save(dst)


# ---------- 比例尺横线检测 ----------

def _horizontal_segments(bw: np.ndarray, min_len: int) -> list[dict]:
    """提取二值图里所有水平线段(合并相邻行),返回 [{y, x1, x2, length}]。"""
    h, w = bw.shape
    # 每行找连续亮段
    segs = []
    for y in range(h):
        row = bw[y]
        start = None
        for x in range(w):
            if row[x]:
                if start is None:
                    start = x
            else:
                if start is not None and x - start >= min_len:
                    segs.append({'y': y, 'x1': start, 'x2': x - 1, 'length': x - start})
                start = None
        if start is not None and w - start >= min_len:
            segs.append({'y': y, 'x1': start, 'x2': w - 1, 'length': w - start})
    return segs


# 底部元数据栏里的比例尺(主力路径)
#
# 这类仪器的比例尺是连同参数一起**烧录在底部那条黑底白字的元数据栏**里的。
# 栏里比例尺有**两种形态**,判据要同时认(FEI 和 ZEISS 各占一种):
#
#   FEI   `|—— 40 μm ——|` —— 一条横线被中间的数值文字**隔断成左右两段**。
#         栏内各格之间的格线是连续不断的一整条,所以判据取「同一行里出现
#         ≥2 段」就能把格线和边框全部排除。实测 26/26 全中,并用图上的
#         HFW 交叉验证过。
#         ⚠ 这一形态**优先级最高**:FEI 栏里还另有一条更长的格线能骗过竖刻度
#         判据(见 detect_scale_bar_in_strip 里「形态 1」的注释),所以它一旦
#         出现就直接采用,不再和下面的候选比长度。
#
#   ZEISS `10 μm` 在上、`├────────┤` 在下 —— 横线**连续不断**,但两端各有
#         一小段竖刻度。这条横线 n_segs==1,会被上面的「≥2 段」判据直接
#         毙掉,所以另加一个判据:_end_tick_height() 量端点竖刻度的高度。
#         实测竖刻度 15px、横线本体只有 2~3px,对比极强,不会误判。

_TICK_MIN = 8   # 端点竖刻度的最小可见高度(px),见 _end_tick_height


def _dark_strip(gray: np.ndarray, th: int = 120, bright: float = 0.5,
                min_h: int = 10, max_h: float = 0.15, bottom_frac: float = 0.6,
                bottom_margin: float = 0.95) -> Optional[tuple[int, int]]:
    """定位底部黑色元数据栏,返回 (y0, y1);没有这样一整条则 None。

    按行求亮占比 → 切出连续暗行段 → 三重筛后取最高的那段。

    ⚠ 不要改回「从最底行往上逐行走」——ZEISS 这类仪器的图底有一两行半亮的
    抗锯齿像素(实测 10-1.tif 最底行亮占比 0.409,既不算亮也不算暗),那个
    假设会直接带偏:跳亮行的循环不执行,走暗行的循环又立刻撞上栏底那条白线
    停下,最后返回高度 0 的栏 → None。

    ⚠ 也不能只按「最长连续暗段」选:内容区的暗块比栏长得多(实测 8-2 有
    631 行连续暗行),只比长度会选错。真正把关的是**段高上限** max_h ——
    实测各仪器的栏高都只占全图的 7.0~7.2%(77/1103、55/768、111/1536),
    给一倍余量就够,而内容暗块远超这个比例。
    """
    h = gray.shape[0]
    dark = [(gray[y] > th).mean() <= bright for y in range(h)]
    runs: list[tuple[int, int]] = []
    cur: Optional[tuple[int, int]] = None
    for y in range(int(h * bottom_frac), h):
        if dark[y]:
            cur = (y, y) if cur is None else (cur[0], y)
        elif cur is not None:
            runs.append(cur)
            cur = None
    if cur is not None:
        runs.append(cur)
    # 三重筛:① 高度落在 [min_h, max_h] ② 段底贴着图底(内容区暗块够不到图底)
    runs = [r for r in runs
            if min_h < r[1] - r[0] + 1 <= h * max_h
            and r[1] >= h * bottom_margin]
    return max(runs, key=lambda r: r[1] - r[0]) if runs else None


def _end_tick_height(gray: np.ndarray, x: int, y: int,
                     half: int = 12, th: int = 180) -> int:
    """比例尺端点竖刻度的可见高度:x 列在 y±half 内有多少行是亮的。

    ZEISS 的比例尺是「数值在横线上方 + 横线连续 + 两端各一小段竖刻度」,
    竖刻度就是把它和栏内普通文字/格线区分开的东西 —— 实测刻度高 15px,
    而横线本体只有 2~3px(逐列亮像素数 15 : 1)。

    ⚠ 必须在**未做开运算的原始灰度**上量:竖刻度只有约 3px 宽,会被
    (7,1) 的横向开运算整段腐蚀掉,在 bw 上量不出这个高度差。
    """
    h, w = gray.shape
    y0, y1 = max(0, y - half), min(h, y + half + 1)
    x0, x1 = max(0, x - 1), min(w, x + 2)
    if y1 <= y0 or x1 <= x0:
        return 0
    return int(((gray[y0:y1, x0:x1] > th).any(axis=1)).sum())


def detect_scale_bar_in_strip(gray: np.ndarray) -> Optional[dict]:
    """在底部元数据栏里检测比例尺横线,返回 {y, x1, x2, length, n_segs}。

    length 就是页面要的那个「原比例尺像素长度」,即比例尺两端的整体跨度 ——
    FEI 型(横线被文字隔断)要把中间文字占的那段算进去,不能只量其中一段;
    ZEISS 型(横线连续)本来就是整条,直接取首尾即可。

    两种形态的判据见上方注释块。找不到返回 None(由调用方退回手动量取)。
    """
    w = gray.shape[1]
    strip = _dark_strip(gray)
    if strip is None:
        return None
    y0, y1 = strip
    region = gray[y0:y1 + 1, :]

    # 白线二值化 + 水平开运算:先削弱文字笔画的干扰,只留偏长的水平结构
    bw = cv2.threshold(region, 180, 255, cv2.THRESH_BINARY)[1]
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1)))

    min_len = max(8, int(w * 0.03))
    rows: dict[int, list[tuple[int, int]]] = {}
    for s in _horizontal_segments(bw, min_len):
        rows.setdefault(s['y'], []).append((s['x1'], s['x2']))
    if not rows:
        return None

    # 比例尺横线有 2~3px 厚,逐行看会把同一段数成好几行 —— 按 y 邻域
    # (±3)聚成一条候选,再统计「单行最多出现过几段」。
    bands: list[dict] = []
    for yy in sorted(rows):
        if bands and yy - bands[-1]['y_last'] <= 3:
            bands[-1]['rows'][yy] = rows[yy]
            bands[-1]['y_last'] = yy
        else:
            bands.append({'y_first': yy, 'y_last': yy, 'rows': {yy: rows[yy]}})

    cands = []
    for b in bands:
        segs = [s for line in b['rows'].values() for s in line]
        x1 = min(s[0] for s in segs)
        x2 = max(s[1] for s in segs)
        length = x2 - x1
        # 太短的像噪点/标点,太长的像贯通边框
        # ⚠ 上限别按「实测比例尺只占 5~9% 图宽」收紧 —— FEI 的条长占 19~30%,
        #   收紧到 0.15 会让全部 FEI 图报废。
        if not (w * 0.03 <= length <= w * 0.45):
            continue
        cands.append({'y': b['y_first'] + y0, 'x1': x1, 'x2': x2,
                      'length': length,
                      'n_segs': max(len(line) for line in b['rows'].values())})
    if not cands:
        return None

    # ---- 形态 1:FEI。横线被数值文字隔断成 ≥2 段。----
    # ⚠ 这一支必须**先判、判完就返回**,不能跟下面那支混在一个池子里按 length 排序。
    #   FEI 栏里恒有一条更长的格线(实测 1.tif 的 y=1063,len 459,一直顶到右边框),
    #   它两端也各有一根竖线,竖刻度判据照样通过 —— 按长度排就会选中这条假的,
    #   26 张 FEI 全部回归。
    fei = [c for c in cands if c['n_segs'] >= 2]
    if fei:
        return max(fei, key=lambda c: c['length'])

    # ---- 形态 2:ZEISS。横线连续,但两端各有短竖刻度。----
    # ZEISS 栏内通常只有这一个候选,所以上面那支不会挡住它。
    best = None
    for c in cands:
        if c['x1'] <= 1 or c['x2'] >= w - 3:
            continue        # 贴着图左右边缘的是边框,不是比例尺端点
        t1 = _end_tick_height(gray, c['x1'], c['y'])
        t2 = _end_tick_height(gray, c['x2'], c['y'])
        if t1 < _TICK_MIN or t2 < _TICK_MIN:
            continue
        if best is None or c['length'] > best['length']:
            best = c
    return best


# ---------- 字体(跨平台) ----------

# 比例尺上的 "μm" 和角标是画在图上烧死的,字体找不到就会渲染成豆腐块,
# 所以这里按平台给一串候选路径,挨个试第一个存在的。
_FONT_CANDIDATES = (
    # Windows
    r'C:\Windows\Fonts\arial.ttf',
    r'C:\Windows\Fonts\segoeui.ttf',
    r'C:\Windows\Fonts\msyh.ttc',
    r'C:\Windows\Fonts\simhei.ttf',
    # Linux:Debian/Ubuntu、Fedora、Arch 三套命名都覆盖
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    '/usr/share/fonts/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/TTF/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
    # macOS
    '/System/Library/Fonts/Supplemental/Arial.ttf',
    '/Library/Fonts/Arial.ttf',
    '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
    '/System/Library/Fonts/Helvetica.ttc',
    '/System/Library/Fonts/PingFang.ttc',
)
_USER_FONT_DIRS = ('~/.local/share/fonts', '~/.fonts', '~/Library/Fonts')
_FONT_GLOBS = ('DejaVuSans.ttf', 'Arial.ttf', 'NotoSansCJK*.ttc', 'NotoSans*.ttf')


@lru_cache(maxsize=1)
def _pick_font_path() -> Optional[str]:
    """挑一个能渲染 "μm" 的 TrueType 字体,找不到返回 None。

    整个进程只探测一次(字体装在哪儿不会中途变),别放到 _get_font 里每次跑。
    """
    if sys.platform == 'win32':
        windir = os.environ.get('WINDIR') or 'C:\\Windows'
        extra = Path(windir) / 'Fonts'
        for name in ('arial.ttf', 'segoeui.ttf', 'msyh.ttc', 'simhei.ttf'):
            p = extra / name
            if p.is_file():
                return str(p)
    for p in _FONT_CANDIDATES:
        if Path(p).is_file():
            return p
    for d in _USER_FONT_DIRS:
        base = Path(d).expanduser()
        if not base.is_dir():
            continue
        for pattern in _FONT_GLOBS:
            hit = next(iter(base.rglob(pattern)), None)
            if hit is not None:
                return str(hit)
    try:
        # 装了 matplotlib 的话借它自带的 DejaVu(只在这里用,不是硬依赖)
        from matplotlib import font_manager
        return font_manager.findfont('DejaVu Sans')
    except Exception:      # noqa: BLE001
        return None


@lru_cache(maxsize=64)
def _get_font(size: int):
    """按字号取字体对象(缓存:批量几十张时不会反复读 TTF)。"""
    path = _pick_font_path()
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001
            pass
    try:
        # Pillow ≥10.1 才会按 size 缩放;更老的版本忽略 size,字会很小但不崩
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ---------- 绘制比例尺 ----------

def compose_image(img: np.ndarray, bar_px: int, value: float, unit: str = 'μm',
                  position: str = 'bottom-left', color: str = 'white',
                  label_tl: str = '', label_tr: str = '',
                  margin_pct: float = 3.0, thickness_pct: float = 0.4,
                  bar_font_pct: float = 3.0,
                  label_font_pct: Optional[float] = None) -> np.ndarray:
    """在图上合成最终结果:比例尺(数字居中在横线上方) + 左上/右上标签。

    bar_px: 比例尺横线像素长度(这条必须对应真实物理比例,所以保持像素)。
    value/unit: 数值标注,如 value=2, unit='μm' → '2 μm'。
    position: 'bottom-left' / 'bottom-right'(固定位置,留边不贴边)。
    color: 'white' / 'black'(比例尺与文字颜色,自动加反色描边保证可见)。

    除横线长度以外的尺寸,一律**按图宽的百分比**给(字号、线粗、留边、端点、描边)。
    这样裁剪框无论拉多大缩多小,字和线在图里的相对大小都恒定;几张结果图在 PPT 里
    统一缩到同宽时,比例尺看着一样大。若改成绝对像素,小图被放大显示时字会显得特别大。
    label_font_pct 不传就跟随比例尺字号。
    """
    line_color = (255, 255, 255) if color == 'white' else (0, 0, 0)
    outline = (0, 0, 0) if color == 'white' else (255, 255, 255)

    rgb = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    pil = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    h, w = rgb.shape[:2]

    def px(pct: float, lo: int = 1) -> int:
        """图宽百分比 → 像素(带下限,免得图很小时算出来是 0)。"""
        return max(lo, int(round(w * pct / 100.0)))

    font_size = px(bar_font_pct, 8)
    label_size = px(label_font_pct, 8) if label_font_pct else font_size
    margin = px(margin_pct, 4)
    thickness = px(thickness_pct, 1)
    tick = px(0.55, 3)          # 比例尺两端短竖线的半高
    gap = px(0.9, 2)            # 数字与横线之间的空隙
    outline_w = px(0.13, 1)     # 描边也按比例,否则小图上描边会糊住笔画
    font = _get_font(font_size)  # _get_font 优先 arial
    label_font = _get_font(label_size)

    def outlined_line(p0, p1, fill, width):
        draw.line((*p0, *p1), fill=outline, width=width + 2)
        draw.line((*p0, *p1), fill=fill, width=width)

    def outlined_text(pos, text, fnt):
        tx, ty = pos
        for dx in (-outline_w, 0, outline_w):
            for dy in (-outline_w, 0, outline_w):
                if dx or dy:
                    draw.text((tx + dx, ty + dy), text, fill=outline, font=fnt)
        draw.text((tx, ty), text, fill=line_color, font=fnt)

    # ---- 比例尺(底部固定位置,留边不贴边) ----
    line_x0 = margin if position == 'bottom-left' else w - margin - bar_px
    line_y = h - margin
    line_y = max(line_y, font_size + margin + gap)  # 上方有空间放数字
    # 横线 + 描边
    outlined_line((line_x0, line_y), (line_x0 + bar_px, line_y), line_color, thickness)
    # 端点短竖线
    for ex in (line_x0, line_x0 + bar_px):
        outlined_line((ex, line_y - tick), (ex, line_y + tick), line_color, thickness)
    # 数字居中在横线上方
    text = f'{value:g} {unit}'
    tw = draw.textlength(text, font=font)
    text_x = line_x0 + max(0, (bar_px - tw) / 2)
    text_y = line_y - font_size - gap
    outlined_text((text_x, text_y), text, font)

    # ---- 角落标签(字号独立可调,不传就跟随比例尺) ----
    if label_tl:
        outlined_text((margin, margin), label_tl, label_font)
    if label_tr:
        tw = draw.textlength(label_tr, font=label_font)
        outlined_text((w - margin - tw, margin), label_tr, label_font)

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ---------- 整合图(多张拼成一张,每行 N 个) ----------

def compose_grid(paths: list, cols: int = 3, rows: Optional[int] = None,
                 cell_w: int = 640, gap: int = 12,
                 bg: tuple = (255, 255, 255)) -> np.ndarray:
    """把多张 SEM 图拼成一张整合图。

    每张图按第一张的宽高比做 cover 缩放(填满格子、居中裁掉多余),
    所以不管原图尺寸差多少,拼出来都是整齐的网格。

    paths: 图片路径列表(顺序即摆放顺序)。
    cols:  每行几张。
    rows:  行数;传 None 就按图片数量自动算。
           指定了行数而 cols*rows 放不下全部图片时,只取前 cols*rows 张
           (超出部分由调用方提示用户)。
    返回 BGR 图像(与 compose_image 一致,方便同一个出口保存)。
    """
    paths = [str(p) for p in paths]
    if not paths:
        raise ValueError('没有可组合的图片')

    cols = max(1, int(cols))
    if rows is None:
        rows = (len(paths) + cols - 1) // cols
    else:
        rows = max(1, int(rows))
        paths = paths[:rows * cols]   # 放不下的截断,免得画出界

    first = load_rgb(paths[0])
    fh, fw = first.shape[:2]
    cell_h = max(1, int(round(cell_w * fh / fw)))   # 跟第一张同比例

    width = cols * cell_w + (cols + 1) * gap
    height = rows * cell_h + (rows + 1) * gap
    canvas = np.full((height, width, 3), bg, dtype=np.uint8)

    def cover_fit(img: np.ndarray, tw: int, th: int) -> np.ndarray:
        h, w = img.shape[:2]
        scale = max(tw / w, th / h)
        nw, nh = max(tw, int(round(w * scale))), max(th, int(round(h * scale)))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        x0, y0 = (nw - tw) // 2, (nh - th) // 2
        return resized[y0:y0 + th, x0:x0 + tw]

    for i, p in enumerate(paths):
        row, col = divmod(i, cols)
        tile = cover_fit(load_rgb(p), cell_w, cell_h)
        y = gap + row * (cell_h + gap)
        x = gap + col * (cell_w + gap)
        canvas[y:y + cell_h, x:x + cell_w] = tile
    return canvas
