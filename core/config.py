"""全局配置:数据目录。

所有路径都从这里读取,别在别处拼 —— 以后要挪数据位置只改这一个文件。

数据目录全部落在项目内的 `data/` 下,拷走整个文件夹就能接着用;
`data/` 已在 .gitignore 里(里面是用户拖进来的电镜原图,可能涉密)。
"""
from __future__ import annotations

from pathlib import Path

# 项目根目录(本文件在 core/ 下,向上两级)
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / 'data'
SEM_SRC_DIR = DATA_DIR / 'sem_src'    # 拖进来的原图(处理中的工作台内容)
SEM_OUT_DIR = DATA_DIR / 'sem_out'    # 生成的成品图
SEM_GRID_DIR = DATA_DIR / 'sem_grid'  # 拼好的整合图


def ensure_dirs() -> None:
    """保证所有数据目录存在(幂等,可重复调用)。"""
    for d in (DATA_DIR, SEM_SRC_DIR, SEM_OUT_DIR, SEM_GRID_DIR):
        d.mkdir(parents=True, exist_ok=True)


__all__ = ['BASE_DIR', 'DATA_DIR', 'SEM_SRC_DIR', 'SEM_OUT_DIR',
           'SEM_GRID_DIR', 'ensure_dirs']
