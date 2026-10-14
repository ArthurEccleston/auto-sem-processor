"""界面包:只有一个页面,import 即注册路由。

新增页面的步骤(如果以后要加):
1. 新建 views/xxx.py,用 @ui.page('/xxx') 装饰一个函数
2. 在文件底部 import 进来
"""
from __future__ import annotations

import views.sem  # noqa: F401  导入即注册 @ui.page('/')
