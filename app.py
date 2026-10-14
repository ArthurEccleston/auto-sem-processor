"""AutoSEM Processor — 入口。

运行方式:
    python app.py
然后浏览器打开 http://127.0.0.1:8090
(Windows 上也可以双击 run.bat,首次运行会自动装好环境)

换端口/监听地址:
    set LAB_PORT=9000        Linux/mac: export LAB_PORT=9000
    set LAB_HOST=0.0.0.0     # 想让局域网里别的机器访问时才开,注意本机存着 API key
    set LAB_NO_BROWSER=1     # 启动时不自动开浏览器
"""
import os

from nicegui import ui

from core.config import ensure_dirs
import views  # noqa: F401  导入 views 即注册页面路由
from views import theme


def main() -> None:
    ensure_dirs()
    theme.apply()  # 配色 + 全局样式
    # 默认只听本机:这个应用里存着 API key,不该默认对局域网敞开。
    host = os.environ.get('LAB_HOST', '127.0.0.1')
    port = int(os.environ.get('LAB_PORT', '8090'))
    ui.run(
        title='AutoSEM Processor',
        host=host,
        port=port,
        reload=False,            # 开发时可改 True 自动热重载
        show=os.environ.get('LAB_NO_BROWSER') != '1',   # 测试时可跳过自动开浏览器
        favicon='🔬',
    )


if __name__ == '__main__':
    main()
