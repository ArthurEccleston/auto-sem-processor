"""全局主题:灰白配色 + 等线/雅黑字体分层。

在 app.py 启动时调用一次 theme.apply()。字体分层靠 @font-face 的 local()
探测:装了等线(Windows)就用等线渲染字母数字,没装就整段跳过、走后面的
系统字体 —— 在 Linux/mac 上不会报错,只是少一层中文/西文混排的优化。
"""
from __future__ import annotations

from nicegui import ui

# ---------- 配色 ----------
PRIMARY = '#3964FE'    # 主色:宝蓝(按钮/链接/焦点)
PAGE_BG = '#F5F6F8'    # 页面浅灰:卡片是纯白的,底色得压一档才浮得起来
TEXT = '#1F2937'       # 正文深灰
MUTED = '#6B7280'      # 次要灰

# 使用占位符 + replace 避免大段 CSS 里的花括号转义
_CSS = '''
/* 字体分层:英文字母和数字用等线,汉字用微软雅黑(没装则整段回退) */
@font-face {
  font-family: 'DengXianLatin';
  src: local('DengXian'), local('等线');
  unicode-range: U+0000-024F, U+2000-206F;
}
html, body, .q-page, .q-layout, .q-card,
.q-btn__content, .q-field, .q-badge, .q-item__label, .q-uploader {
  font-family: 'DengXianLatin', 'Microsoft YaHei', '微软雅黑',
               'Noto Sans CJK SC', 'PingFang SC', 'Source Han Sans SC',
               sans-serif !important;
}

/* 同时覆盖 :root 和 body(NiceGUI 把 --q-primary 等内联设在 body 上,
   body 自己的内联值会遮蔽 html 继承值,所以 body 也要加 important 才能赢) */
:root, body {
  --q-primary: __PRIMARY__ !important;
  --q-secondary: __MUTED__ !important;
  --q-accent: __PRIMARY__ !important;
  --q-info: __MUTED__ !important;
  --q-dark: __TEXT__ !important;
}

/* 所有链接去掉下划线 */
a, a:hover, a:visited, .q-link { text-decoration: none !important; }

/* 页面底色 */
body, .q-page { background-color: __PAGE_BG__; }

/* ---------- 卡片 ---------- */
.q-card {
  border-radius: 14px !important;
  background-color: #FFFFFF !important;
  border: none !important;
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.12),
              0 1px 3px rgba(15, 23, 42, 0.14) !important;
}
/* 对话框是浮层,阴影可以更重一点,和背景拉开层次 */
.q-dialog .q-card {
  border-radius: 14px !important;
  background-color: #FFFFFF !important;
  border: none !important;
  box-shadow: 0 12px 40px rgba(15, 23, 42, 0.20),
              0 2px 8px rgba(15, 23, 42, 0.10) !important;
}

/* 进页面时:**每张卡片**从下往上滑入。位移 18px、时长 .42s ——
   幅度收着点、节奏干脆。.page-in 只是作用域,对话框里的卡片不受影响。 */
@keyframes cardIn {
  from { opacity: 0; transform: translateY(18px); }
  to   { opacity: 1; transform: translateY(0); }
}
.page-in .q-card {
  animation: cardIn .42s cubic-bezier(.22, .61, .36, 1) both;
}

/* ---------- 方角 / 圆角按钮 ---------- */
/* 按钮/输入框/下拉/表格保持方角 */
.q-btn { border-radius: 0 !important; }
.q-btn--round { border-radius: 50% !important; }

/* 按钮想用圆角矩形就加 .classes('rounded-btn')。
   注意上面那条 `.q-btn` 带 !important,普通的圆角类压不住它 —— 两条都带
   !important 时按**选择器特异性**决胜,所以这里用 `.q-btn.rounded-btn`(0,2,0)
   压过 `.q-btn`(0,1,0)。(Quasar 的类叫 .rounded-borders,Tailwind 也不生成
   rounded-btn 这个名字,自己不用这个类的话它从来没生效过。) */
.q-btn.rounded-btn { border-radius: 8px !important; }

.q-field__control { border-radius: 0 !important; }
.q-menu, .q-expansion-item__container, .q-table { border-radius: 0 !important; }

/* ---------- 上传框 ---------- */
.q-uploader {
  border-radius: 16px !important;
  width: 100% !important;
  overflow: hidden;
}
/* 主上传框:宽40高12(rem)、居中、不贴边、文字居中 */
.sem-upload.q-uploader {
  width: 40rem !important;
  height: 12rem !important;
}
.sem-upload .q-uploader__header-content { justify-content: center !important; }
.sem-upload .q-uploader__list { overflow-y: auto; }
.q-uploader__header .q-btn {
  background-color: rgba(255, 255, 255, 0.22) !important;
  color: #fff !important;
  border-radius: 6px !important;
}
'''


def apply() -> None:
    """应用全局主题(对所有页面生效)。启动时调一次。"""
    css = _CSS
    for token, value in {
        '__PRIMARY__': PRIMARY, '__PAGE_BG__': PAGE_BG,
        '__TEXT__': TEXT, '__MUTED__': MUTED,
    }.items():
        css = css.replace(token, value)
    ui.add_css(css, shared=True)
