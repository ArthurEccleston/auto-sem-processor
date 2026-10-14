"""「⚙ API 设置」弹窗。

只干三件事:显示当前配置从哪来、改它、测它通不通。配置的读写细节都在
core/settings.py 里,这里只做界面。

**保存后立刻生效**,不用重启服务:settings.save_vision() 会清自己的内存缓存,
下面再调一次 vision.invalidate() 把读数缓存也清掉(所以不能反过来让
core/settings.py 去 import vision —— 会成环)。
"""
from __future__ import annotations

from nicegui import run, ui

from core import settings
from integrations import vision

# 常见端点的预设:选中只填 base_url + 一个典型视觉模型名,填完照样能手改。
# base_url 都是"根地址",程序会自己补 /chat/completions;直接粘整条 endpoint 也行。
PRESETS: dict[str, tuple[str, str]] = {
    '— 手动填写 —': ('', ''),
    'DeepSeek': ('https://api.deepseek.com', 'deepseek-v4-flash-vision-exp'),
    'OpenAI': ('https://api.openai.com/v1', 'gpt-4o-mini'),
    '通义千问(阿里云)': ('https://dashscope.aliyuncs.com/compatible-mode/v1',
                        'qwen-vl-max'),
    '智谱 GLM': ('https://open.bigmodel.cn/api/paas/v4', 'glm-4v'),
    'Ollama(本地)': ('http://localhost:11434/v1', 'qwen2.5vl'),
    'vLLM(本地)': ('http://localhost:8000/v1', ''),
}

_BASE_HINT = 'OpenAI 格式的根地址,通常要带 /v1;直接粘整条 …/chat/completions 也行'
_MODEL_HINT = '要能看图的模型(多模态):qwen-vl-max / gpt-4o-mini / glm-4v …'


def refresh_api_status(label) -> None:
    """把「已配置 / 未配置」写到页面右上角那行文字上。"""
    st = settings.status()
    if st['configured']:
        label.set_text(f'● 已配置 · {st["model"]} · 来源:{settings.source_label(st["source"])}')
        label.classes(replace='text-sm text-green-600')
    else:
        label.set_text('○ 未配置 API · 比例尺数值需手填(其它功能不受影响)')
        label.classes(replace='text-sm text-slate-400')


def open_settings_dialog(status_label) -> None:
    """打开设置弹窗。status_label 是页面右上角那行状态文字,保存后要刷新它。"""
    cur = settings.load(refresh=True)

    with ui.dialog() as dlg, ui.card().classes('p-6').style('width: 34rem; max-width: 94vw'):
        with ui.row().classes('w-full items-center gap-3'):
            ui.label('⚙ API 设置').classes('text-xl font-bold')
            ui.label('配好之后,导入图片时自动读出比例尺上的数值') \
                .classes('text-xs text-slate-400')
            ui.space()
            ui.button(icon='close', on_click=dlg.close).props('flat round dense')

        ui.separator().classes('my-2')

        preset_sel = ui.select(list(PRESETS), label='常用端点(选了会自动填下面两项,可改)',
                               value='— 手动填写 —').classes('w-full')

        base_input = ui.input('base_url', value=cur.base_url, placeholder=_BASE_HINT) \
            .classes('w-full')
        key_input = ui.input('api_key', value=cur.api_key, password=True,
                             password_toggle_button=True,
                             placeholder='本地 ollama 通常不用填') \
            .classes('w-full')
        model_input = ui.input('model', value=cur.model, placeholder=_MODEL_HINT) \
            .classes('w-full')

        result_label = ui.label('').classes('text-sm mt-1').style('min-height: 1.4rem')

        with ui.row().classes('w-full items-center gap-2'):
            test_btn = ui.button('测试连接', icon='wifi_tethering') \
                .props('outline color=primary').classes('rounded-btn')

            async def do_test() -> None:
                cfg = settings.VisionConfig(
                    api_key=(key_input.value or '').strip(),
                    base_url=(base_input.value or '').strip(),
                    model=(model_input.value or '').strip(),
                    source='gui')
                test_btn.disable()
                result_label.set_text('测试中…')
                result_label.classes(replace='text-sm mt-1 text-slate-500')
                try:
                    ok, msg = await run.io_bound(settings.test_connection, cfg)
                except Exception as ex:      # noqa: BLE001
                    ok, msg = False, f'测试出错:{ex}'
                finally:
                    test_btn.enable()
                result_label.set_text(('✅ ' if ok else '❌ ') + msg)
                result_label.classes(
                    replace=f'text-sm mt-1 {"text-green-600" if ok else "text-red-600"}')

            test_btn.on_click(do_test)

        ui.separator().classes('my-2')

        def do_save() -> None:
            settings.save_vision((key_input.value or '').strip(),
                                 (base_input.value or '').strip(),
                                 (model_input.value or '').strip())
            vision.invalidate()          # 配置变了,读数缓存要跟着清
            refresh_api_status(status_label)
            ui.notify('已保存,立即生效(不用重启)', type='positive')
            dlg.close()

        def do_clear() -> None:
            settings.clear_gui()
            vision.invalidate()
            refresh_api_status(status_label)
            ui.notify('已清除界面里保存的配置', type='positive')
            dlg.close()

        with ui.row().classes('w-full justify-end gap-2'):
            ui.button('清除界面配置', icon='delete_sweep', on_click=do_clear) \
                .props('flat color=grey').classes('rounded-btn')
            ui.button('保存', icon='save', color='primary', on_click=do_save) \
                .classes('rounded-btn')

        ui.label(f'当前生效:{settings.source_label(cur.source)}'
                 f' · key {cur.masked_key} · 配置写到项目根目录的 .env.local'
                 f'(不进 git)') \
            .classes('text-xs text-slate-400 mt-1')
        ui.label('不配也能用:比例尺的「像素长度」是本地算的,只有要自动读出'
                 '“40 μm”这个数值时才需要 API;读不出来照旧手填即可。') \
            .classes('text-xs text-slate-400')

        def apply_preset(e) -> None:
            base, model = PRESETS.get(e.value or '', ('', ''))
            if base:
                base_input.value = base
            if model:
                model_input.value = model

        preset_sel.on_value_change(apply_preset)

    dlg.open()
