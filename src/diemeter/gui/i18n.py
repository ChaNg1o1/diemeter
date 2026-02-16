"""Lightweight dict-based i18n for the measurement GUI."""

from __future__ import annotations

_STRINGS: dict[str, dict[str, str]] = {
    # -- Mode names --
    "mode.view": {"en": "View", "zh": "查看"},
    "mode.calibrate_line": {"en": "Cal-Line", "zh": "线校准"},
    "mode.calibrate_rect": {"en": "Cal-Rect", "zh": "矩形校准"},
    "mode.polygon_draw": {"en": "Polygon", "zh": "多边形"},
    "mode.polygon_edit": {"en": "Edit", "zh": "编辑"},
    "mode.grid_detect": {"en": "Grid", "zh": "栅格"},
    # -- Toolbar --
    "toolbar.snap": {"en": "Snap", "zh": "吸附"},
    "toolbar.undo": {"en": "Undo", "zh": "撤销"},
    "toolbar.lang": {"en": "中文", "zh": "EN"},
    # -- Help text per mode --
    "help.view": {
        "en": "Click polygon to select | Scroll=zoom | Middle-drag=pan | Right-click=menu",
        "zh": "点击多边形选中 | 滚轮=缩放 | 中键拖拽=平移 | 右键=菜单",
    },
    "help.calibrate_line": {
        "en": "Click two points, enter known distance | Right-click=cancel",
        "zh": "点击两点，输入已知距离 | 右键=取消",
    },
    "help.calibrate_rect": {
        "en": "Click four corners (TL->TR->BR->BL), enter W x H | Right-click=cancel",
        "zh": "点击四角 (左上->右上->右下->左下)，输入 宽 x 高 | 右键=取消",
    },
    "help.polygon_draw": {
        "en": "Click to add vertices | Double-click or right-click=close | Right-click=menu",
        "zh": "点击添加顶点 | 双击或右键=闭合 | 右键=菜单",
    },
    "help.polygon_edit": {
        "en": "Drag vertex | Right-click=menu (delete/insert vertex) | Enter=done",
        "zh": "拖拽顶点 | 右键=菜单 (删除/插入顶点) | Enter=完成",
    },
    "help.grid_detect": {
        "en": "Drag to select ROI for pattern detection | Right-click=cancel",
        "zh": "拖拽选择 ROI 进行栅格检测 | 右键=取消",
    },
    # -- Dialog prompts --
    "dialog.enter_distance": {
        "en": "Enter known distance (e.g. '14.5 mm'):",
        "zh": "输入已知距离 (例如 '14.5 mm'):",
    },
    "dialog.enter_rect_dim": {
        "en": "Enter dimensions 'W x H unit' (e.g. '10 x 8 mm'):",
        "zh": "输入尺寸 '宽 x 高 单位' (例如 '10 x 8 mm'):",
    },
    "dialog.perspective_warp": {
        "en": "Apply perspective correction?",
        "zh": "是否应用透视校正？",
    },
    "dialog.export_path": {
        "en": "Export path (e.g. 'results.csv' or 'results.json'):",
        "zh": "导出路径 (例如 'results.csv' 或 'results.json'):",
    },
    "dialog.calibrate_line_title": {
        "en": "Line Calibration",
        "zh": "线校准",
    },
    "dialog.calibrate_rect_title": {
        "en": "Rectangle Calibration",
        "zh": "矩形校准",
    },
    "dialog.export_title": {
        "en": "Export Results",
        "zh": "导出结果",
    },
    "dialog.perspective_title": {
        "en": "Perspective Correction",
        "zh": "透视校正",
    },
    # -- Context menu --
    "ctx.edit_polygon": {"en": "Edit polygon", "zh": "编辑多边形"},
    "ctx.delete_polygon": {"en": "Delete polygon", "zh": "删除多边形"},
    "ctx.close_polygon": {"en": "Close polygon", "zh": "闭合多边形"},
    "ctx.cancel": {"en": "Cancel", "zh": "取消"},
    "ctx.delete_vertex": {"en": "Delete vertex", "zh": "删除顶点"},
    "ctx.insert_vertex": {"en": "Insert vertex", "zh": "插入顶点"},
    "ctx.done_editing": {"en": "Done editing", "zh": "完成编辑"},
    # -- Status bar --
    "status.scale": {"en": "Scale: {ppu:.2f} px/{unit} ± {unc:.2f}", "zh": "比例: {ppu:.2f} px/{unit} ± {unc:.2f}"},
    "status.not_calibrated": {"en": "Scale: not calibrated", "zh": "比例: 未校准"},
    "status.snap_on": {"en": "Snap: ON", "zh": "吸附: 开"},
    "status.snap_off": {"en": "Snap: OFF", "zh": "吸附: 关"},
    "status.cursor": {"en": "({x:.1f}, {y:.1f}) px", "zh": "({x:.1f}, {y:.1f}) px"},
    "status.cursor_phys": {
        "en": " = ({px:.3f}, {py:.3f}) {unit}",
        "zh": " = ({px:.3f}, {py:.3f}) {unit}",
    },
    "status.current_area_px": {
        "en": "Area: {area:.1f} px²",
        "zh": "面积: {area:.1f} px²",
    },
    "status.current_area_phys": {
        "en": " = {area:.4g} {unit}²",
        "zh": " = {area:.4g} {unit}²",
    },
    "status.grid": {
        "en": "Grid: pitch=({px:.3g}, {py:.3g}) {unit}, conf={conf:.2f}",
        "zh": "栅格: 间距=({px:.3g}, {py:.3g}) {unit}, 置信度={conf:.2f}",
    },
    # -- Window --
    "window.title": {"en": "diemeter", "zh": "diemeter"},
}

_current_lang: str = "en"


def tr(key: str, **kwargs: object) -> str:
    """Look up a translated string by key, with optional format arguments."""
    entry = _STRINGS.get(key)
    if entry is None:
        return key
    text = entry.get(_current_lang, entry.get("en", key))
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError):
            return text
    return text


def set_language(lang: str) -> None:
    """Set the current language ('en' or 'zh')."""
    global _current_lang
    if lang not in ("en", "zh"):
        raise ValueError(f"Unsupported language: {lang!r}")
    _current_lang = lang


def get_language() -> str:
    """Return the current language code."""
    return _current_lang
