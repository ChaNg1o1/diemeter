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
    "mode.calibrate_pad_pitch": {"en": "Pad Pitch", "zh": "焊盘间距"},
    # -- Toolbar --
    "toolbar.snap": {"en": "Snap", "zh": "吸附"},
    "toolbar.undo": {"en": "Undo", "zh": "撤销"},
    "toolbar.save": {"en": "Save", "zh": "保存"},
    "toolbar.export": {"en": "Export", "zh": "导出"},
    "toolbar.unit": {"en": "Unit", "zh": "单位"},
    "toolbar.snap_radius": {"en": "Radius", "zh": "半径"},
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
    "help.calibrate_pad_pitch": {
        "en": "Drag ROI over bond pad row, enter known pitch | Right-click=cancel",
        "zh": "在焊盘行上拖选 ROI，输入已知间距 | 右键=取消",
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
    "dialog.calibration_quality_title": {
        "en": "Calibration Quality Warning",
        "zh": "校准质量警告",
    },
    "dialog.input_error_title": {
        "en": "Input Error",
        "zh": "输入错误",
    },
    "dialog.calibrate_pad_pitch_title": {
        "en": "Bond Pad Pitch Calibration",
        "zh": "焊盘间距校准",
    },
    "dialog.enter_pad_pitch": {
        "en": "Detected pitch: {pitch_px:.2f} px (confidence={conf:.3f}).\nEnter known pad pitch (e.g. '80 um'):",
        "zh": "检测到间距: {pitch_px:.2f} px (置信度={conf:.3f})。\n输入已知焊盘间距 (例如 '80 um'):",
    },
    "dialog.pad_pitch_failed": {
        "en": "No periodic pattern detected in the selected ROI.\nTry selecting a region with a clear row of bond pads.",
        "zh": "在选定 ROI 中未检测到周期性图案。\n请尝试选择有明显焊盘行的区域。",
    },
    "dialog.invalid_length": {
        "en": "Please enter a positive length, e.g. '14.5 mm'.",
        "zh": "请输入正数长度，例如 '14.5 mm'。",
    },
    "dialog.invalid_rect": {
        "en": "Please enter positive dimensions, e.g. '10 x 8 mm'.",
        "zh": "请输入正数尺寸，例如 '10 x 8 mm'。",
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
    "status.snap_on": {"en": "Snap: ON (r={radius}px)", "zh": "吸附: 开 (半径={radius}px)"},
    "status.snap_off": {"en": "Snap: OFF (r={radius}px)", "zh": "吸附: 关 (半径={radius}px)"},
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
    # -- Info panel --
    "info.title": {"en": "Info", "zh": "信息"},
    "info.calibration": {"en": "Calibration", "zh": "校准"},
    "info.not_calibrated": {"en": "Not calibrated", "zh": "未校准"},
    "info.scale": {
        "en": "Scale: {ppu:.4f} px/{unit} ± {unc:.4f}",
        "zh": "比例: {ppu:.4f} px/{unit} ± {unc:.4f}",
    },
    "info.method_line": {"en": "Method: line ({n} lines)", "zh": "方法: 线校准 ({n} 条)"},
    "info.method_rect": {"en": "Method: rect", "zh": "方法: 矩形校准"},
    "info.cal_rel_unc": {
        "en": "Relative scale uncertainty: {rel:.3f}%",
        "zh": "比例相对不确定度: {rel:.3f}%",
    },
    "info.cal_sigma_assumption": {
        "en": "Endpoint localization model: sigma={sigma:.2f} px",
        "zh": "端点定位模型: sigma={sigma:.2f} px",
    },
    "info.cal_consistency": {
        "en": "Consistency: chi2={chi2:.3g}, p={p:.4f}, pass@{conf:.0f}%: {status}",
        "zh": "一致性: chi2={chi2:.3g}, p={p:.4f}, {conf:.0f}%阈值: {status}",
    },
    "info.cal_consistent": {"en": "yes", "zh": "通过"},
    "info.cal_inconsistent": {"en": "no", "zh": "不通过"},
    "info.cal_line_px": {"en": "Length (px)", "zh": "长度(px)"},
    "info.cal_line_known": {"en": "Known", "zh": "已知长度"},
    "info.cal_line_sigma": {"en": "sigma_ppu", "zh": "PPU标准差"},
    "info.cal_line_delta_pct": {"en": "Delta %", "zh": "偏差%"},
    "info.cal_line_source": {"en": "Source", "zh": "来源"},
    "info.perspective": {"en": "Perspective: corrected", "zh": "透视: 已校正"},
    "info.polygon": {"en": "Polygon", "zh": "多边形"},
    "info.none_selected": {"en": "None selected", "zh": "未选中"},
    "info.area": {
        "en": "Area: {area:.6g} {unit}² ± {unc:.4g} ({rel:.2f}%)",
        "zh": "面积: {area:.6g} {unit}² ± {unc:.4g} ({rel:.2f}%)",
    },
    "info.unc_decomp": {
        "en": "  vertex σ: {v:.4g} {unit}²  |  ppu σ: {p:.4g} {unit}²",
        "zh": "  顶点σ: {v:.4g} {unit}²  |  比例σ: {p:.4g} {unit}²",
    },
    "info.perimeter": {
        "en": "Perimeter: {perim:.6g} {unit} ± {unc:.4g}",
        "zh": "周长: {perim:.6g} {unit} ± {unc:.4g}",
    },
    "info.bbox": {
        "en": "BBox: {w:.6g} × {h:.6g} {unit} ± ({wu:.4g}, {hu:.4g})",
        "zh": "外接矩形: {w:.6g} × {h:.6g} {unit} ± ({wu:.4g}, {hu:.4g})",
    },
    "info.bbox_angle": {
        "en": "BBox angle: {angle:.1f}°",
        "zh": "外接矩形角度: {angle:.1f}°",
    },
    "info.vertices_summary": {
        "en": "Vertices: {n}  |  Area(px): {apx:.2f}",
        "zh": "顶点: {n}  |  面积(px): {apx:.2f}",
    },
    "info.not_measured": {"en": "Not measured (no calibration)", "zh": "未测量 (未校准)"},
    "info.vtx_snap": {"en": "snap", "zh": "吸附"},
    "info.vtx_conf": {"en": "conf", "zh": "置信度"},
    "info.vtx_sigma": {"en": "sigma(px)", "zh": "sigma(px)"},
    "info.vtx_manual": {"en": "manual", "zh": "手动"},
    "info.grid": {"en": "Grid", "zh": "栅格"},
    "info.no_detections": {"en": "No detections", "zh": "无检测"},
    "info.grid_detections": {"en": "Grid Detections", "zh": "栅格检测"},
    # -- Calibration quality warnings --
    "warn.cal.single_line_short": {
        "en": "Single-line calibration is underconstrained. Line length={px:.1f}px (<{min_px:.0f}px recommended), endpoint sigma={sigma:.2f}px, current relative scale uncertainty={rel:.3f}%. Use longer reference, add more lines, or switch to rectangle calibration.",
        "zh": "单线校准约束不足。线长={px:.1f}px（建议>={min_px:.0f}px），端点sigma={sigma:.2f}px，当前比例相对不确定度={rel:.3f}%。建议使用更长参考线、增加多条校准线，或改用矩形校准。",
    },
    "warn.cal.inconsistent": {
        "en": "Calibration lines are inconsistent: n={n}, chi2={chi2:.3g}, p={p:.4f} (<{conf:.0f}% consistency threshold). Relative scale uncertainty={rel:.3f}%. Check reference lengths and same-plane assumption.",
        "zh": "校准线不一致：n={n}, chi2={chi2:.3g}, p={p:.4f}（低于{conf:.0f}%一致性阈值）。比例相对不确定度={rel:.3f}%。请检查参考长度输入和同平面假设。",
    },
    "warn.cal.high_unc": {
        "en": "Scale uncertainty is high after line calibration: n={n}, relative uncertainty={rel:.3f}% (target <= {threshold:.3f}%). Add longer/more calibration lines or use rectangle calibration.",
        "zh": "线校准后比例不确定度偏高：n={n}, 相对不确定度={rel:.3f}%（目标 <= {threshold:.3f}%）。建议增加更长/更多校准线或使用矩形校准。",
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
