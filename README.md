# Diemeter

面向半导体芯片照片及类似平面对象的高精度摄影测量面积估算工具。

## 功能特性

- **三种标定方法**
  - **直线标定** — 在已知长度的参考线上绘制，使用 IRLS 鲁棒拟合并进行亚像素端点不确定度估计
  - **矩形标定** — 标记已知尺寸的矩形，采用蒙特卡洛方法传播 PPU 不确定度
  - **焊盘间距标定** — 对焊盘行进行 FFT 自相关分析，输入已知间距即可自动完成标定
- **亚像素边缘检测** — Canny + Devernay 算法，结合 KD 树实现边缘吸附（snap-to-edge）
- **多边形测量** — 交互式顶点绘制，输出面积与周长，并进行完整的解析不确定度传播
- **最小面积外接框** — 输出有向外接矩形（OBB）及蒙特卡洛不确定度
- **栅格检测** — 二维 FFT 自相关，用于检测焊盘阵列、像素阵列等重复图案的间距
- **不确定度分解** — 顶点位置贡献与标定（PPU）贡献分别列出，按正交方式合并
- **会话持久化** — 以 JSON 格式保存 / 加载完整会话
- **结果导出** — 支持 CSV 与 JSON，包含全部数值细节
- **无头引擎** — `engine/controller.py` 提供不依赖 Qt 的完整功能，适用于脚本与测试
- **多语言支持** — 可插拔翻译层（`gui/i18n.py`）

## 依赖要求

| 依赖 | 版本 |
|---|---|
| Python | ≥ 3.9 |
| PySide6 | ≥ 6.6 |
| OpenCV（`opencv-contrib-python`） | ≥ 4.13 |
| NumPy | ≥ 2.0 |
| SciPy | ≥ 1.13 |

## 安装

```bash
# 克隆仓库
git clone https://github.com/ChaNg1o1/diemeter.git
cd diemeter

# 可编辑模式安装
pip install -e .

# 包含开发依赖（pytest、coverage）
pip install -e ".[dev]"
```

## 使用

```bash
diemeter <图片路径>                          # 在 GUI 中打开图片
diemeter <图片路径> --session session.json   # 恢复已保存的会话
diemeter <图片路径> --edge-snap              # 启动时开启边缘吸附
diemeter <图片路径> --snap-radius 20         # 自定义吸附半径（像素）
```

### 操作流程

1. **打开图片** — 通过命令行传入图片路径
2. **标定** — 在工具栏选择三种标定方法之一
3. **绘制多边形** — 依次点击芯片区域顶点，双击关闭
4. **测量** — 面积、周长和外接框连同 ± 不确定度显示在信息面板
5. **导出** — 通过 *文件 → 导出* 保存为 CSV / JSON
6. **会话** — *文件 → 保存会话* / *加载会话* 随时恢复进度

### 快捷键

| 按键 | 功能 |
|---|---|
| `Escape` | 取消当前操作 / 返回查看模式 |
| `Ctrl+Z` | 撤销上一个顶点 |
| `Ctrl+S` | 保存会话 |
| `Ctrl+E` | 导出结果 |

## 项目结构

```
src/diemeter/
├── cli.py              命令行入口
├── model.py            领域类型（Unit、Calibration、Polygon、Session 等）
├── calibration.py      PPU 估算（直线 / 矩形 / 焊盘间距）
├── edge.py             Canny + Devernay 亚像素边缘，KD 树吸附
├── grid.py             FFT 自相关栅格 / 间距检测
├── measurement.py      面积、周长、外接框及不确定度
├── uncertainty.py      解析与蒙特卡洛不确定度传播
├── session.py          JSON 序列化 / 反序列化
├── export.py           CSV / JSON 导出
├── engine/
│   └── controller.py   无头控制器（不依赖 Qt）
└── gui/
    ├── app.py          主窗口（QMainWindow）
    ├── canvas.py       可缩放图像画布
    ├── handlers.py     鼠标 / 键盘事件处理
    ├── modes.py        交互模式状态机
    ├── toolbar.py      工具栏动作
    ├── info_panel.py   可停靠结果面板
    ├── items.py        图形叠加层元素
    └── i18n.py         翻译辅助工具
```

## 开发

```bash
# 运行所有测试
pytest

# 带覆盖率报告
pytest --cov=src/diemeter --cov-report=term-missing
```

## 许可证

MIT

