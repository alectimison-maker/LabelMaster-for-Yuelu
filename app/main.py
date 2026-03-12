#!/usr/bin/env python3
import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Avoid cv2-embedded Qt plugin path conflicts on some systems.
os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", "/usr/lib/x86_64-linux-gnu/qt5/plugins")
os.environ.setdefault("QT_PLUGIN_PATH", "/usr/lib/x86_64-linux-gnu/qt5/plugins")

from PyQt5.QtCore import QPoint, QPointF, QRect, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF, QTransform
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtSvg import QSvgRenderer

import cv2
import numpy as np

# cv2 may override Qt plugin path to its bundled directory; force system Qt plugins back.
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins"
os.environ["QT_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins"

try:
    from openvino import Core, Type, Layout, Tensor
    from openvino import preprocess as ov_pre

    HAVE_OPENVINO = True
except Exception:
    HAVE_OPENVINO = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = [
    "B1",
    "B2",
    "B3",
    "B4",
    "BS",
    "BO",
    "BG",
    "R1",
    "R2",
    "R3",
    "R4",
    "RS",
    "RO",
    "RG",
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: c for i, c in enumerate(CLASS_NAMES)}


@dataclass
class Annotation:
    cls: str
    pts: np.ndarray  # shape=(4,2), order: TL, BL, BR, TR
    score: float = 0.0


def copy_anns(src: List[Annotation]) -> List[Annotation]:
    return [Annotation(cls=a.cls, pts=a.pts.copy(), score=float(a.score)) for a in src]


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def clamp_pts_to_img(pts: np.ndarray, w: int, h: int) -> np.ndarray:
    out = pts.copy()
    out[:, 0] = np.clip(out[:, 0], 0.0, float(max(0, w - 1)))
    out[:, 1] = np.clip(out[:, 1], 0.0, float(max(0, h - 1)))
    return out


def bbox_from_pts(pts: np.ndarray) -> Tuple[float, float, float, float]:
    x1 = float(np.min(pts[:, 0]))
    y1 = float(np.min(pts[:, 1]))
    x2 = float(np.max(pts[:, 0]))
    y2 = float(np.max(pts[:, 1]))
    return x1, y1, x2, y2


def order_tl_bl_br_tr(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    idx = np.argsort(pts[:, 1])
    top = pts[idx[:2]]
    bot = pts[idx[2:]]
    tl = top[np.argmin(top[:, 0])]
    tr = top[np.argmax(top[:, 0])]
    bl = bot[np.argmin(bot[:, 0])]
    br = bot[np.argmax(bot[:, 0])]
    return np.array([tl, bl, br, tr], dtype=np.float32)


def keep_model_corner_order(pts: np.ndarray) -> np.ndarray:
    # 保留模型语义点序：1左上 2左下 3右下 4右上（auto_pre.cpp同源）
    return np.asarray(pts, dtype=np.float32).reshape(4, 2).copy()


def nms_indices(boxes: List[List[int]], confs: List[float], score_th: float, nms_th: float) -> List[int]:
    if not boxes:
        return []
    idx = cv2.dnn.NMSBoxes(boxes, confs, score_th, nms_th)
    if idx is None or len(idx) == 0:
        return []
    if isinstance(idx, np.ndarray):
        return [int(x) for x in idx.reshape(-1).tolist()]
    out = []
    for x in idx:
        if isinstance(x, (list, tuple, np.ndarray)):
            out.append(int(x[0]))
        else:
            out.append(int(x))
    return out


def map_auto_pre_cls(color_id: int, class_id: int) -> Optional[str]:
    # auto_pre.cpp映射思路：蓝=0 红=1；class 0..8
    if color_id not in (0, 1):
        return None
    team = "B" if color_id == 0 else "R"

    if class_id == 4:
        return None  # 5号过滤

    if class_id == 0:
        suffix = "1"
    elif class_id == 1:
        suffix = "2"
    elif class_id == 2:
        suffix = "3"
    elif class_id == 3:
        suffix = "4"
    elif class_id == 5:
        suffix = "O"
    elif class_id == 6:
        suffix = "G"
    elif class_id in (7, 8):
        suffix = "S"
    else:
        return None

    cls = f"{team}{suffix}"
    return cls if cls in CLASS_NAMES else None


def normalize_cls_to_name(token: str) -> str:
    s = token.strip().upper()
    if s in CLASS_TO_ID:
        return s
    if s.isdigit():
        idx = int(s)
        if idx in ID_TO_CLASS:
            return ID_TO_CLASS[idx]
    return "BG"


def normalize_cls_to_id(token: str) -> int:
    s = token.strip().upper()
    if s in CLASS_TO_ID:
        return CLASS_TO_ID[s]
    if s.isdigit():
        idx = int(s)
        if idx in ID_TO_CLASS:
            return idx
    return CLASS_TO_ID["BG"]


def rotate_image_for_detect(img: np.ndarray, mode: int) -> np.ndarray:
    # mode: 0 原图, 1 顺时针90, 2 逆时针90, 3 180
    if mode == 0:
        return img
    if mode == 1:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if mode == 2:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return cv2.rotate(img, cv2.ROTATE_180)


def inv_rotate_points_to_original(pts: np.ndarray, mode: int, w: int, h: int) -> np.ndarray:
    # pts基于旋转后图像坐标，映射回原图坐标
    if mode == 0:
        return pts.copy()

    out = pts.copy().astype(np.float32)
    if mode == 1:
        # cw90 inverse: x = y', y = h-1-x'
        x_new = out[:, 1]
        y_new = (h - 1) - out[:, 0]
    elif mode == 2:
        # ccw90 inverse: x = w-1-y', y = x'
        x_new = (w - 1) - out[:, 1]
        y_new = out[:, 0]
    else:
        # 180 inverse
        x_new = (w - 1) - out[:, 0]
        y_new = (h - 1) - out[:, 1]

    out[:, 0] = x_new
    out[:, 1] = y_new
    return out


class SmartArmorDetector:
    """
    优先使用 auto_pre.cpp 同款 OpenVINO(yolov5.xml/bin)流程。
    同时支持 model-opt.onnx 作为补充引擎，做旋转增强后并集NMS，提升旋转目标召回。
    """

    def __init__(self, ov_xml_path: str, onnx_path: str):
        self.ov_xml_path = ov_xml_path
        self.onnx_path = onnx_path

        self.input_w = 640
        self.input_h = 640
        self.score_thresh = 0.55
        self.nms_thresh = 0.45

        self.use_rot_aug = True
        self.rot_modes = [0, 1, 2]  # 原图 + 90度双向增强

        self.ov_request = None
        self.cv2_net = None
        self.engine_info = "none"

        self._load_models()

    def _load_models(self):
        ov_ok = False
        onnx_ok = False

        if HAVE_OPENVINO and self.ov_xml_path and os.path.isfile(self.ov_xml_path):
            try:
                core = Core()
                model = core.read_model(self.ov_xml_path)
                ppp = ov_pre.PrePostProcessor(model)
                ppp.input().tensor().set_element_type(Type.u8).set_shape([1, 640, 640, 3]).set_layout(
                    Layout("NHWC")
                ).set_color_format(ov_pre.ColorFormat.BGR)
                ppp.input().model().set_layout(Layout("NCHW"))
                ppp.input().preprocess().convert_element_type(Type.f32).convert_color(
                    ov_pre.ColorFormat.RGB
                ).scale(255.0)
                model = ppp.build()
                compiled = core.compile_model(model, "CPU")
                self.ov_request = compiled.create_infer_request()
                ov_ok = True
            except Exception:
                self.ov_request = None

        if self.onnx_path and os.path.isfile(self.onnx_path):
            try:
                self.cv2_net = cv2.dnn.readNetFromONNX(self.onnx_path)
                onnx_ok = True
            except Exception:
                self.cv2_net = None

        if ov_ok and onnx_ok:
            self.engine_info = "openvino_xml + cv2_onnx(ensemble)"
        elif ov_ok:
            self.engine_info = "openvino_xml"
        elif onnx_ok:
            self.engine_info = "cv2_onnx"
        else:
            self.engine_info = "none"

    def detect_mode1(self, bgr: np.ndarray) -> List[Annotation]:
        return self._detect_ensemble(bgr, forced_cls=None)

    def detect_boxes_only(self, bgr: np.ndarray, forced_cls: str) -> List[Annotation]:
        return self._detect_ensemble(bgr, forced_cls=forced_cls)

    def _detect_ensemble(self, bgr: np.ndarray, forced_cls: Optional[str]) -> List[Annotation]:
        if self.engine_info == "none":
            return []

        cands: List[Annotation] = []
        modes = self.rot_modes if self.use_rot_aug else [0]

        for rot_mode in modes:
            img_rot = rotate_image_for_detect(bgr, rot_mode)

            if self.ov_request is not None:
                cands.extend(self._detect_openvino_single(img_rot, rot_mode, bgr.shape[1], bgr.shape[0], forced_cls))

            if self.cv2_net is not None:
                cands.extend(self._detect_onnx_single(img_rot, rot_mode, bgr.shape[1], bgr.shape[0], forced_cls))

        if not cands:
            return []

        boxes: List[List[int]] = []
        confs: List[float] = []
        for a in cands:
            x1, y1, x2, y2 = bbox_from_pts(a.pts)
            boxes.append([int(x1), int(y1), int(max(1.0, x2 - x1)), int(max(1.0, y2 - y1))])
            confs.append(float(a.score))

        keep = nms_indices(boxes, confs, self.score_thresh, self.nms_thresh)
        out = [cands[i] for i in keep]
        return out

    def _detect_openvino_single(
        self, img: np.ndarray, rot_mode: int, orig_w: int, orig_h: int, forced_cls: Optional[str]
    ) -> List[Annotation]:
        h_img, w_img = img.shape[:2]
        scale = min(float(self.input_w) / float(h_img), float(self.input_w) / float(w_img))
        h_resize = int(round(h_img * scale))
        w_resize = int(round(w_img * scale))

        input_blob = np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8)
        resized = cv2.resize(img, (w_resize, h_resize), interpolation=cv2.INTER_LINEAR)
        input_blob[:h_resize, :w_resize] = resized

        self.ov_request.set_input_tensor(Tensor(input_blob[None, ...]))
        self.ov_request.infer()
        output = self.ov_request.get_output_tensor().data[0]

        out: List[Annotation] = []
        for r in output:
            conf = sigmoid(float(r[8]))
            if conf < self.score_thresh:
                continue

            color_id = int(np.argmax(r[9:13]))
            class_id = int(np.argmax(r[13:22]))

            cls = forced_cls if forced_cls else map_auto_pre_cls(color_id, class_id)
            if not cls:
                continue

            pts = np.array(
                [
                    [float(r[0]) / scale, float(r[1]) / scale],
                    [float(r[2]) / scale, float(r[3]) / scale],
                    [float(r[4]) / scale, float(r[5]) / scale],
                    [float(r[6]) / scale, float(r[7]) / scale],
                ],
                dtype=np.float32,
            )

            pts = inv_rotate_points_to_original(pts, rot_mode, orig_w, orig_h)
            pts = keep_model_corner_order(clamp_pts_to_img(pts, orig_w, orig_h))
            out.append(Annotation(cls=cls, pts=pts, score=float(conf)))

        return out

    def _detect_onnx_single(
        self, img: np.ndarray, rot_mode: int, orig_w: int, orig_h: int, forced_cls: Optional[str]
    ) -> List[Annotation]:
        h_img, w_img = img.shape[:2]
        scale = min(float(self.input_w) / float(h_img), float(self.input_w) / float(w_img))
        h_resize = int(round(h_img * scale))
        w_resize = int(round(w_img * scale))

        input_blob = np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8)
        resized = cv2.resize(img, (w_resize, h_resize), interpolation=cv2.INTER_LINEAR)
        input_blob[:h_resize, :w_resize] = resized

        rgb = cv2.cvtColor(input_blob, cv2.COLOR_BGR2RGB)
        blob = cv2.dnn.blobFromImage(rgb, scalefactor=1.0 / 255.0, size=(640, 640), swapRB=False, crop=False)
        self.cv2_net.setInput(blob)
        output = self.cv2_net.forward().reshape(-1, 22)

        out: List[Annotation] = []
        for r in output:
            conf = sigmoid(float(r[8]))
            if conf < self.score_thresh:
                continue

            color_id = int(np.argmax(r[9:13]))
            class_id = int(np.argmax(r[13:22]))
            cls = forced_cls if forced_cls else map_auto_pre_cls(color_id, class_id)
            if not cls:
                continue

            pts = np.array(
                [
                    [float(r[0]) / scale, float(r[1]) / scale],
                    [float(r[2]) / scale, float(r[3]) / scale],
                    [float(r[4]) / scale, float(r[5]) / scale],
                    [float(r[6]) / scale, float(r[7]) / scale],
                ],
                dtype=np.float32,
            )
            pts = inv_rotate_points_to_original(pts, rot_mode, orig_w, orig_h)
            pts = keep_model_corner_order(clamp_pts_to_img(pts, orig_w, orig_h))
            out.append(Annotation(cls=cls, pts=pts, score=float(conf)))

        return out


class ImageCanvas(QWidget):
    selectionChanged = pyqtSignal(int)
    annotationsChanged = pyqtSignal()
    STYLE_DEFAULT = 0
    STYLE_ICON_PERSPECTIVE = 1

    BIG_ANCHORS = (
        QPointF(0.0, 140.61),  # TL
        QPointF(0.0, 347.39),  # BL
        QPointF(871.0, 347.39),  # BR
        QPointF(871.0, 140.61),  # TR
    )
    SMALL_ANCHORS = (
        QPointF(0.0, 143.26),  # TL
        QPointF(0.0, 372.74),  # BL
        QPointF(557.0, 372.74),  # BR
        QPointF(557.0, 143.26),  # TR
    )

    def __init__(self, class_provider, icons_dir: str = ""):
        super().__init__()
        self.setMinimumSize(960, 640)
        self.class_provider = class_provider

        self.cv_img: Optional[np.ndarray] = None
        self.qimg: Optional[QImage] = None
        self.anns: List[Annotation] = []

        self.scale = 1.0
        self.offset = QPoint(0, 0)

        self.selected_idx = -1
        self.drag_handle_idx = -1
        self.draw_start: Optional[QPointF] = None
        self.draw_end: Optional[QPointF] = None
        self.box_style = self.STYLE_DEFAULT

        self.icons_dir = icons_dir
        self.svg_renderers: Dict[str, QSvgRenderer] = {}
        self.icon_style_ready = False
        self._load_svg_renderers()

    def _load_svg_renderers(self):
        self.svg_renderers.clear()
        self.icon_style_ready = False
        if not self.icons_dir or not os.path.isdir(self.icons_dir):
            return

        # 与 ATLabelMaster 同名图标
        keys = ["1", "2", "3", "4", "5", "B3", "B4", "B5", "O", "Gs", "Gb", "Bs", "Bb"]
        for key in keys:
            svg_path = os.path.join(self.icons_dir, f"{key}.svg")
            if not os.path.isfile(svg_path):
                continue
            renderer = QSvgRenderer(svg_path, self)
            if renderer.isValid():
                self.svg_renderers[key] = renderer

        self.icon_style_ready = len(self.svg_renderers) > 0

    def has_icon_style(self) -> bool:
        return self.icon_style_ready

    def set_box_style(self, style_idx: int):
        self.box_style = self.STYLE_ICON_PERSPECTIVE if style_idx == self.STYLE_ICON_PERSPECTIVE else self.STYLE_DEFAULT
        self.update()

    def _infer_big_by_shape(self, pts: np.ndarray) -> bool:
        # 用四点几何形状推断大小装甲（宽高比越大越偏向大装甲）
        top_w = float(np.linalg.norm(pts[3] - pts[0]))
        bot_w = float(np.linalg.norm(pts[2] - pts[1]))
        left_h = float(np.linalg.norm(pts[1] - pts[0]))
        right_h = float(np.linalg.norm(pts[2] - pts[3]))
        w = 0.5 * (top_w + bot_w)
        h = max(1e-6, 0.5 * (left_h + right_h))
        ratio = w / h
        return ratio >= 2.6

    def _pick_icon(self, ann: Annotation) -> Tuple[Optional[QSvgRenderer], bool]:
        if not self.icon_style_ready:
            return None, False

        cls = ann.cls.strip().upper()
        suffix = cls[1:] if len(cls) >= 2 else ""
        is_big = self._infer_big_by_shape(ann.pts)

        if suffix == "1":
            key, is_big = "1", True
        elif suffix == "2":
            key, is_big = "2", False
        elif suffix == "3":
            key = "B3" if is_big and "B3" in self.svg_renderers else "3"
        elif suffix == "4":
            key = "B4" if is_big and "B4" in self.svg_renderers else "4"
        elif suffix == "5":
            key = "B5" if is_big and "B5" in self.svg_renderers else "5"
        elif suffix == "O":
            key, is_big = "O", False
        elif suffix == "S":
            key = "Bb" if is_big and "Bb" in self.svg_renderers else "Bs"
        elif suffix == "G":
            key = "Gb" if is_big and "Gb" in self.svg_renderers else "Gs"
        else:
            key = "Gs"
            is_big = False

        renderer = self.svg_renderers.get(key)
        if renderer is None or not renderer.isValid():
            # 兜底：尽量保证可渲染
            renderer = self.svg_renderers.get("Gs") or self.svg_renderers.get("1")
            is_big = False
        return renderer, is_big

    def _draw_default_box(self, p: QPainter, qpts: List[QPoint], is_sel: bool):
        x1 = min(pt.x() for pt in qpts)
        y1 = min(pt.y() for pt in qpts)
        x2 = max(pt.x() for pt in qpts)
        y2 = max(pt.y() for pt in qpts)
        p.setPen(QPen(QColor(255, 220, 0), 2) if is_sel else QPen(QColor(0, 255, 0), 2))
        p.drawRect(QRect(QPoint(x1, y1), QPoint(x2, y2)))
        p.setPen(QPen(QColor(0, 200, 255), 1))
        for i in range(4):
            p.drawLine(qpts[i], qpts[(i + 1) % 4])

    def _draw_icon_perspective(self, p: QPainter, ann: Annotation, qpts: List[QPoint]) -> bool:
        renderer, is_big = self._pick_icon(ann)
        if renderer is None:
            return False

        # 复刻 ATLabelMaster 的透视流程：
        # 先把 SVG 坐标系映射到当前画布坐标系，再用“锚点(画布坐标) -> 标注四点”做单应变换。
        painter_quad = QPolygonF(
            [
                QPointF(0.0, 0.0),
                QPointF(0.0, float(self.height())),
                QPointF(float(self.width()), float(self.height())),
                QPointF(float(self.width()), 0.0),
            ]
        )
        if is_big:
            svg_quad = QPolygonF(
                [
                    QPointF(0.0, 0.0),
                    QPointF(0.0, 478.0),
                    QPointF(871.0, 478.0),
                    QPointF(871.0, 0.0),
                ]
            )
            anchors = QPolygonF(self.BIG_ANCHORS)
        else:
            svg_quad = QPolygonF(
                [
                    QPointF(0.0, 0.0),
                    QPointF(0.0, 516.0),
                    QPointF(557.0, 516.0),
                    QPointF(557.0, 0.0),
                ]
            )
            anchors = QPolygonF(self.SMALL_ANCHORS)

        svg2painter = QTransform()
        if not QTransform.quadToQuad(svg_quad, painter_quad, svg2painter):
            return False
        src = svg2painter.map(anchors)

        dst = QPolygonF([QPointF(float(pt.x()), float(pt.y())) for pt in qpts])
        H = QTransform()
        if not QTransform.quadToQuad(src, dst, H):
            return False

        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.setTransform(H, True)
        renderer.render(p)
        p.restore()
        return True

    def set_image(self, cv_img: np.ndarray):
        self.cv_img = cv_img
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self.qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.selected_idx = -1
        self.drag_handle_idx = -1
        self.draw_start = None
        self.draw_end = None
        self.selectionChanged.emit(-1)
        self.update()

    def set_annotations(self, anns: List[Annotation]):
        self.anns = anns
        self.selected_idx = -1 if not anns else min(self.selected_idx, len(anns) - 1)
        self.selectionChanged.emit(self.selected_idx)
        self.update()

    def get_annotations(self) -> List[Annotation]:
        return self.anns

    def set_selected_index(self, idx: int):
        if idx < -1 or idx >= len(self.anns):
            return
        self.selected_idx = idx
        self.selectionChanged.emit(self.selected_idx)
        self.update()

    def remove_by_index(self, idx: int):
        if 0 <= idx < len(self.anns):
            self.anns.pop(idx)
            self.selected_idx = min(idx, len(self.anns) - 1)
            self.selectionChanged.emit(self.selected_idx)
            self.annotationsChanged.emit()
            self.update()

    def remove_selected(self):
        if 0 <= self.selected_idx < len(self.anns):
            self.anns.pop(self.selected_idx)
            self.selected_idx = min(self.selected_idx, len(self.anns) - 1)
            self.selectionChanged.emit(self.selected_idx)
            self.annotationsChanged.emit()
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 20, 20))

        if self.qimg is None:
            p.setPen(QColor(220, 220, 220))
            p.drawText(self.rect(), Qt.AlignCenter, "打开图片文件夹后开始标注")
            return

        w, h = self.qimg.width(), self.qimg.height()
        rw, rh = self.width(), self.height()
        self.scale = min(rw / w, rh / h)
        dw, dh = int(w * self.scale), int(h * self.scale)
        ox, oy = (rw - dw) // 2, (rh - dh) // 2
        self.offset = QPoint(ox, oy)

        p.drawPixmap(QRect(ox, oy, dw, dh), QPixmap.fromImage(self.qimg))
        p.setRenderHint(QPainter.Antialiasing, True)
        font = QFont()
        font.setPointSize(10)
        p.setFont(font)

        for idx, ann in enumerate(self.anns):
            qpts = [QPoint(int(round(ox + x * self.scale)), int(round(oy + y * self.scale))) for x, y in ann.pts]
            x1 = min(pt.x() for pt in qpts)
            y1 = min(pt.y() for pt in qpts)
            x2 = max(pt.x() for pt in qpts)
            y2 = max(pt.y() for pt in qpts)
            is_sel = idx == self.selected_idx

            if self.box_style == self.STYLE_ICON_PERSPECTIVE:
                drawn = self._draw_icon_perspective(p, ann, qpts)
                if drawn:
                    # 图标样式下保留细边线，便于人工核对边界
                    p.setPen(QPen(QColor(255, 220, 0), 2) if is_sel else QPen(QColor(0, 255, 0), 1))
                    for i in range(4):
                        p.drawLine(qpts[i], qpts[(i + 1) % 4])
                else:
                    self._draw_default_box(p, qpts, is_sel)
            else:
                self._draw_default_box(p, qpts, is_sel)

            # 框编号 + 类别
            id_text = f"#{idx + 1} {ann.cls}"
            text_pos = QPoint(x1 + 4, max(14, y1 - 4))
            p.setPen(QPen(QColor(0, 0, 0), 3))
            p.drawText(text_pos, id_text)
            p.setPen(QPen(QColor(0, 255, 0), 1))
            p.drawText(text_pos, id_text)

            # 角点和点号
            for i, pt in enumerate(qpts):
                p.setPen(QPen(QColor(255, 0, 0), 2))
                p.setBrush(QColor(255, 0, 0))
                radius = 5 if is_sel else 4
                p.drawEllipse(pt, radius, radius)
                p.setPen(QPen(QColor(0, 0, 0), 3))
                p.drawText(pt + QPoint(6, -6), str(i + 1))
                p.setPen(QPen(QColor(255, 255, 255), 1))
                p.drawText(pt + QPoint(6, -6), str(i + 1))

        if self.draw_start is not None and self.draw_end is not None:
            a = self.img_to_screen(self.draw_start)
            b = self.img_to_screen(self.draw_end)
            p.setPen(QPen(QColor(255, 140, 0), 2, Qt.DashLine))
            p.drawRect(QRect(a, b).normalized())

    def screen_to_img(self, pt: QPoint) -> Optional[QPointF]:
        if self.qimg is None:
            return None
        x = (pt.x() - self.offset.x()) / self.scale
        y = (pt.y() - self.offset.y()) / self.scale
        if x < 0 or y < 0 or x >= self.qimg.width() or y >= self.qimg.height():
            return None
        return QPointF(float(x), float(y))

    def img_to_screen(self, pt: QPointF) -> QPoint:
        return QPoint(
            int(round(self.offset.x() + pt.x() * self.scale)),
            int(round(self.offset.y() + pt.y() * self.scale)),
        )

    def _hit_handle(self, img_pt: QPointF) -> int:
        if self.selected_idx < 0 or self.selected_idx >= len(self.anns):
            return -1
        ann = self.anns[self.selected_idx]
        px, py = float(img_pt.x()), float(img_pt.y())
        best, best_d = -1, 1e9
        for i, (x, y) in enumerate(ann.pts):
            d = (float(x) - px) ** 2 + (float(y) - py) ** 2
            if d < best_d:
                best, best_d = i, d
        th = (8.0 / max(self.scale, 1e-6)) ** 2
        return best if best_d <= th else -1

    def _hit_detection(self, img_pt: QPointF) -> int:
        p = QPointF(float(img_pt.x()), float(img_pt.y()))
        for i in range(len(self.anns) - 1, -1, -1):
            poly = QPolygonF([QPointF(float(x), float(y)) for x, y in self.anns[i].pts])
            if poly.containsPoint(p, Qt.OddEvenFill):
                return i
        return -1

    def mousePressEvent(self, e):
        if self.qimg is None:
            return
        img_pt = self.screen_to_img(e.pos())
        if img_pt is None:
            return

        if e.button() == Qt.RightButton:
            hit = self._hit_detection(img_pt)
            if hit >= 0:
                self.remove_by_index(hit)
            return

        if e.button() != Qt.LeftButton:
            return

        if self.selected_idx >= 0:
            h = self._hit_handle(img_pt)
            if h >= 0:
                self.drag_handle_idx = h
                return

        hit = self._hit_detection(img_pt)
        if hit >= 0:
            self.selected_idx = hit
            self.selectionChanged.emit(self.selected_idx)
            self.update()
            return

        self.draw_start = img_pt
        self.draw_end = img_pt
        self.drag_handle_idx = -1

    def mouseMoveEvent(self, e):
        img_pt = self.screen_to_img(e.pos())
        if img_pt is None or self.qimg is None:
            return

        if self.drag_handle_idx >= 0 and 0 <= self.selected_idx < len(self.anns):
            x = clamp01(img_pt.x() / self.qimg.width()) * (self.qimg.width() - 1)
            y = clamp01(img_pt.y() / self.qimg.height()) * (self.qimg.height() - 1)
            self.anns[self.selected_idx].pts[self.drag_handle_idx] = [x, y]
            self.update()
            return

        if self.draw_start is not None:
            self.draw_end = img_pt
            self.update()

    def mouseReleaseEvent(self, e):
        if self.qimg is None:
            return

        if e.button() == Qt.LeftButton and self.drag_handle_idx >= 0:
            self.drag_handle_idx = -1
            self.annotationsChanged.emit()
            self.update()
            return

        if e.button() != Qt.LeftButton or self.draw_start is None or self.draw_end is None:
            return

        x1, y1 = self.draw_start.x(), self.draw_start.y()
        x2, y2 = self.draw_end.x(), self.draw_end.y()
        self.draw_start = None
        self.draw_end = None

        if abs(x2 - x1) < 6 or abs(y2 - y1) < 6:
            self.update()
            return

        l, r = min(x1, x2), max(x1, x2)
        t, b = min(y1, y2), max(y1, y2)
        pts = np.array([[l, t], [l, b], [r, b], [r, t]], dtype=np.float32)
        pts = clamp_pts_to_img(pts, self.qimg.width(), self.qimg.height())

        self.anns.append(Annotation(cls=self.class_provider(), pts=pts, score=0.0))
        self.selected_idx = len(self.anns) - 1
        self.selectionChanged.emit(self.selected_idx)
        self.annotationsChanged.emit()
        self.update()


def resolve_save_root(image_path: str, explicit_save_root: str) -> str:
    if explicit_save_root:
        return explicit_save_root
    return os.path.dirname(image_path)


def label_paths_for(image_path: str, save_root: str) -> Tuple[str, str]:
    base = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
    d9 = os.path.join(save_root, "labels_9")
    d13 = os.path.join(save_root, "labels_13")
    return os.path.join(d9, base), os.path.join(d13, base)


def safe_remove(path: str):
    if os.path.isfile(path):
        os.remove(path)


def write_labels(image_path: str, anns: List[Annotation], save_root: str, save_mode: str):
    # save_mode: both | 9 | 13
    img = cv2.imread(image_path)
    if img is None:
        return
    h, w = img.shape[:2]

    os.makedirs(save_root, exist_ok=True)
    f9, f13 = label_paths_for(image_path, save_root)

    need_9 = save_mode in ("both", "9")
    need_13 = save_mode in ("both", "13")

    if need_9:
        os.makedirs(os.path.dirname(f9), exist_ok=True)
    if need_13:
        os.makedirs(os.path.dirname(f13), exist_ok=True)

    # 未选中的格式清理，避免旧文件残留
    if not need_9:
        safe_remove(f9)
    if not need_13:
        safe_remove(f13)

    if not anns:
        if need_9:
            open(f9, "w", encoding="utf-8").close()
        if need_13:
            open(f13, "w", encoding="utf-8").close()
        return

    file9 = open(f9, "w", encoding="utf-8") if need_9 else None
    file13 = open(f13, "w", encoding="utf-8") if need_13 else None

    try:
        for ann in anns:
            pts = ann.pts.astype(np.float64)
            q = np.zeros_like(pts)
            q[:, 0] = np.clip(pts[:, 0] / w, 0.0, 1.0)
            q[:, 1] = np.clip(pts[:, 1] / h, 0.0, 1.0)

            x1, y1, x2, y2 = bbox_from_pts(pts)
            cx = clamp01(((x1 + x2) * 0.5) / w)
            cy = clamp01(((y1 + y2) * 0.5) / h)
            bw = clamp01((x2 - x1) / w)
            bh = clamp01((y2 - y1) / h)

            p = [f"{v:.6f}" for v in q.reshape(-1)]
            cls_id = normalize_cls_to_id(ann.cls)
            if file9 is not None:
                file9.write(f"{cls_id} {' '.join(p)}\n")
            if file13 is not None:
                file13.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {' '.join(p)}\n")
    finally:
        if file9 is not None:
            file9.close()
        if file13 is not None:
            file13.close()


def read_labels_if_any(image_path: str) -> List[Annotation]:
    img = cv2.imread(image_path)
    if img is None:
        return []
    h, w = img.shape[:2]

    base = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
    root = os.path.dirname(image_path)
    p13 = os.path.join(root, "labels_13", base)
    p9 = os.path.join(root, "labels_9", base)

    path = p13 if os.path.isfile(p13) else p9 if os.path.isfile(p9) else ""
    if not path:
        return []

    anns: List[Annotation] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) not in (9, 13):
                continue
            cls = normalize_cls_to_name(parts[0])
            nums = [float(x) for x in parts[1:]]
            if len(parts) == 13:
                nums = nums[4:]
            if len(nums) != 8:
                continue
            pts = np.array(nums, dtype=np.float32).reshape(4, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            pts = keep_model_corner_order(clamp_pts_to_img(pts, w, h))
            anns.append(Annotation(cls=cls, pts=pts, score=0.0))
    return anns


class MainWindow(QMainWindow):
    def __init__(self, ov_xml_path: str, onnx_path: str):
        super().__init__()
        self.setWindowTitle("LabelMaster for Yuelu")
        self.resize(1600, 940)

        self.detector = SmartArmorDetector(ov_xml_path=ov_xml_path, onnx_path=onnx_path)
        self.image_dir = ""
        self.image_paths: List[str] = []
        self.current_idx = -1
        self.cache: Dict[str, List[Annotation]] = {}
        self.cv_current: Optional[np.ndarray] = None
        self.save_root = ""
        self.last_detect_backup: Dict[str, List[Annotation]] = {}
        self.filtered_indices: List[int] = []
        self._updating_box_selector = False

        self.btn_open = QPushButton("打开图片文件夹")
        self.btn_prev = QPushButton("上一张 (Q)")
        self.btn_next = QPushButton("下一张 (E)")

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["模式一：智能标注", "模式二：同兵种智能拉框"])
        self.class_combo = QComboBox()
        self.class_combo.addItems(CLASS_NAMES)

        self.btn_detect = QPushButton("智能标注当前 (Space)")
        self.btn_detect_all = QPushButton("智能标注全部(仅预标注)")
        self.btn_rollback = QPushButton("回退当前智能标注")

        self.save_mode_combo = QComboBox()
        self.save_mode_combo.addItems(["保存全部(9+13)", "仅9点", "仅13点"])
        self.btn_choose_save_root = QPushButton("选择保存路径")
        self.lbl_save_root = QLabel("保存路径: 使用图片目录")
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["筛选: 全部", "筛选: 无框", "筛选: 单框", "筛选: 多框"])
        self.style_combo = QComboBox()
        self.style_combo.addItems(["框样式: 默认线框", "框样式: 图标透视(icons)"])
        self.box_selector = QComboBox()
        self.box_selector.addItem("当前框: 无")
        self.btn_delete_box = QPushButton("删除选中框")

        self.btn_save = QPushButton("保存当前 (S)")
        self.btn_save_all = QPushButton("全部保存")

        top1 = QHBoxLayout()
        for w in [
            self.btn_open,
            self.btn_prev,
            self.btn_next,
            self.mode_combo,
            self.class_combo,
            self.btn_detect,
            self.btn_detect_all,
            self.btn_rollback,
        ]:
            top1.addWidget(w)

        top2 = QHBoxLayout()
        for w in [
            self.save_mode_combo,
            self.btn_choose_save_root,
            self.lbl_save_root,
            self.filter_combo,
            self.style_combo,
            self.box_selector,
            self.btn_delete_box,
            self.btn_save,
            self.btn_save_all,
        ]:
            top2.addWidget(w)

        self.image_list = QListWidget()
        self.image_list.setMinimumWidth(330)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        icons_dir = os.path.join(base_dir, "assets", "icons")
        self.canvas = ImageCanvas(class_provider=self.get_selected_class, icons_dir=icons_dir)

        splitter = QSplitter()
        splitter.addWidget(self.image_list)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.status = QLabel("就绪")

        root = QVBoxLayout()
        root.addLayout(top1)
        root.addLayout(top2)
        root.addWidget(splitter)
        root.addWidget(self.status)

        cw = QWidget()
        cw.setLayout(root)
        self.setCentralWidget(cw)

        self.btn_open.clicked.connect(self.open_dir)
        self.btn_prev.clicked.connect(self.prev_img)
        self.btn_next.clicked.connect(self.next_img)
        self.btn_detect.clicked.connect(self.detect_current)
        self.btn_detect_all.clicked.connect(self.detect_all_preview)
        self.btn_rollback.clicked.connect(self.rollback_current_detect)
        self.btn_choose_save_root.clicked.connect(self.choose_save_root)
        self.btn_save.clicked.connect(self.save_current)
        self.btn_save_all.clicked.connect(self.save_all)
        self.btn_delete_box.clicked.connect(self.delete_selected_box)

        self.image_list.currentRowChanged.connect(self.on_select_idx)
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        self.filter_combo.currentIndexChanged.connect(self.on_filter_changed)
        self.style_combo.currentIndexChanged.connect(self.on_style_changed)
        self.box_selector.currentIndexChanged.connect(self.on_box_selector_changed)
        self.canvas.selectionChanged.connect(self.on_canvas_selection_changed)
        self.canvas.annotationsChanged.connect(self.on_canvas_annotations_changed)
        self.on_mode_changed()
        self.on_style_changed()

        self.status.setText(
            f"检测引擎: {self.detector.engine_info} | 旋转增强: {'开' if self.detector.use_rot_aug else '关'}"
        )

    def keyPressEvent(self, e):
        key = e.key()
        if key == Qt.Key_Q:
            self.prev_img()
            return
        if key == Qt.Key_E:
            self.next_img()
            return
        if key == Qt.Key_S:
            self.save_current()
            return
        if key == Qt.Key_Space:
            self.detect_current()
            return
        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            self.canvas.remove_selected()
            self.status.setText("已删除当前选中目标（需保存后生效）")
            return
        super().keyPressEvent(e)

    def get_selected_class(self) -> str:
        return self.class_combo.currentText()

    def get_save_mode(self) -> str:
        idx = self.save_mode_combo.currentIndex()
        if idx == 1:
            return "9"
        if idx == 2:
            return "13"
        return "both"

    def _ann_count_for_path(self, path: str) -> int:
        anns = self.cache.get(path)
        if anns is None:
            return 0
        return len(anns)

    def _accept_by_filter(self, count: int) -> bool:
        idx = self.filter_combo.currentIndex()
        if idx == 1:
            return count == 0
        if idx == 2:
            return count == 1
        if idx == 3:
            return count >= 2
        return True

    def refresh_image_list(self, keep_path: Optional[str] = None):
        if keep_path is None:
            keep_path = self._current_path()

        self.filtered_indices = []
        self.image_list.blockSignals(True)
        self.image_list.clear()
        for i, p in enumerate(self.image_paths):
            c = self._ann_count_for_path(p)
            if not self._accept_by_filter(c):
                continue
            self.filtered_indices.append(i)
            self.image_list.addItem(f"[{c}] {os.path.basename(p)}")
        self.image_list.blockSignals(False)

        if not self.filtered_indices:
            self.current_idx = -1
            self.canvas.set_annotations([])
            self.box_selector.clear()
            self.box_selector.addItem("当前框: 无")
            self.status.setText("当前筛选下没有图片。")
            return

        target_global = self.filtered_indices[0]
        if keep_path and keep_path in self.image_paths:
            ki = self.image_paths.index(keep_path)
            if ki in self.filtered_indices:
                target_global = ki

        row = self.filtered_indices.index(target_global)
        self.image_list.setCurrentRow(row)

    def update_current_list_item_text(self):
        row = self.image_list.currentRow()
        if row < 0 or row >= len(self.filtered_indices):
            return
        gidx = self.filtered_indices[row]
        if gidx < 0 or gidx >= len(self.image_paths):
            return
        p = self.image_paths[gidx]
        c = self._ann_count_for_path(p)
        item = self.image_list.item(row)
        if item is not None:
            item.setText(f"[{c}] {os.path.basename(p)}")

    def refresh_box_selector(self):
        anns = self.canvas.get_annotations()
        sel = self.canvas.selected_idx
        self._updating_box_selector = True
        self.box_selector.blockSignals(True)
        self.box_selector.clear()
        if not anns:
            self.box_selector.addItem("当前框: 无")
            self.box_selector.setCurrentIndex(0)
        else:
            for i, a in enumerate(anns):
                self.box_selector.addItem(f"#{i + 1} {a.cls}")
            self.box_selector.setCurrentIndex(max(0, sel))
        self.box_selector.blockSignals(False)
        self._updating_box_selector = False

    def on_box_selector_changed(self, row: int):
        if self._updating_box_selector:
            return
        anns = self.canvas.get_annotations()
        if not anns:
            return
        idx = max(0, min(row, len(anns) - 1))
        self.canvas.set_selected_index(idx)

    def on_canvas_selection_changed(self, idx: int):
        anns = self.canvas.get_annotations()
        if not anns:
            return
        idx = max(0, min(idx, len(anns) - 1))
        self._updating_box_selector = True
        self.box_selector.blockSignals(True)
        self.box_selector.setCurrentIndex(idx)
        self.box_selector.blockSignals(False)
        self._updating_box_selector = False

    def on_canvas_annotations_changed(self):
        path = self._current_path()
        if path:
            self.cache[path] = copy_anns(self.canvas.get_annotations())
        self.refresh_box_selector()
        self.update_current_list_item_text()

    def on_filter_changed(self):
        self.refresh_image_list()

    def delete_selected_box(self):
        anns = self.canvas.get_annotations()
        if not anns:
            return
        idx = self.box_selector.currentIndex()
        if idx < 0:
            idx = self.canvas.selected_idx
        if idx < 0:
            idx = 0
        self.canvas.remove_by_index(idx)
        self.status.setText("已删除指定误判框（需保存后落盘）")

    def choose_save_root(self):
        d = QFileDialog.getExistingDirectory(self, "选择标注保存目录", self.save_root or self.image_dir or os.path.expanduser("~"))
        if not d:
            return
        self.save_root = d
        self.lbl_save_root.setText(f"保存路径: {d}")

    def on_mode_changed(self):
        is_mode2 = self.mode_combo.currentIndex() == 1
        self.class_combo.setEnabled(is_mode2)

    def on_style_changed(self):
        idx = self.style_combo.currentIndex()
        if idx == ImageCanvas.STYLE_ICON_PERSPECTIVE and not self.canvas.has_icon_style():
            QMessageBox.warning(self, "提示", "图标样式资源未加载，已切回默认线框。")
            self.style_combo.blockSignals(True)
            self.style_combo.setCurrentIndex(ImageCanvas.STYLE_DEFAULT)
            self.style_combo.blockSignals(False)
            idx = ImageCanvas.STYLE_DEFAULT
        self.canvas.set_box_style(idx)

    def _stash_current_canvas(self):
        path = self._current_path()
        if path:
            self.cache[path] = copy_anns(self.canvas.get_annotations())

    def open_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择图片文件夹", self.image_dir or os.path.expanduser("~"))
        if not d:
            return
        files = []
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if os.path.isfile(p) and os.path.splitext(name.lower())[1] in IMAGE_EXTS:
                files.append(p)
        if not files:
            QMessageBox.warning(self, "提示", "该目录没有图片文件。")
            return

        self.image_dir = d
        self.image_paths = files
        self.current_idx = 0
        self.cache.clear()
        self.last_detect_backup.clear()
        for p in self.image_paths:
            self.cache[p] = read_labels_if_any(p)
        self.refresh_image_list(keep_path=self.image_paths[0])

    def on_select_idx(self, idx: int):
        if idx < 0 or idx >= len(self.filtered_indices):
            return
        next_global_idx = self.filtered_indices[idx]
        if self.current_idx != -1 and self.current_idx != next_global_idx:
            self._stash_current_canvas()
        self.current_idx = next_global_idx
        self.load_current()

    def load_current(self):
        path = self._current_path()
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            self.status.setText(f"读取失败: {path}")
            return

        self.cv_current = img
        self.canvas.set_image(img)

        anns = self.cache.get(path)
        if anns is None:
            anns = read_labels_if_any(path)
            self.cache[path] = copy_anns(anns)
        self.canvas.set_annotations(copy_anns(anns))
        self.refresh_box_selector()

        self.status.setText(
            f"[{self.current_idx + 1}/{len(self.image_paths)}] {os.path.basename(path)} | 当前标注: {len(anns)}"
        )

    def prev_img(self):
        row = self.image_list.currentRow()
        if row > 0:
            self.image_list.setCurrentRow(row - 1)

    def next_img(self):
        row = self.image_list.currentRow()
        if row + 1 < self.image_list.count():
            self.image_list.setCurrentRow(row + 1)

    def _current_path(self) -> Optional[str]:
        if 0 <= self.current_idx < len(self.image_paths):
            return self.image_paths[self.current_idx]
        return None

    def detect_current(self):
        path = self._current_path()
        if not path or self.cv_current is None:
            return

        before = self.cache.get(path, [])
        self.last_detect_backup[path] = copy_anns(before)

        if self.mode_combo.currentIndex() == 0:
            anns = self.detector.detect_mode1(self.cv_current)
        else:
            anns = self.detector.detect_boxes_only(self.cv_current, self.get_selected_class())

        self.cache[path] = copy_anns(anns)
        self.canvas.set_annotations(copy_anns(anns))
        self.refresh_box_selector()
        self.update_current_list_item_text()

        # 单张智能标注保持自动保存，支持回退恢复文件
        save_root = resolve_save_root(path, self.save_root)
        write_labels(path, anns, save_root, self.get_save_mode())

        self.status.setText(f"智能标注完成: {os.path.basename(path)} | {len(anns)} 个目标（已保存，可回退）")

    def rollback_current_detect(self):
        path = self._current_path()
        if not path:
            return
        if path not in self.last_detect_backup:
            self.status.setText("当前图片没有可回退的智能标注记录")
            return

        anns = copy_anns(self.last_detect_backup[path])
        self.cache[path] = copy_anns(anns)
        self.canvas.set_annotations(copy_anns(anns))
        self.refresh_box_selector()
        self.update_current_list_item_text()

        save_root = resolve_save_root(path, self.save_root)
        write_labels(path, anns, save_root, self.get_save_mode())

        self.status.setText(f"已回退当前智能标注: {os.path.basename(path)} | {len(anns)} 个目标")

    def detect_all_preview(self):
        if not self.image_paths:
            return

        self._stash_current_canvas()

        pd = QProgressDialog("正在智能标注全部图片（仅预标注，不保存）...", "取消", 0, len(self.image_paths), self)
        pd.setWindowTitle("批量预标注")
        pd.setWindowModality(Qt.WindowModal)
        pd.show()

        mode = self.mode_combo.currentIndex()
        forced = self.get_selected_class()
        count_obj = 0
        done = 0

        for i, path in enumerate(self.image_paths, start=1):
            QApplication.processEvents()
            if pd.wasCanceled():
                break

            img = cv2.imread(path)
            if img is None:
                pd.setValue(i)
                continue

            prev = self.cache.get(path, [])
            self.last_detect_backup[path] = copy_anns(prev)

            anns = self.detector.detect_mode1(img) if mode == 0 else self.detector.detect_boxes_only(img, forced)
            self.cache[path] = copy_anns(anns)

            count_obj += len(anns)
            done += 1
            pd.setValue(i)

        pd.close()
        self.load_current()
        self.refresh_image_list(keep_path=self._current_path())
        self.status.setText(
            f"批量预标注完成: {done} 张, 总目标 {count_obj}。请人工筛选后点击“全部保存”再落盘。"
        )

    def save_current(self):
        path = self._current_path()
        if not path:
            return

        anns = self.canvas.get_annotations()
        self.cache[path] = copy_anns(anns)
        self.update_current_list_item_text()

        save_root = resolve_save_root(path, self.save_root)
        write_labels(path, anns, save_root, self.get_save_mode())

        self.status.setText(f"已保存当前: {os.path.basename(path)} | {len(anns)} 个")

    def save_all(self):
        if not self.image_paths:
            return

        self._stash_current_canvas()

        total = 0
        for path in self.image_paths:
            anns = self.cache.get(path, [])
            save_root = resolve_save_root(path, self.save_root)
            write_labels(path, anns, save_root, self.get_save_mode())
            total += len(anns)

        out_dir_show = self.save_root if self.save_root else "图片目录(labels_9/labels_13)"
        self.status.setText(f"全部保存完成，总目标数: {total} | 输出目录: {out_dir_show}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ov-xml", default="", help="yolov5.xml 路径（OpenVINO）")
    parser.add_argument("--model", default="", help="model-opt.onnx 路径（补充引擎）")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_ov = os.path.join(base_dir, "assets", "models", "yolov5.xml")
    if not os.path.isfile(default_ov):
        alt = "/home/aliouswe/下载/yolov5_test_develop/build/yolov5.xml"
        default_ov = alt if os.path.isfile(alt) else ""

    default_onnx = os.path.join(base_dir, "assets", "models", "model-opt.onnx")

    ov_xml_path = args.ov_xml or default_ov
    onnx_path = args.model or default_onnx

    app = QApplication(sys.argv)
    w = MainWindow(ov_xml_path=ov_xml_path, onnx_path=onnx_path)
    w.show()
    sys.exit(app.exec_())
