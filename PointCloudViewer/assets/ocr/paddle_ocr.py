#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PaddleOCR helper for PointCloudViewer.

CLI:  python paddle_ocr.py <image> <output.json> [lang] [use_cls]
Server: python paddle_ocr.py --server
"""

from __future__ import annotations

import base64
import inspect
import json
import os
import sys

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

DEFAULT_OCR_VERSION = "PP-OCRv4"
DEFAULT_REC_MODEL = "PP-OCRv4_mobile_rec"
MAX_REC_WIDTH = 640

_OCR_CACHE: dict[tuple[str, bool], object] = {}
_TEXT_REC = None


def write_result(path: str, ok: bool, full_text: str, words: list, error: str) -> None:
    payload = {"ok": ok, "fullText": full_text, "words": words, "error": error}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def result_payload(ok: bool, full_text: str, words: list, error: str) -> dict:
    return {"ok": ok, "fullText": full_text, "words": words, "error": error}


def box_from_poly(poly) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def box_from_rec(rec_box) -> tuple[float, float, float, float]:
    arr = list(rec_box)
    if len(arr) >= 4:
        x0, y0, x1, y1 = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
    return 0.0, 0.0, 0.0, 0.0


def make_word(text: str, conf: float, x0: float, y0: float, x1: float, y1: float) -> dict:
    return {
        "text": text,
        "confidence": conf * 100.0 if conf <= 1.0 else conf,
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
    }


def create_ocr(lang: str, use_cls: bool):
    from paddleocr import PaddleOCR

    params = inspect.signature(PaddleOCR.__init__).parameters
    kwargs: dict = {
        "lang": lang,
        "enable_mkldnn": False,
    }

    if "ocr_version" in params:
        kwargs["ocr_version"] = DEFAULT_OCR_VERSION
    if "use_doc_orientation_classify" in params:
        kwargs["use_doc_orientation_classify"] = False
    if "use_doc_unwarping" in params:
        kwargs["use_doc_unwarping"] = False
    if "use_textline_orientation" in params:
        kwargs["use_textline_orientation"] = use_cls
    elif "use_angle_cls" in params:
        kwargs["use_angle_cls"] = use_cls

    return PaddleOCR(**kwargs)


def get_ocr(lang: str, use_cls: bool):
    key = (lang, use_cls)
    cached = _OCR_CACHE.get(key)
    if cached is not None:
        return cached
    ocr = create_ocr(lang, use_cls)
    _OCR_CACHE[key] = ocr
    return ocr


def get_text_rec():
    global _TEXT_REC
    if _TEXT_REC is not None:
        return _TEXT_REC
    from paddleocr import TextRecognition

    _TEXT_REC = TextRecognition(model_name=DEFAULT_REC_MODEL, enable_mkldnn=False)
    return _TEXT_REC


def load_bgr(img_path: str = "", image_b64: str | None = None):
    import cv2
    import numpy as np

    if image_b64:
        buf = base64.b64decode(image_b64)
        return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if img_path:
        return cv2.imread(img_path)
    return None


def maybe_downscale(bgr, max_width: int = MAX_REC_WIDTH):
    import cv2

    h, w = bgr.shape[:2]
    if w <= max_width:
        return bgr
    scale = max_width / float(w)
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(bgr, (max_width, new_h), interpolation=cv2.INTER_AREA)


def run_inference(ocr, image, use_cls: bool):
    if hasattr(ocr, "predict"):
        return list(ocr.predict(image))
    return ocr.ocr(image, cls=use_cls)


def _as_list(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def parse_v3_result(item) -> list[dict]:
    words: list[dict] = []
    rec_texts = _as_list(item.get("rec_texts"))
    rec_scores = _as_list(item.get("rec_scores"))
    rec_polys = item.get("rec_polys")
    if rec_polys is None:
        rec_polys = item.get("dt_polys")
    rec_polys = _as_list(rec_polys)
    rec_boxes = _as_list(item.get("rec_boxes"))

    for i, text in enumerate(rec_texts):
        if not text:
            continue
        conf = float(rec_scores[i]) if i < len(rec_scores) else 0.0
        poly = rec_polys[i] if i < len(rec_polys) else None
        box = rec_boxes[i] if i < len(rec_boxes) else None
        if poly is not None and len(poly) >= 4:
            x0, y0, x1, y1 = box_from_poly(poly)
        elif box is not None and len(box) >= 4:
            x0, y0, x1, y1 = box_from_rec(box)
        else:
            x0 = y0 = x1 = y1 = 0.0
        words.append(make_word(str(text), conf, x0, y0, x1, y1))
    return words


def parse_v2_result(lines) -> list[dict]:
    words: list[dict] = []
    if not lines:
        return words
    for line in lines:
        if not line or len(line) < 2:
            continue
        box, rec = line[0], line[1]
        text = rec[0] if isinstance(rec, (list, tuple)) else str(rec)
        conf = float(rec[1]) if isinstance(rec, (list, tuple)) and len(rec) > 1 else 0.0
        x0, y0, x1, y1 = box_from_poly(box)
        words.append(make_word(text, conf, x0, y0, x1, y1))
    return words


def parse_rec_only_result(raw, img_w: int, img_h: int) -> list[dict]:
    words: list[dict] = []
    for item in raw:
        if hasattr(item, "get"):
            text = item.get("rec_text", "")
            conf = float(item.get("rec_score", 0.0))
        else:
            continue
        if not text:
            continue
        words.append(make_word(str(text), conf, 0.0, 0.0, float(img_w), float(img_h)))
    return words


def parse_results(raw) -> list[dict]:
    if not raw:
        return []
    first = raw[0]
    if hasattr(first, "get"):
        keys = first.keys() if hasattr(first, "keys") else []
        if "rec_texts" in keys or "dt_polys" in keys:
            return parse_v3_result(first)
    if isinstance(first, list):
        return parse_v2_result(first)
    return []


def recognize_image(
    img_path: str = "",
    lang: str = "ch",
    use_cls: bool = False,
    rec_only: bool = False,
    image_b64: str | None = None,
) -> dict:
    try:
        import cv2  # noqa: F401
        from paddleocr import PaddleOCR  # noqa: F401
    except ImportError as exc:
        return result_payload(False, "", [], f"未安装 paddleocr: {exc}")

    try:
        bgr = load_bgr(img_path, image_b64)
        if bgr is None:
            return result_payload(False, "", [], "无法读取图像")

        bgr = maybe_downscale(bgr)
        img_h, img_w = bgr.shape[:2]

        if rec_only:
            rec = get_text_rec()
            raw = list(rec.predict(bgr))
            words = parse_rec_only_result(raw, img_w, img_h)
        else:
            ocr = get_ocr(lang, use_cls)
            raw = run_inference(ocr, bgr, use_cls)
            words = parse_results(raw)

        texts = [w["text"] for w in words]
        return result_payload(True, "".join(texts), words, "")
    except Exception as exc:  # noqa: BLE001
        return result_payload(False, "", [], str(exc))


def run_ocr(img_path: str, out_path: str, lang: str = "ch", use_cls: bool = True) -> int:
    payload = recognize_image(img_path=img_path, lang=lang, use_cls=use_cls)
    write_result(out_path, payload["ok"], payload["fullText"], payload["words"], payload["error"])
    return 0 if payload["ok"] else 1


def _silence_logs() -> None:
    os.environ["GLOG_minloglevel"] = "3"
    os.environ["FLAGS_minloglevel"] = "2"
    try:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    except OSError:
        pass


def run_server() -> int:
    _silence_logs()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stdin.reconfigure(encoding="utf-8")

    try:
        get_text_rec()
        get_ocr("ch", False)
        print(json.dumps({"ready": True}, ensure_ascii=False), flush=True)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ready": False, "error": str(exc)}, ensure_ascii=False), flush=True)
        return 1

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line.lower() in ("quit", "exit", "shutdown"):
            break
        try:
            req = json.loads(line)
            img = req.get("image", "")
            image_b64 = req.get("image_b64")
            lang = req.get("lang", "ch")
            use_cls = bool(req.get("use_cls", False))
            rec_only = bool(req.get("rec_only", False))
            payload = recognize_image(
                img_path=img,
                lang=lang,
                use_cls=use_cls,
                rec_only=rec_only,
                image_b64=image_b64,
            )
            print(json.dumps(payload, ensure_ascii=False), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps(result_payload(False, "", [], str(exc)), ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--server":
        return run_server()

    if len(sys.argv) < 3:
        print("usage: paddle_ocr.py <image> <output.json> [lang] [use_cls]", file=sys.stderr)
        print("       paddle_ocr.py --server", file=sys.stderr)
        return 1
    img = sys.argv[1]
    out = sys.argv[2]
    lang = sys.argv[3] if len(sys.argv) > 3 else "ch"
    use_cls = True
    if len(sys.argv) > 4:
        use_cls = sys.argv[4].lower() not in ("0", "false", "no")
    return run_ocr(img, out, lang, use_cls)


if __name__ == "__main__":
    sys.exit(main())
