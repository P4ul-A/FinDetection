"""Run a local Finwave-compatible fin detection server.

The remote `/fin-detect` endpoint returns base64-encoded cropped fin images plus
normalized and absolute bounding boxes. This local server mirrors that response
shape while using a locally trained Ultralytics YOLO model.

Example:
    python3 scripts/spawn_findetection_server.py

Then point scripts/fin_finder.py at:
    base_url = "http://127.0.0.1:8000/api/inference"
"""

from __future__ import annotations

import argparse
import base64
import logging
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps
from ultralytics import YOLO


DEFAULT_MODEL_PATH = Path("deployment_model_risso.pt")
LOGGER = logging.getLogger("findetection_server")


class FinDetectionService:
    def __init__(
        self,
        model_path: Path,
        image_size: int,
        confidence: float,
        iou: float,
        device: str | None,
        max_detections: int,
        crop_padding: float,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {model_path}. "
                "Finish training first or pass --model-path."
            )

        self.model = YOLO(str(model_path))
        self.image_size = image_size
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.max_detections = max_detections
        self.crop_padding = crop_padding

    def detect(self, image: Image.Image) -> dict[str, Any]:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        result = self.model.predict(
            source=image,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            max_det=self.max_detections,
            verbose=False,
        )[0]

        detections: list[dict[str, Any]] = []
        if result.boxes is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().tolist()
            boxes_xywhn = result.boxes.xywhn.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()

            for xyxy, xywhn, score, class_id in zip(
                boxes_xyxy, boxes_xywhn, confidences, classes
            ):
                x1, y1, x2, y2 = clamp_xyxy(xyxy, width, height, self.crop_padding)
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append(
                    {
                        "xyxy": (x1, y1, x2, y2),
                        "xywhn": xywhn,
                        "score": float(score),
                        "class_id": int(class_id),
                    }
                )

        cropped_images: list[str] = []
        proportion_boxes: list[dict[str, Any]] = []
        absolute_boxes: list[dict[str, Any]] = []

        for detection in detections:
            x1, y1, x2, y2 = detection["xyxy"]
            center_x, center_y, box_w_norm, box_h_norm = detection["xywhn"]
            crop = image.crop((round(x1), round(y1), round(x2), round(y2)))
            cropped_images.append(image_to_base64_jpeg(crop))

            box_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            absolute_w = x2 - x1
            absolute_h = y2 - y1

            proportion_boxes.append(
                {
                    "x": round(float(center_x), 6),
                    "y": round(float(center_y), 6),
                    "w": round(float(box_w_norm), 6),
                    "h": round(float(box_h_norm), 6),
                    "id": box_id,
                    "insertDateTime": timestamp,
                    "confidence": round(detection["score"], 6),
                    "classId": detection["class_id"],
                }
            )
            absolute_boxes.append(
                {
                    "x": round(float(x1), 4),
                    "y": round(float(y1), 4),
                    "w": round(float(absolute_w), 4),
                    "h": round(float(absolute_h), 4),
                    "id": box_id,
                    "insertDateTime": timestamp,
                    "confidence": round(detection["score"], 6),
                    "classId": detection["class_id"],
                }
            )

        return {
            "response": {
                "sourceId": None,
                "croppedImages": cropped_images,
                "extractedImages": cropped_images,
                "proportionBoxes": proportion_boxes,
                "absoluteBoxes": absolute_boxes,
            }
        }


def clamp_xyxy(
    xyxy: list[float],
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, xyxy)
    if padding_ratio > 0:
        pad_x = (x2 - x1) * padding_ratio
        pad_y = (y2 - y1) * padding_ratio
        x1 -= pad_x
        y1 -= pad_y
        x2 += pad_x
        y2 += pad_y

    return (
        max(0.0, min(x1, float(image_width))),
        max(0.0, min(y1, float(image_height))),
        max(0.0, min(x2, float(image_width))),
        max(0.0, min(y2, float(image_height))),
    )


def image_to_base64_jpeg(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def read_uploaded_image(file: UploadFile) -> Image.Image:
    try:
        image = Image.open(file.file)
        image.load()
        return image
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.") from exc


def create_app(service: FinDetectionService) -> FastAPI:
    app = FastAPI(title="Local FinDetection API", version="1.0")

    @app.get("/")
    def root() -> dict[str, Any]:
        return {"status": "ok", "service": "findetection"}

    @app.get("/api/inference")
    def inference_root() -> dict[str, Any]:
        return {"status": "ok", "service": "findetection"}

    @app.post("/api/inference/fin-detect")
    def fin_detect(file: UploadFile = File(...)) -> dict[str, Any]:
        image = read_uploaded_image(file)
        return service.detect(image)

    @app.post("/api/inference/vvi-detect")
    def vvi_detect(file: UploadFile = File(...)) -> dict[str, Any]:
        read_uploaded_image(file)
        return {"response": {"class": "valid"}}

    @app.post("/api/inference/fin-identify")
    def fin_identify() -> dict[str, Any]:
        raise HTTPException(
            status_code=501,
            detail="Local fin-identify is not implemented by this detector service.",
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spawn a local fin detection API server.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu, 0, mps.")
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.0,
        help="Optional crop padding as a fraction of detected box width/height.",
    )
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    service = FinDetectionService(
        model_path=args.model_path,
        image_size=args.imgsz,
        confidence=args.conf,
        iou=args.iou,
        device=args.device,
        max_detections=args.max_det,
        crop_padding=args.crop_padding,
    )
    app = create_app(service)
    LOGGER.info("Serving local fin detection model from %s", args.model_path)
    LOGGER.info("Use base_url=http://%s:%s/api/inference in scripts/fin_finder.py", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
