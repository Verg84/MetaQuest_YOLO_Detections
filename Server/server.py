import asyncio
import base64
import binascii
import json
import os

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import cv2
import numpy as np
import uvicorn

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO


CONFIDENCE_THRESHOLD = 0.3
MAX_JPEG_BYTES = 4 * 1024 * 1024
MODEL_PATH = os.getenv("YOLO_MODEL", "yolo11n.pt")

model = None
executor = None
active_camera = None
latest_display = None


# =============================
# YOLO INFERENCE
# =============================

def process_frame(message):
    jpeg = base64.b64decode(
        message["image"],
        validate=True,
    )

    if not jpeg or len(jpeg) > MAX_JPEG_BYTES:
        raise ValueError("JPEG is empty or exceeds 4 MB")

    frame = cv2.imdecode(
        np.frombuffer(jpeg, np.uint8),
        cv2.IMREAD_COLOR,
    )

    if frame is None:
        raise ValueError("Could not decode JPEG")

    if max(frame.shape[:2]) > 4096:
        raise ValueError("Image dimensions exceed 4096 pixels")

    result = model.predict(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )[0]

    detections = []

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        detections.append({
            "label": result.names[class_id],
            "confidence": round(confidence, 3),
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2),
            "cx": round((x1 + x2) / 2, 2),
            "cy": round((y1 + y2) / 2, 2),
        })

    response = {
        "id": message["id"],
        "camPos": message["camPos"],
        "camRot": message["camRot"],
        "imageWidth": frame.shape[1],
        "imageHeight": frame.shape[0],
        "detections": detections,
    }

    return response, result.plot()


# =============================
# CAMERA DISPLAY
# =============================

async def display_frames():
    global latest_display

    # Keep OpenCV window operations on the main thread.
    while True:
        if latest_display is not None:
            frame = latest_display
            latest_display = None

            cv2.imshow(
                "Quest Camera - YOLO",
                frame,
            )

        cv2.waitKey(1)
        await asyncio.sleep(0.03)


# =============================
# STARTUP / SHUTDOWN
# =============================

@asynccontextmanager
async def lifespan(app):
    global model, executor

    print(f"Loading YOLO: {MODEL_PATH}", flush=True)

    model = YOLO(MODEL_PATH)
    executor = ThreadPoolExecutor(max_workers=1)

    display_task = asyncio.create_task(display_frames())

    try:
        yield

    finally:
        display_task.cancel()

        await asyncio.gather(
            display_task,
            return_exceptions=True,
        )

        executor.shutdown(
            wait=True,
            cancel_futures=True,
        )

        cv2.destroyAllWindows()


app = FastAPI(lifespan=lifespan)


# =============================
# QUEST CAMERA WEBSOCKET
# =============================

@app.websocket("/camera_detector")
async def detector_socket(ws: WebSocket):
    global active_camera, latest_display

    await ws.accept()

    if active_camera is not None:
        await ws.close(
            code=1008,
            reason="One Quest camera at a time",
        )
        return

    active_camera = ws
    print("Quest camera connected", flush=True)

    # Keep only the newest frame waiting for inference.
    queue = asyncio.Queue(maxsize=1)

    async def receiver():
        while True:
            text = await ws.receive_text()

            try:
                message = json.loads(text)

                if not isinstance(message, dict):
                    raise ValueError("Expected a JSON object")

                required_fields = (
                    "id",
                    "image",
                    "camPos",
                    "camRot",
                )

                if not all(
                    key in message
                    for key in required_fields
                ):
                    raise ValueError(
                        "Required fields: id, image, camPos, camRot"
                    )

                if not isinstance(message["image"], str):
                    raise ValueError(
                        "image must be a base64 JPEG string"
                    )

                max_base64_length = (
                    4 * ((MAX_JPEG_BYTES + 2) // 3)
                )

                if len(message["image"]) > max_base64_length:
                    raise ValueError(
                        "Encoded image exceeds 4 MB JPEG limit"
                    )

            except (ValueError, TypeError) as error:
                await ws.close(
                    code=1008,
                    reason=str(error)[:120],
                )
                return

            if queue.full():
                queue.get_nowait()

            queue.put_nowait(message)

    async def worker():
        global latest_display

        loop = asyncio.get_running_loop()

        while True:
            message = await queue.get()

            try:
                response, annotated = (
                    await loop.run_in_executor(
                        executor,
                        process_frame,
                        message,
                    )
                )

                latest_display = annotated

                await ws.send_json(response)

            except (ValueError, binascii.Error) as error:
                await ws.send_json({
                    "id": message["id"],
                    "error": str(error),
                })

    receiver_task = asyncio.create_task(receiver())
    worker_task = asyncio.create_task(worker())

    try:
        done, _ = await asyncio.wait(
            [receiver_task, worker_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in done:
            task.result()

    except WebSocketDisconnect:
        pass

    except Exception as error:
        print(f"Camera error: {error}", flush=True)

        try:
            await ws.close(
                code=1011,
                reason="Camera processing failed",
            )
        except Exception:
            pass

    finally:
        receiver_task.cancel()
        worker_task.cancel()

        await asyncio.gather(
            receiver_task,
            worker_task,
            return_exceptions=True,
        )

        # Wait for any inference still running after cancellation.
        await asyncio.get_running_loop().run_in_executor(
            executor,
            lambda: None,
        )

        latest_display = None
        cv2.destroyAllWindows()
        active_camera = None

        print("Quest camera disconnected", flush=True)


# =============================
# RUN SERVER
# =============================

if __name__ == "__main__":
    print(
        "Unity URL: ws://10.0.0.46:8765/camera_detector"
    )

    uvicorn.run(
        app,
        host="10.0.0.46",
        port=8765,
        workers=1,
        ws="websockets",
        ws_max_size=6 * 1024 * 1024,
        ws_per_message_deflate=False,
    )