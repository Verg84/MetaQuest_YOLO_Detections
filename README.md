# Meta Quest Camera Object Detector

Stream real-world camera frames from a Meta Quest Unity app to a Python YOLO detector. The server displays annotated frames in an OpenCV window and returns detection coordinates to Unity over WebSocket.

## Requirements

- Meta Quest 3 or Quest 3S with camera access support, developer mode enabled, and camera permission granted.
- Unity **6000.3.14f1**, the version used by this project.
- Meta XR Core SDK and Mixed Reality Utility Kit (MRUK), available through Meta XR All-in-One SDK.
- NativeWebSocket for Unity.
- Python 3.10 or newer on a desktop PC with a graphical display.
- PC and Quest connected to the same local network.

## Project layout

```text
Assets/          Unity scenes, scripts, and assets
Packages/        Unity package dependencies
ProjectSettings/ Unity project configuration
Server/server.py Python FastAPI and YOLO server
```

## Start the Python server

Open a terminal in the repository root:

```powershell
cd Server
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install fastapi "uvicorn[standard]" ultralytics opencv-python numpy "websockets>=14,<16"
.\.venv\Scripts\python.exe server.py
```

The server listens on **port 8765** at `/camera_detector`. The default `yolo11n.pt` weights download on first launch if they are not present. Do not install `opencv-python-headless`: the server uses an OpenCV display window.

To use a custom Ultralytics detection model:

```powershell
$env:YOLO_MODEL = "C:\path\to\best.pt"
.\.venv\Scripts\python.exe server.py
```

Run one server worker because the camera, model, and display state are held in one process. Stop the server with Ctrl+C.

## Configure Unity

1. Open this repository folder in Unity Hub using Unity 6000.3.14f1. Allow package restoration and asset import to complete.
2. If NativeWebSocket needs to be installed, open Package Manager and install from Git URL:
   `https://github.com/endel/NativeWebSocket.git#upm-2`.
3. Open the project scene. Ensure it has a Camera Rig, **Passthrough**, and **Passthrough Camera Access**. Passthrough displays your surroundings in the headset; Camera Access supplies the frames sent to Python.
4. Enable camera access on OVR Manager and configure runtime headset camera permission. Grant permission when the app launches.
5. On `QuestYoloClient`, assign the `PassthroughCameraAccess` component to **Camera Access**.
6. Run `ipconfig` on the PC and identify its active Wi-Fi/Ethernet IPv4 address. Set the client's **Server URL** in the Inspector to:
   `ws://YOUR_PC_IP:8765/camera_detector`.
7. Start with **5 FPS**, **640-pixel frame width**, and **JPEG quality 70**.
8. Configure Android XR and player settings through Meta's Project Setup Tool. Use OpenXR, IL2CPP, ARM64, and Internet Access set to Require.
9. Build and run on the Quest. Rebuild whenever you change the Inspector URL in the installed app.

The server should print `Quest camera connected`. An OpenCV window then displays the Quest feed with YOLO bounding boxes, class labels, and confidence values. Unity receives detection JSON; this project does not automatically turn those pixel coordinates into world-space overlays.

## WebSocket messages

Unity sends JSON **text**, with the JPEG encoded as base64:

```json
{
  "id": 1,
  "image": "<base64 JPEG>",
  "camPos": {"x": 0, "y": 1.5, "z": 0},
  "camRot": {"x": 0, "y": 0, "z": 0, "w": 1}
}
```

The server responds with JSON:

```json
{
  "id": 1,
  "camPos": {"x": 0, "y": 1.5, "z": 0},
  "camRot": {"x": 0, "y": 0, "z": 0, "w": 1},
  "imageWidth": 640,
  "imageHeight": 480,
  "detections": [
    {
      "label": "cup",
      "confidence": 0.92,
      "x1": 100,
      "y1": 120,
      "x2": 200,
      "y2": 250,
      "cx": 150,
      "cy": 185
    }
  ]
}
```

Coordinates are pixels in the transmitted image, with the origin at the top-left. The response echoes the frame ID and camera pose. No detections are represented by an empty array. The server may skip waiting frames if inference cannot keep up. Decoding errors return an `id` and `error` message.

## Troubleshooting

- **No connection:** check the PC IP, port 8765, endpoint, Python process, and Windows Firewall's private-network access. Guest networks can prevent devices from reaching each other.
- **Connected but no frames:** check the camera reference, component state, and camera permission.
- **Virtual scene visible but no surroundings:** add the Passthrough building block, check the underlay and camera background alpha, and remove duplicate standalone cameras.
- **Upside-down feed:** toggle Flip Vertically on the Unity client.
- **Slow detections:** lower frame rate or resolution, or use a supported GPU setup.

This prototype uses unencrypted WebSockets without authentication on a trusted local network. Keep API keys and credentials out of the repository. Ignore virtual environments, generated Unity folders, and downloaded model weights.

## References

- [Meta Passthrough Camera API setup](https://developers.meta.com/vr/documentation/unity/unity-pca-documentation/)
- [NativeWebSocket](https://github.com/endel/NativeWebSocket)
- [Ultralytics prediction documentation](https://docs.ultralytics.com/modes/predict/)
- [FastAPI WebSockets](https://fastapi.tiangolo.com/advanced/websockets/)
