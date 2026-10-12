using Meta.XR;
using Meta.XR.MRUtilityKit;
using NativeWebSocket;
using System;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

public class QuestYoloClient : MonoBehaviour
{
    [Header("Connection")]
    public string serverUrl =
        "ws://10.0.0.64:8765/camera_detector";

    public PassthroughCameraAccess cameraAccess;

    [Header("JPEG capture")]
    [Range(1, 20)]
    public int framesPerSecond = 5;

    [Range(160, 1280)]
    public int frameWidth = 640;

    [Range(10, 95)]
    public int jpegQuality = 70;

    public bool flipVertically;
    public bool logDetections = true;

    public YoloResponse LatestResponse { get; private set; }

    public event Action<YoloResponse> DetectionsReceived;

    private WebSocket socket;
    private RenderTexture capture;
    private Texture2D pixels;

    private bool stopping;
    private bool sending;
    private bool paused;

    private float nextFrame;
    private int frameId;

    // =============================
    // JSON MESSAGE TYPES
    // =============================

    [Serializable]
    public class CameraFrame
    {
        public int id;
        public string image;
        public Vector3 camPos;
        public Quaternion camRot;
    }

    [Serializable]
    public class Detection
    {
        public string label;
        public float confidence;

        public float x1;
        public float y1;
        public float x2;
        public float y2;

        public float cx;
        public float cy;
    }

    [Serializable]
    public class YoloResponse
    {
        public int id;

        public Vector3 camPos;
        public Quaternion camRot;

        public int imageWidth;
        public int imageHeight;

        public Detection[] detections;
        public string error;
    }

    // =============================
    // CONNECT / RECONNECT
    // =============================

    private async void Start()
    {
        if (cameraAccess == null)
        {
            Debug.LogError(
                "QuestYoloClient: assign Camera Access in the Inspector."
            );

            enabled = false;
            return;
        }

        while (!stopping)
        {
            var connection = new WebSocket(serverUrl);
            socket = connection;

            connection.OnOpen += () =>
            {
                Debug.Log("YOLO server connected");
            };

            connection.OnError += error =>
            {
                Debug.LogWarning(
                    "YOLO connection: " + error
                );
            };

            connection.OnClose += code =>
            {
                Debug.Log(
                    "YOLO connection closed: " + code
                );
            };

            connection.OnMessage += bytes =>
            {
                if (!stopping && socket == connection)
                {
                    HandleResponse(bytes);
                }
            };

            try
            {
                await connection.Connect();

                while (
                    !stopping &&
                    connection.State == WebSocketState.Open
                )
                {
                    await Task.Delay(200);
                }
            }
            catch (Exception error)
            {
                if (!stopping)
                {
                    Debug.LogWarning(
                        "YOLO connection: " + error.Message
                    );
                }
            }
            finally
            {
                try
                {
                    await connection.Close();
                }
                catch (Exception)
                {
                }
            }

            if (!stopping)
            {
                await Task.Delay(2000);
            }
        }
    }

    // =============================
    // RECEIVE YOLO JSON
    // =============================

    private void HandleResponse(byte[] bytes)
    {
        YoloResponse response;

        try
        {
            string json = Encoding.UTF8.GetString(bytes);
            response = JsonUtility.FromJson<YoloResponse>(json);
        }
        catch (Exception error)
        {
            Debug.LogWarning(
                "Invalid YOLO JSON: " + error.Message
            );
            return;
        }

        if (response == null)
        {
            return;
        }

        if (!string.IsNullOrEmpty(response.error))
        {
            Debug.LogWarning(
                "YOLO frame " + response.id +
                ": " + response.error
            );
            return;
        }

        LatestResponse = response;

        if (logDetections)
        {
            int count = response.detections?.Length ?? 0;

            Debug.Log(
                "YOLO frame " + response.id +
                ": " + count + " detections"
            );
        }

        DetectionsReceived?.Invoke(response);
    }

    // =============================
    // CAPTURE AND SEND FRAMES
    // =============================

    private async void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        socket?.DispatchMessageQueue();
#endif

        if (
            stopping ||
            paused ||
            sending ||
            Time.unscaledTime < nextFrame ||
            socket == null ||
            socket.State != WebSocketState.Open ||
            !cameraAccess.IsPlaying
        )
        {
            return;
        }

        Texture source = cameraAccess.GetTexture();

        if (
            source == null ||
            source.width < 1 ||
            source.height < 1
        )
        {
            return;
        }

        var connection = socket;

        nextFrame =
            Time.unscaledTime +
            1f / Mathf.Max(1, framesPerSecond);

        sending = true;

        try
        {
            // Pose of the selected RGB camera.
            Pose pose = cameraAccess.GetCameraPose();

            int width = Mathf.Clamp(frameWidth, 160, 1280);

            int height = Mathf.Max(
                1,
                Mathf.RoundToInt(
                    width * (float)source.height / source.width
                )
            );

            EnsureTextures(width, height);

            if (flipVertically)
            {
                Graphics.Blit(
                    source,
                    capture,
                    new Vector2(1, -1),
                    new Vector2(0, 1)
                );
            }
            else
            {
                Graphics.Blit(source, capture);
            }

            RenderTexture previous = RenderTexture.active;

            try
            {
                RenderTexture.active = capture;

                pixels.ReadPixels(
                    new Rect(0, 0, width, height),
                    0,
                    0
                );

                pixels.Apply(false);
            }
            finally
            {
                RenderTexture.active = previous;
            }

            byte[] jpeg = pixels.EncodeToJPG(
                Mathf.Clamp(jpegQuality, 10, 95)
            );

            var message = new CameraFrame
            {
                id = ++frameId,
                image = Convert.ToBase64String(jpeg),
                camPos = pose.position,
                camRot = pose.rotation
            };

            string json = JsonUtility.ToJson(message);

            await connection.SendText(json);
        }
        catch (Exception error)
        {
            if (!stopping)
            {
                Debug.LogWarning(
                    "Quest frame send: " + error.Message
                );
            }
        }
        finally
        {
            sending = false;
        }
    }

    // =============================
    // TEXTURE MANAGEMENT
    // =============================

    private void EnsureTextures(int width, int height)
    {
        if (
            capture != null &&
            capture.width == width &&
            capture.height == height
        )
        {
            return;
        }

        ReleaseTextures();

        capture = new RenderTexture(
            width,
            height,
            0,
            RenderTextureFormat.ARGB32
        );

        capture.Create();

        pixels = new Texture2D(
            width,
            height,
            TextureFormat.RGB24,
            false
        );
    }

    private void ReleaseTextures()
    {
        if (capture != null)
        {
            capture.Release();
            Destroy(capture);
            capture = null;
        }

        if (pixels != null)
        {
            Destroy(pixels);
            pixels = null;
        }
    }

    // =============================
    // PAUSE / CLEANUP
    // =============================

    private void OnApplicationPause(bool value)
    {
        paused = value;
    }

    private async void OnDestroy()
    {
        stopping = true;
        ReleaseTextures();

        if (socket == null)
        {
            return;
        }

        try
        {
            socket.CancelConnection();
            await socket.Close();
        }
        catch (Exception)
        {
        }
    }
}