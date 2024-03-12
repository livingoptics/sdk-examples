"""
# Object detection helper methods

"""
import cv2
from lo.sdk.helpers._import import import_extra

huggingface_hub = import_extra("huggingface_hub", extra="yolo")
hf_hub_download = huggingface_hub.hf_hub_download
ultralytics = import_extra("ultralytics", extra="yolo")
YOLO = ultralytics.YOLO
cuda = import_extra("torch.cuda", extra="yolo")


def get_from_huggingface_model(
    repo_id: str = "jaredthejelly/yolov8s-face-detection", filename: str = "YOLOv8-face-detection.pt"
) -> YOLO:
    """
    Get a hugging face model
    Args:
        repo_id: (str) - Name of the repo to download the model from
        filename: (str) - Name of the model file to download

    Returns:
        model: (ultralytics.YOLO) - The loaded model object
    """
    # Initialise face detector through hugging face
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
    )
    device = "cpu"
    if cuda.is_available():
        device = "gpu"
    model = YOLO(model_path)
    model.to(device)
    return model


def select_location_cv2_callback(event, x, y, flags, params):
    """
    This method should be used in conjunction with cv2.setMouseCallback.
    ```python
    scene = np.random.random((256, 256)) # dummy image
    params = {}
    cv2.imshow("scene", scene)
    cv2.setMouseCallback("scene", select_location_cv2_callback, params)
    ```
    Args:
        event: (cv2.MouseEventTypes) - one of the cv:2.MouseEventTypes constants.
        x: (float) - the x-coordinate of the mouse event.
        y: (float) - the y-coordinate of the mouse event.
        flags: (cv2.MouseEventFlags) - one of the cv2.MouseEventFlags constants.
        params: (dict) - dict of parameters that we will set in the callback

    Returns:

    """
    if event == cv2.EVENT_LBUTTONDOWN:
        params["click_location"] = [x, y]
