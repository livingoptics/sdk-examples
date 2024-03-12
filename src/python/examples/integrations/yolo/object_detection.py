"""
# Object detection with LOCamera
The LOCamera can be used in conjunction with third party libraries such as [YOLO](https://github.com/ultralytics/ultralytics) to detect objects and faces in
the RGB scene view.

This example demonstrates face detection using a YOLO model on the scene view.

Commandline:
    `python object_detection.py`

!!! Note

    Workstation users should run this from inside their Python [virtual environment](../../install-guide.md#quick-install)


Tips:
    Data locations

    - Input : None
    - Output : None

    This example will run indefinitely, so press ctrl+c to close it.

Installation:
    Installed as part of the sdk.
    Custom installations may require you to install lo-sdk extras:

    - `pip install '{wheel}[yolo]'`


"""

import cv2
import numpy as np

# Hack to get around PyQt incompatibility
img = (np.random.random((256, 256)) * 255).astype(np.uint8)
cv2.imshow("dummy", img)
cv2.waitKey(1)
cv2.destroyAllWindows()

from lo.sdk.api.acquisition.data.formats import LORAWtoRGB8
from lo.sdk.api.analysis.helpers import draw_rois_and_labels
from lo.sdk.api.camera.camera import LOCamera
from lo.sdk.integrations.yolo.helpers import get_from_huggingface_model


def main():
    model = get_from_huggingface_model()
    with LOCamera() as cam:
        cam.frame_rate = 10000000
        cam.gain = 100
        cam.exposure = 633333

        while True:
            try:
                (encoded_info, encoded_frame), (scene_info, scene_frame) = cam.get_frame()
                scene_frame = LORAWtoRGB8(scene_frame)
                scene_frame = np.flipud(scene_frame)

                low_res_frame = cv2.normalize(scene_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)[::3, ::3]

                # Apply face detection on low resolution scene frame for speed
                boxes = model(low_res_frame)
                boxes = boxes[0].boxes.data.detach().cpu().numpy()

                # Reformat ROIs to be [(x1, y1, w, h)]
                boxes = np.asarray([(b[0], b[1], b[2] - b[0], b[3] - b[2]) for b in boxes])
                labels = [1] * len(boxes)
                low_res_frame = draw_rois_and_labels(low_res_frame, boxes, labels)
                cv2.imshow(low_res_frame)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
