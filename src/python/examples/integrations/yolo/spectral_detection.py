"""
This spectral detection tool combines standard RGB detection methods (in this case provided by a
[YOLO](https://github.com/ultralytics/ultralytics) model) with spectral detection to more robustly classify objects.

This class has been designed to run with the Living Optics `analysis` tool. You can try it with some Living Optics
[sample data](../../data-samples-overview.md) using the command below.

Or simply omit the `--calibration-path` and `--file` arguments to try it with a live Living Optics Camera.

Commandline:
    ```cmd
    python analysis.py
        --file /datastore/lo/share/samples/face-spoofing/face-spoof-demo.lo-raw
        --calibration-path /datastore/lo/share/samples/face-spoofing/demo-calibration-face-spoofing
        --analysis lo.sdk.integrations.huggingface.spectral_detection.SpectralDetectionAnalysis`
    ```

!!! Note

    Workstation users should run this from inside their Python [virtual environment](../../install-guide.md#quick-install)

Tips:
    Data locations

    - Input : /datastore/lo/share/samples/face_spoofing/face-spoof-demo.lo-raw
    - Input : /datastore/lo/share/samples/face_spoofing/demo-calibration-face-spoofing
    - Output : None

Installation:
    Installed as part of the sdk.
    Custom installations may require you to install lo-sdk extras:

    - `pip install '{wheel}[yolo]'`
"""


from typing import Tuple

import cv2
import numpy as np
from lo.sdk.api.acquisition.data import Calibration
from lo.sdk.api.acquisition.data.coordinates import (
    NearestUpSample,
    SceneToSpectralIndex,
)
from lo.sdk.api.acquisition.data.formats import LORAWtoDEBAYEREDGRAY12, LORAWtoRGB8
from lo.sdk.api.analysis.classifier import spectral_roi_classifier
from lo.sdk.api.analysis.enums import MetricTypes, get_method_definition
from lo.sdk.api.analysis.helpers import check_new_spectrum, draw_rois_and_labels
from lo.sdk.integrations.yolo.helpers import get_from_huggingface_model
from lo.sdk.tools.analysis.apps.spectral_decode import SpectralDecode


class SpectralDetectionAnalysis(SpectralDecode):
    """
    This class can be used stand alone but was designed to work with the LO analysis tool.

    The purpose of this class is to run a Hugging face model on the scene view of the Living Optics Camera to perform object
    detection. Spectral classification is then run to enhance the object detection accuracy. For example, spectral
    classification is robust to face-spoofing, whereas conventional, deep learning models operating on RGB images, are
    not.

    By default, this class loads a face detection model.
    """

    def __init__(self, **kwargs):
        super(SpectralDetectionAnalysis, self).__init__(**kwargs)
        self.upsampler = None
        self.point_finder = None
        self.model = None
        self.object_names = {1: "Unclassified"}
        self.stored_spectra = {}

    def init(self, calibration: Calibration, **kwargs):
        super().init(calibration)
        self.upsampler = NearestUpSample(calibration.sampling_coordinates, output_shape=(1920, 1920), origin=(64, 256))
        self.point_finder = SceneToSpectralIndex(calibration.sampling_coordinates)

    def __call__(
        self,
        frame: Tuple,
        spectrum: np.ndarray,
        measure: MetricTypes = MetricTypes.SAM,
        scale_factor: int = 3,
        classification_threshold: float = 0.2,
        repo_id: str = "jaredthejelly/yolov8s-face-detection",
        file_name: str = "YOLOv8-face-detection.pt",
        store_classified_spectra: bool = True,
        storage_threshold: float = 0.1,
        **kwargs,
    ) -> Tuple[list, np.ndarray, np.ndarray]:
        """
        Run multiclass spectral classification on ROIs detected by a Hugging face model.
        Args:
            frame: Tuple
            spectrum: (array_like) - target spectrum to classify against
            measure: (Enum) - selected spectral classification metric
            scale_factor: (int) - amount of down-sampling to apply to the scene view before running the Hugging face
                model on it. Larger number = Faster inference
            classification_threshold: (float) - Threshold above/below (depending on selected metric type) to consider a
                pixel part of the foreground class
            repo_id: (str) - Hugging face model repo ID.
            file_name: (str) - Hugging face model name within the Hugging face repo
            store_classified_spectra: (bool) - whether to store the spectra of classified objects to classify against
                later
            storage_threshold: (float) - minimum spectral angle difference between a newly classified spectrum and all
                previously classified spectra, in order for it to be stored. (Setting this too low will result in
                storing the spectrum of the same object multiple times due to lighting variation between frames - this
                could use a lot of memory).
            **kwargs:

        Returns:
            frame: (Tuple[list, np.ndarray, np.ndarray])
            spectra: (array_like)
            bounding_box_overlay: (array_like)

        """
        if self.model is None:
            self.model = get_from_huggingface_model(repo_id, file_name)

        (encoded_frame_info, encoded_frame), (scene_frame_info, scene_frame) = frame
        frame = [[encoded_frame_info, encoded_frame], [scene_frame_info, scene_frame]]

        metadata, scene_frame, spectra = self.spectral_decoder(frame, scene_decoder=LORAWtoRGB8)

        scene_frame = np.flipud(scene_frame)
        low_res_frame = ((scene_frame / np.max(scene_frame))[::scale_factor, ::scale_factor]) * 255

        # Apply face detection on low resolution scene frame for speed
        boxes = self.model(low_res_frame)
        boxes = boxes[0].boxes.data.detach().cpu().numpy()

        # Convert bounding box format to (x1, y1, w, h)
        bbox_coordinates = (
            np.array([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes.astype(np.int32)]) * scale_factor
        )

        # Select spectra to classify against
        target_spectra = [spectrum] + [self.stored_spectra[k] for k in self.object_names if k in self.stored_spectra]
        target_spectra = np.asarray(target_spectra)

        metric = get_method_definition(measure)

        class_idxs, confidences, classified_spectra, segmentations = spectral_roi_classifier(
            spectra,
            target_spectra,
            metric,
            bbox_coordinates / scale_factor,
            self.spectral_decoder.sampling_coordinates,
            classification_threshold=classification_threshold,
            scale_factor=scale_factor,
        )

        for l, c, s in zip(class_idxs, confidences, classified_spectra):
            m_type = getattr(MetricTypes, metric.__name__)
            if (
                c >= 1 - classification_threshold
                and m_type.value == 0
                or c >= classification_threshold
                and m_type.value == 1
            ) and l > 1:
                self.object_names[l] = f"Object {l - 1}"
                stored_spectra = np.array([self.stored_spectra[k] for k in self.object_names if k in stored_spectra])
                if store_classified_spectra and check_new_spectrum(s, storage_threshold, stored_spectra):
                    self.stored_spectra[l] = s

        bounding_box_overlay = self.get_bounding_box_overlay(
            scene_frame,
            bbox_coordinates,
            class_idxs,
            confidences,
        )

        frame[1][1] = LORAWtoDEBAYEREDGRAY12(frame[1][1])
        return frame, spectra, bounding_box_overlay.astype(np.uint8)

    def get_bounding_box_overlay(
        self, scene: np.ndarray, rois: np.ndarray, class_idxs: np.ndarray, confidences: np.ndarray
    ) -> np.ndarray:
        """
        Create an overlay with bounding boxes, class labels and confidence scores
        Args:
            scene: (array_like) - the scene view from an Living Optics Camera
            rois: (array_like) - List of ROIs [(x1, y1, w, h)]
            class_idxs: (array_like) - List of class indexes generated by the spectral_roi_classifier method
            confidences: (array_like) - List of confidence scores

        Returns:
            overlay: (array_like)
        """
        overlay = np.zeros_like(scene)

        overlay = np.flipud(
            cv2.cvtColor(
                draw_rois_and_labels(overlay, rois, class_idxs, confidences, class_labels=self.object_names),
                cv2.COLOR_RGB2GRAY,
            )
        )

        return overlay.astype(np.uint8)
