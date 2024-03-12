#!/usr/bin/python3

# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

"""
# Spectral segmentation from file.

This example demonstrates how to find spectra that are similar to the spectrum extracted from the centre of the scene
and produce an overlay from a pre-captured file.

Whilst colour based segmentation can have unreliable results when working with RGB data it is very effective when
working with hyperspectral data. This is because many objects have unique hyperspectral "fingerprints" which are lost
when projecting down from the continuous space of real light to the quantised space of RGB.

With and Living Optics Camera, we have 33 times as much colour resolution, with the Vis-nir spectrum being divided into 96
channels, rather than the 3 of RGB cameras.

This means that objects which may appear the same colour to a conventional RGB camera, are easily distinguished by the
Living Optics Camera.

Commandline:
    `python spectral_segmentation_from_file.py`

!!! Note

    Workstation users should run this from inside their Python [virtual environment](../../../install-guide.md#quick-install)

Tips:
    Data locations

    - Input : /datastore/lo/share/samples/spectral-segmentation/oil-and-water-demo.loraw
    - Output : None

    [Download]("../../../../../../docs/docs/sdk/data-samples-overview.md") the Living Optics sample data folder and unpack it in `/datastore/lo/share/samples`.

    To run this example application you should be in {install-dir}/src/python/examples/applications

    This example will run indefinitely, so press ctrl+c to close it.

Installation:
    Installed as part of the sdk.
    Custom installations may require you to install lo-sdk extras:

    - `pip install {wheel}['cv2']`


"""
import os
from pathlib import Path

import cv2
import numpy as np
from lo.sdk.api.acquisition.data.coordinates import (
    NearestUpSample,
    SceneToSpectralIndex,
)
from lo.sdk.api.acquisition.data.decode import SpectralDecoder
from lo.sdk.api.acquisition.data.formats import LORAWtoRGB8
from lo.sdk.api.acquisition.io.open import open
from lo.sdk.api.analysis.metrics import spectral_angles
from lo.sdk.api.analysis.segmentations import spectral_segmentation
from lo.sdk.helpers.path import getdatastorepath


def main():
    sample_folder = os.path.join(getdatastorepath(), "lo", "share", "samples", "spectral-segmentation")
    calibration_folder = os.path.join(sample_folder, "demo-calibration-spectral-segmentation")
    decoder = SpectralDecoder.from_calibration(calibration_folder)

    file = os.path.join(sample_folder, "oil-and-water-demo.loraw")

    # Spectral angle threshold below which to assume that a spectra belongs to the foreground class and above to the
    # background class
    threshold = 0.12

    with open(file) as cam:
        # Instantiate and upsampler to convert from sparse spectra to a dense spectral image
        upsampler = NearestUpSample(decoder.sampling_coordinates)

        # Instantiate an indexer to convert from scene coordinates to the index of the nearest spectrum
        indexer = SceneToSpectralIndex(decoder.sampling_coordinates)

        # Set the coordinates to extract the target spectrum from
        scene_coords = (1000, 1300)
        spectral_idx = indexer(scene_coords)

        # Read a frame
        frame = cam.read()

        # Decode the frame
        info, scene_frame, spectra = decoder(frame, scene_decoder=LORAWtoRGB8)

        # Select the target spectrum
        target_spectra = spectra[spectral_idx, :]

        for frame in cam:
            try:
                # Decode the spectra and convert the scene view to RGB8
                info, scene_frame, spectra = decoder(frame, scene_decoder=LORAWtoRGB8)

                # Get a spectral segmentation map based on the spectral angle between the decoded spectra and the
                # target spectrum.
                sa = spectral_segmentation(
                    spectra, target_spectra, threshold, metric=spectral_angles, upsampling_method=upsampler
                )
                heatmap = np.uint8(255 * sa)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO)
                alpha = 0.3
                beta = 1 - alpha
                scene_frame = cv2.cvtColor(scene_frame, cv2.COLOR_BGR2RGB)
                super_imposed_img = cv2.addWeighted(heatmap, alpha, scene_frame, beta, 0)
                # put frames on the image queue
                cv2.imshow("Scene", cv2.resize(scene_frame, (640, 480)))
                cv2.imshow("Spectral segmentation", cv2.resize(super_imposed_img, (640, 480)))
                cv2.waitKey(1)

            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
