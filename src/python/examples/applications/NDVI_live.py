#!/usr/bin/python3

# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

"""
# Normalised difference vegetation index (NDVI) live.

In this application example, we demonstrate how to calculate and display a band ratio from a live camera feed.

NDVI is most often used in satellite imaging to determine the amount of living vegetation, dead vegetation and
non-vegetation there is in a given area.

Band-ratio metrics have many applications beside this such as:

- Estimating blood oxygenation
- Water detection
- Distinguishing between different cell types in biological samples
- Measuring the Brix index of liquids
- Etc.

This code can easily be adapted to look at other band-ratios by changing the values provided to the
`vis_limits` and `nir_limits` of the `NDVI_overlay()`.

Commandline:
    `python NDVI_live.py`

!!! Note

    Workstation users should run this from inside their Python [virtual environment](../../../install-guide.md#quick-install)


Tips:
    Data locations

    - Input : /datastore/lo/share/calibrations/latest_calibration
    - Output : None

    This example will run indefinitely, so press ctrl+c to close it.

    To run this example application you should be in {install-dir}/src/python/examples/applications

Installation:
    Installed as part of the sdk.
    Custom installations may require you to install lo-sdk extras:

    - `pip install {wheel}['matplotlib']`

"""

import os
from pathlib import Path

from lo.sdk.api.acquisition.data.coordinates import NearestUpSample
from lo.sdk.api.acquisition.data.decode import SpectralDecoder
from lo.sdk.api.acquisition.data.formats import LORAWtoRGB8
from lo.sdk.api.analysis.overlays import NDVI_overlay
from lo.sdk.api.camera.camera import LOCamera
from lo.sdk.helpers.path import getdatastorepath
from lo.sdk.integrations.matplotlib.simple_viewer import LOMPLViewer


def main():
    calibration_folder = os.path.join(getdatastorepath(), "lo", "share", "calibrations", "latest_calibration")
    decode = SpectralDecoder.from_calibration(calibration_folder)

    # get image viewers
    viewer = LOMPLViewer()
    scene_view = viewer.add_scene_view(title="Scene view")
    ndvi_map_view = viewer.add_scene_view(title="NDVI map")

    # Open the camera
    with LOCamera() as cam:
        # Change some settings using camera level api
        cam.frame_rate = 20000000
        cam.gain = 100
        cam.exposure = 63334

        upsampler = NearestUpSample(
            decode.sampling_coordinates, output_shape=(1920, 1920), origin=(64, 256)
        )  # This object up-samples spectral coordinates to the desired output resolution.

        # process data from camera
        for frame in cam:
            try:
                info, scene_frame, spectra = decode(frame, scene_decoder=LORAWtoRGB8)
                ndvi_map = NDVI_overlay(spectra, info.wavelengths, upsampling_method=upsampler)

                # put frames on the image queue
                scene_view.update(scene_frame)
                ndvi_map_view.update(ndvi_map)
                viewer.render()

            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
