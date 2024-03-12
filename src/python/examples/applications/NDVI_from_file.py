# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

"""
# Normalised difference vegetation index (NDVI) from file.

In this application example, we demonstrate how to calculate and display a band ratio from a pre-captured file.

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
    `python NDVI_from_file.py`

!!! Note

    Workstation users should run this from inside their Python [virtual environment](../../../install-guide.md#quick-install)


Tips:
    Data locations

    - Input : /datastore/lo/share/samples/ndvi/NDVI-demo.loraw
    - Input : /datastore/lo/share/samples/ndvi/demo-calibration-ndvi
    - Output : None

    [Download]("../../../data-samples-overview.md") the Living Optics sample data folder and unpack it in `/datastore/lo/share/samples`.

    To run this example application you should be in {install-dir}/src/python/examples/applications

Installation:
    Installed as part of the sdk.
    Custom installations may require you to install lo-sdk extras:

    - `pip install {wheel}['matplotlib']`

"""

import os

from lo.sdk.api.acquisition.data.coordinates import NearestUpSample
from lo.sdk.api.acquisition.data.decode import SpectralDecoder
from lo.sdk.api.acquisition.data.formats import LORAWtoRGB8
from lo.sdk.api.acquisition.io.open import open
from lo.sdk.api.analysis.overlays import NDVI_overlay
from lo.sdk.helpers.path import getdatastorepath
from lo.sdk.integrations.matplotlib.simple_viewer import LOMPLViewer


def main():
    sample_folder = os.path.join(getdatastorepath(), "lo", "share", "samples", "ndvi")
    calibration_folder = os.path.join(sample_folder, "demo-calibration-ndvi")

    file = os.path.join(sample_folder, "NDVI-demo.loraw")
    decode = SpectralDecoder.from_calibration(calibration_folder)

    # get image viewers
    viewer = LOMPLViewer()
    scene_view = viewer.add_scene_view(title="Scene view")
    ndvi_map_view = viewer.add_scene_view(title="NDVI map")

    # Open the camera
    with open(file) as cam:
        # change some settings using camera level api
        upsampler = NearestUpSample(
            decode.sampling_coordinates, output_shape=(1920, 1920), origin=(64, 256)
        )  # This object upsamples spectral coordinates to teh desired output resolution.

        # process data from camera
        for frame in cam:
            try:
                info, scene_frame, spectra = decode(frame, scene_decoder=LORAWtoRGB8)
                ndvi_map = NDVI_overlay(
                    spectra, info.wavelengths, upsampling_method=upsampler, vis_limits=(400, 700), nir_limits=(700, 900)
                )

                # put frames on the image queue
                scene_view.update(scene_frame)
                ndvi_map_view.update(ndvi_map)
                viewer.render()
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
