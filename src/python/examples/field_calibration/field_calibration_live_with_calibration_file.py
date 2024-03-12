#!/usr/bin/python3

# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

"""
# Run a field calibration from a pre-captured filtered image to improve spectral decoding accuracy

This example demonstrates how to connect to the LOCamera, capture data, use a pre-captured 600nm filtered image to
improve spectral decoding.

Commandline:
    `python field_calibration_live_with_calibration_file.py`

!!! Note

    Workstation users should run this from inside their Python [virtual environment](../../../install-guide.md#quick-install)


Tips:
    Data locations

    - Input : /datastore/lo/share/data/flat-field-filtered.loraw
    - Output : None

    Please ensure you have run the live field calibration example before running this:
        'python field_calibration_live.py'

    To run this example application you should be in {install-dir}/src/python/examples/field_calibration

Installation:
    Installed as part of the sdk.
    Custom installations may require you to install lo-sdk extras:

    - `pip install {wheel}['matplotlib']`

"""
import os
from glob import glob

from lo.sdk.api.acquisition.data.decode import SpectralDecoder
from lo.sdk.api.acquisition.data.formats import LORAWtoRGB8
from lo.sdk.api.camera.camera import LOCamera
from lo.sdk.helpers.path import getdatastorepath
from lo.sdk.integrations.matplotlib.simple_viewer import LOMPLViewer


def main():
    calibration_folder = os.path.join(getdatastorepath(), "lo", "share", "calibrations", "latest_calibration")

    data_folder = os.path.join(getdatastorepath(), "lo", "share", "data")

    calibration_files = glob(os.path.join(data_folder, "flat-field-filtered-*.loraw"))

    if len(calibration_files) == 0:
        raise FileNotFoundError(
            f"filtered-flat-field.loraw file not found in {data_folder}, please ensure you have run: "
            f"field_calibration_live.py example"
        )

    calibration_file = calibration_files[-1]

    # Initialise a decoder, updating the calibration artefact with a pre-captured filtered image.
    decode = SpectralDecoder.from_calibration(calibration_folder, calibration_file)

    # get image viewers
    viewer = LOMPLViewer()
    scene_view = viewer.add_scene_view(title="Scene view")
    spectra_view = viewer.add_spectra_view(title="Extracted spectra")

    with LOCamera() as cam:
        cam.frame_rate = int(5120e3)  # μhz
        cam.exposure = int(1950e3)  # μs
        cam.gain = 0  # change the settings if running live
        for frame in cam:
            # Decode the spectra from the encoded view and convert the scene view to RGB8
            try:
                processed_frame = decode(frame, scene_decoder=LORAWtoRGB8)

                info, scene, spectra = processed_frame

                # Display the scene
                scene_view.update(scene)
                # For 8bit and fp32 format converters see lo.sdk.api.acquisition.data.formats: LORAWtoSPECTRA8
                # and LORAWtoSPECTRAF32

                # Display the spectra
                spectra_view.update(spectra=spectra, wavelengths=info.wavelengths)
                viewer.render()
            except:
                break


if __name__ == "__main__":
    main()
