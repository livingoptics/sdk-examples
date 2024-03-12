#!/usr/bin/python3

# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

"""
# Run a field calibration from a pre-captured filtered image to improve spectral decoding accuracy

This example demonstrates how to run field calibrations from file, and then decode the filtered image.

Commandline:
    `python field_calibration_from_file.py`

!!! Note

    Workstation users should run this from inside their Python [virtual environment](../../../install-guide.md#quick-install)


Tips:
    Data locations

    - Input : datastore/lo/share/samples/field-calibration/flat-field-filtered.loraw
    - Output : None

    [Download]("../../../data-samples-overview.md") the Living Optics sample data folder and unpack it in `/datastore/lo/share/samples`.

    To run this example application you should be in {install-dir}/src/python/examples/field_calibration


Installation:
    Installed as part of the sdk.
    Custom installations may require you to install lo-sdk extras:

    - `pip install {wheel}['matplotlib']`

"""
import os

from lo.sdk.api.acquisition.data.decode import SpectralDecoder
from lo.sdk.api.acquisition.data.formats import LORAWtoRGB8
from lo.sdk.api.acquisition.io.open import open
from lo.sdk.helpers.path import getdatastorepath
from lo.sdk.integrations.matplotlib.simple_viewer import LOMPLViewer


def main():
    sample_folder = os.path.join(getdatastorepath(), "lo", "share", "samples", "field-calibration")
    calibration_folder = os.path.join(sample_folder, "demo-calibration")

    calibration_file = os.path.join(sample_folder, "flat_field_filtered.loraw")

    # Initialise a decoder, updating the calibration artefact with a pre-captured filtered image.
    decode = SpectralDecoder.from_calibration(calibration_folder, calibration_file)

    # get image viewers
    viewer = LOMPLViewer()
    scene_view = viewer.add_scene_view(title="Scene view")
    spectra_view = viewer.add_spectra_view(title="Extracted spectra")

    with open(calibration_file) as f:
        for frame in f:
            # Decode the spectra from the encoded view and convert the scene view to RGB8
            processed_frame = decode(frame, scene_decoder=LORAWtoRGB8)

            info, scene, spectra = processed_frame

            # Display the scene
            scene_view.update(scene)
            # For 8bit and fp32 format converters see lo.sdk.api.acquisition.data.formats: LORAWtoSPECTRA8
            # and LORAWtoSPECTRAF32

            # Display the spectra
            spectra_view.update(spectra=spectra, wavelengths=info.wavelengths)
            viewer.render()


if __name__ == "__main__":
    main()
