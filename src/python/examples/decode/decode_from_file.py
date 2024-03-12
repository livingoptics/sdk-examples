"""
# Decode a raw camera data file, view and save.

This example demonstrates how to open and decode a loraw file and save the decoded information to the `.lo` format file
of the encoded view and display the scene and spectra using matplotlib.

Commandline:
    `python decode_from_file.py`

!!! Note

    Workstation users should run this from inside their Python [virtual environment](../../../install-guide.md#quick-install)


Tips:
    Data locations

    - Input : /datastore/lo/share/data/raw_test.loraw
    - Output : /datastore/lo/share/data/decoded_test.loraw

    Download the LO sample data folder and unpack it in /datastore/lo/share/samples.

    To run this example application you should be in {install-dir}/src/python/examples/decode

    This example will run indefinitely, so press ctrl+c to close it.


Installation:
    Installed as part of the sdk.
    Custom installations may require you to install lo-sdk extras:

    - `pip install {wheel}['matplotlib']`

"""

import os
import os.path as op

from lo.sdk.api.acquisition.data.decode import SpectralDecoder
from lo.sdk.api.acquisition.io.open import open
from lo.sdk.helpers.path import getdatastorepath
from lo.sdk.helpers.time import timestamp
from lo.sdk.integrations.matplotlib.simple_viewer import LOMPLViewer


def main():
    # If you have not already run the raw data capture example, this example will fail. Run that one first
    # (../capture/capture-loraw.py)
    # Latest Calibration is a symlink to the calibration for the camera that is connected.

    lo_share_dir = os.path.join(getdatastorepath(), "lo", "share")
    sample_folder = os.path.join(lo_share_dir, "samples", "decode")
    calibration_path = os.path.join(sample_folder, "demo-calibration-decode")
    decoder = SpectralDecoder.from_calibration(calibration_path)
    filename = "raw_test.loraw"
    filepath = os.path.join(sample_folder, filename)
    assert op.exists(filepath), (
        "If you have not already run the raw data capture example, this example will fail. Run that one first"
        "(../capture/capture-loraw.py)"
    )

    output_dir = os.path.join(lo_share_dir, "data", "decode")
    os.makedirs(output_dir, exist_ok=True)
    current_time = timestamp()
    output_filename = f"decoded-test-{current_time}.lo"
    outfile = os.path.join(output_dir, output_filename)

    # get image and spectra viewers
    viewer = LOMPLViewer()
    scene_view = viewer.add_scene_view(title="Scene view")
    spectra_view = viewer.add_spectra_view(title="Extracted spectra")

    with open(filepath) as f:
        with open(outfile, "w", format="lo") as outf:
            for frame in f:
                processed_frame = decoder(frame)
                outf.write(processed_frame)

                metadata, scene, spectra = processed_frame

                # Display the scene and every forth spectra
                # For 8bit and fp32 format converters see lo.sdk.api.acquisition.data.formats: LORAWtoSPECTRA8
                # and LORAWtoSPECTRAF32
                scene_view.update(scene)
                spectra_view.update(spectra=spectra[::4, :], wavelengths=metadata.wavelengths)
                viewer.render()


if __name__ == "__main__":
    main()
