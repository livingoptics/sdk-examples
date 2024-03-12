# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

"""
# Spectral conversion to RGB from file.

This example demonstrates a simple method for estimating the "human-readable" colour of a hyperspectral image.
Whilst the Living Optics Camera splits the Vis-Nir spectrum into 96 different colour axes, the human eye only has a 3 axis colour
space. This means, knowing what a spectrum would look like to the human eye is difficult at a glance.

By selecting the channels at the centre of the frequency response curve of each of the three different cone cells in the
human eye (the cells responsible for perceiving coloured light), we can estimate what a spectrum would look like to the
human eye.

Commandline:
    `python spectra_to_rgb.py`

!!! Note

    Workstation users should run this from inside their Python [virtual environment](../../../install-guide.md#quick-install)


Tips:
    Data locations

    - Input : /datastore/lo/share/samples/spectra-to-rgb/spectra-to-rgb-demo.loraw
    - Input : /datastore/lo/share/samples/spectra-to-rgb/flat-field-600nm.loraw
    - Input : /datastore/lo/share/samples/spectra-to-rgb/demo-calibration-spectra-to-rgb
    - Output : None

    [Download]("../../../../../../docs/docs/sdk/data-samples-overview.md") the Living Optics sample data folder and unpack it in `/datastore/lo/share/samples`.

    To run this example application you should be in {install-dir}/src/python/examples/applications

Installation:
    Installed as part of the sdk.
    Custom installations may require you to install lo-sdk extras:

    - `pip install {wheel}['matplotlib']`

"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from lo.sdk.api.acquisition.data.decode import SpectralDecoder
from lo.sdk.api.acquisition.io.open import open
from lo.sdk.helpers.path import getdatastorepath


def simple_rgb(spectra, wavelengths):
    rgb_idx = [np.argmin(np.abs(wavelengths - w)) for w in [625, 550, 475]]

    return spectra[rgb_idx]


def main():
    # folder containing your data, factory calibration artifact and calibration update file
    sample_folder = os.path.join(getdatastorepath(), "lo", "share", "samples", "spectra-to-rgb")

    # video file for processing
    video_file_name = os.path.join(sample_folder, "spectra-to-rgb.loraw")

    # calibration artifact folder
    calibration_folder = os.path.join(sample_folder, "demo-calibration-spectra-to-rgb")

    # drift compensation file name
    calibration_update_file = os.path.join(sample_folder, "flat-field-600nm.loraw")

    # Build spectral decoder based on factory calibration - calibration_folder
    # if supplied, use drift compensation 600nm filtered image to update calibration - calibration_update_file
    decoder = SpectralDecoder.from_calibration(calibration_folder, calibration_update_file)

    # open matplotlib figure
    fgr = plt.figure()
    ax = fgr.add_subplot()
    ax.xaxis.set_label_text("wavelength [nm]")
    ax.yaxis.set_label_text("spectral radiance [arb.]")
    lines = []
    # load video file and plot mean spectra for the scene
    with open(video_file_name, mode="r") as video_file:
        noFrames = len(video_file)
        print(f"Number of frames in video: {noFrames}")
        firstFrame = True
        for i, frame in enumerate(video_file):
            try:
                # extract frame (frame) from video sequence (video_file)
                # extract metadata, scene and spectra from the main frame
                metadata, scene_frame, spectra = decoder(frame)

                # extract start time from frame metadata metadata
                frameTime = metadata.timestamp_s
                if firstFrame:
                    startTime = frameTime

                # calculate the average of all spectra within the image
                scene_average = np.mean(spectra, axis=0)

                # calculate rgb colour of spectra based on a simple set of wavelengths
                rgb = simple_rgb(scene_average, metadata.wavelengths)
                rgb = rgb / rgb.max()

                # plot line using matplotlib for the video sequence
                if not firstFrame:
                    line = lines[i - 1].pop(0)
                    line.set_color((line.get_color() / 3) + [0.6, 0.6, 0.6])
                line = ax.plot(metadata.wavelengths, scene_average, label="Scene average", color=rgb)

                # update figure title to the current frame time in the sequence
                ax.set_title(str(frameTime - startTime) + "s")
                plt.show(block=False)

                lines.append(line)

                # pause for the time in seconds between frames
                plt.draw()
                plt.pause(1e6 / metadata.frame_rate)

                # switch off first frame toggle
                firstFrame = False

            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
