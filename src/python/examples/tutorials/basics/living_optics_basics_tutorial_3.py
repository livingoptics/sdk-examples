import logging
import os

import numpy as np
from lo.sdk.api.acquisition.data.decode import SpectralDecoder
from lo.sdk.api.acquisition.data.formats import LORAWtoLOGRAY12, LORAWtoRGB8
from lo.sdk.api.acquisition.io.open import open
from lo.sdk.api.camera.camera import LOCamera
from lo.sdk.helpers._import import import_extra
from lo.sdk.helpers.path import getdatastorepath
from lo.sdk.helpers.time import timestamp

# IMPORTANT: You MUST add the following line to set the log level BEFORE
# you import matplotlib, otherwise (on Linux) you will get a HUGE amount
# of debug logs
logging.getLogger("matplotlib").setLevel(logging.WARNING)
plt = import_extra("matplotlib.pyplot", extra="matplotlib")

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("PngImagePlugin").setLevel(logging.WARNING)


def main():
    # We need the factory calibration for our camera in order to decode the spectral information from the encoded view.
    # A simlink to a default is stored here if we are on an NVIDIA Jetson AGX Orin
    # Look here /datastore/lo/share/calibration/CAMERA_SERIAL_NUMBER/... you can see f# and  focal length in mm
    calibration_folder = os.path.join(getdatastorepath(), "lo", "share", "calibrations", "latest_calibration")
    decoder = SpectralDecoder.from_calibration(calibration_folder)

    # Set up a place to save data to
    data_folder = os.path.join(getdatastorepath(), "lo", "share", "data", "tutorial_3")
    os.makedirs(data_folder, exist_ok=True)
    current_time = timestamp()
    filename = f"decoded-data-tutorial-3-{current_time}.lo"
    decoded_filepath = os.path.join(data_folder, filename)
    print(f"Saving to {decoded_filepath}")

    # Connect to the camera - the 'with' context will automatically open and close the camera when we enter/exit it.
    with LOCamera() as cam:
        with open(decoded_filepath, mode="w", format="lo") as decoded_f:
            # Set camera settings - these will need to change depending on light levels and application
            cam.frame_rate = int(5120e3)  # μhz
            cam.exposure = int(1950e3)  # μs
            cam.gain = 0

            fig, axs = plt.subplots(1, 3)
            fig.suptitle("Living Optics Camera")
            axs[0].set_title("Scene view")
            axs[1].set_title("Decoded Spectra")
            axs[2].set_title("Flipped RGB Scene view")
            axs[1].set_xlabel("Wavelength (nm)")
            axs[1].set_ylabel("Spectral Radiance")

            # Iterate over 10 frames and save them to filepath
            for i, raw_frame in enumerate(cam):
                processed_frame = decoder(
                    raw_frame, scene_decoder=LORAWtoLOGRAY12, description="Tutorial 3 - " "Decoded" "Data Basics"
                )
                decoded_f.write(processed_frame)

                rgb_scene = LORAWtoRGB8(raw_frame[1][1])  # Pass the raw scene view to the RGB converter method

                metadata, scene_frame, spectra = processed_frame

                axs[0].imshow(processed_frame[1] / np.max(scene_frame))
                axs[1].plot(metadata.wavelengths, spectra.T)
                axs[2].imshow(np.flipud(rgb_scene / np.max(rgb_scene)))
                plt.tight_layout()
                plt.pause(0.2)

                if i >= 10:
                    break

    with open(decoded_filepath, "r") as f:
        # Iterate over each saved frame
        # Each frame in the .lo format is a tuple of (metadata, scene image, spectra array)
        for metadata, scene_frame, spectra in f:
            for v in vars(metadata):
                print(v, getattr(metadata, v))


if __name__ == "__main__":
    main()
