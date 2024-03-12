import logging
import os

import numpy as np
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
    # Find the living Optics Datastore
    data_folder = os.path.join(getdatastorepath(), "lo", "share", "data", "tutorial_2")

    # Create a directory to save the data into
    os.makedirs(data_folder, exist_ok=True)

    # Name a file for saving the .loraw format data to
    current_time = timestamp()
    raw_filename = f"raw-data-tutorial-2-{current_time}.loraw"
    raw_filepath = os.path.join(data_folder, raw_filename)
    print(f"Saving raw data to {raw_filepath}")

    with LOCamera() as cam:
        # Set camera settings - these will need to change depending on light levels and application
        cam.frame_rate = int(5000e3)  # μhz
        cam.exposure = int(195e3)  # μs
        cam.gain = 0

        fig, axs = plt.subplots(1, 2)
        fig.suptitle("Living Optics Camera")
        axs[0].set_title("Scene view")
        axs[1].set_title("Encoded view")
        with open(raw_filepath, mode="w", format="loraw") as raw_f:
            for i, frame in enumerate(cam):
                (encoded_metadata, encoded_frame), (scene_metadata, scene_frame) = frame
                raw_f.write(frame)

                axs[0].imshow(scene_frame / np.max(scene_frame))
                axs[1].imshow(encoded_frame / np.max(encoded_frame))
                plt.pause(0.2)

                # Leave the 'with' context after capturing 10 frames.
                if i >= 10:
                    break

    # Read the saved data
    # Read the saved data using the `open` method
    print("reading file {} with the Living Optics `open` method".format(raw_filepath))
    with open(raw_filepath, mode="r") as f:
        for frame in f:
            # Each frame in the saved file is a tuple of tuples.
            # The first tuple in each frame is the metadata and image date for the encoded view.
            # The second tuple in each frame is the metadata and image data for the scene view.
            (encoded_metadata, encoded_frame), (scene_metadata, scene_frame) = frame

            # Explore some of the metadata saved in the loraw format.
            print(f"{encoded_metadata.vd.timestamp.tv_usec=}")
            print(f"{scene_metadata.vd.timestamp.tv_usec=}")
            print(f"{encoded_frame.mean()=}")
            print(f"{scene_frame.mean()=}")

    # Read the saved data using the `LOCamera`
    print("reading file {} with the LOCamera".format(raw_filepath))
    with open(raw_filepath, mode="r") as open_file:
        with LOCamera(file=open_file) as cam:
            for frame in cam:
                # Each frame in the saved file is a tuple of tuples.
                # The first tuple in each frame is the metadata and image date for the encoded view.
                # The second tuple in each frame is the metadata and image data for the scene view.
                (encoded_metadata, encoded_frame), (scene_metadata, scene_frame) = frame

                # Explore some of the metadata saved in the loraw format.
                print(f"{encoded_metadata.vd.timestamp.tv_usec=}")
                print(f"{scene_metadata.vd.timestamp.tv_usec=}")
                print(f"{encoded_frame.mean()=}")
                print(f"{scene_frame.mean()=}")


if __name__ == "__main__":
    main()
