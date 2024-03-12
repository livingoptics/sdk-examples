from lo.sdk.api.camera.camera import LOCamera


def main():
    # Connect to the camera - the 'with' context will automatically open and close the camera when we enter/exit it.
    with LOCamera() as cam:
        # Set camera settings - these will need to change depending on light levels and application
        cam.frame_rate = int(5120e3)  # μhz
        cam.exposure = int(1950e3)  # μs
        cam.gain = 0

        for i, frame in enumerate(cam):
            (encoded_metadata, encoded_frame), (scene_metadata, scene_frame) = frame
            if i >= 10:
                break


if __name__ == "__main__":
    main()
