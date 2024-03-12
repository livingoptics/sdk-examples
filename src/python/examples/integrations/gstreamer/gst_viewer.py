#!/usr/bin/python3

# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

"""
# GStreamer viewer

An example viewer using Python and GStreamer.


Installation:
    Installed as part of the sdk.
    Custom installations may require you to install lo-sdk extras:
    - `pip install {wheel}['gstreamer']`


Example:
    ```python
        from lo.sdk.api.camera.camera import LOCamera
        from lo.sdk.helpers.platform import isjetsonagx

        # get an image viewer for each sensor
        iv0 = LOGstImageView(output_shape=(640, 480))
        iv1 = LOGstImageView(output_shape=(640, 480))

        # Simulated Camera unless on jetson
        f = None if isjetsonagx() else ""
        with LOCamera(file=f) as cam:
            # change some settings using camera level api
            cam.frame_rate = 20000000
            cam.gain = 100
            cam.exposure = 33334
            print(f"{cam.frame_rate=}, {cam.exposure=}, {cam.gain=}")

            for (encoded_info, encoded_frame), (scene_info, scene_frame) in cam:
                # put frames on the image queue
                iv0.put(encoded_frame)
                iv1.put(scene_frame)
    ```
"""

import logging
import time
from typing import Tuple

import gi
from lo.sdk.helpers._import import import_extra

gi.require_version("Gst", "1.0")
import gi.repository.GLib as GLib
import gi.repository.Gst as Gst

_logger = logging.getLogger(__name__)


class LOGstImageView:
    """
    GStreamer image viewer
    """

    def __init__(self, input_shape: Tuple = None, output_shape: Tuple = None, input_format: str = "GRAY16_LE"):
        """Class for displaying image data using gstreamer.

        Is an example of pushing data to an appsink for further processing.
        Running the equivilent of this gst-launch script
        ``` bash
        appsrc name=src block=True emit-signals=true is_live=True ! queue ! capsfilter ! {input_caps} ! videoconvert ! videoscale ! {output_caps} ! autovideosink sync=False
        ```
        where `input_shape`,`output_shape` and `caps` are determined by parameters or inferred by the frame passed to `put()`.

        Args:
            input_shape (Tuple, optional): (height,width). Defaults to None which estimates the shape from the input data.
            output_shape (Tuple, optional): (height,width). Defaults to None which displays at the same resoultion as the input.
            input_format (str, optional): Gstreamer format which matches the format of the np array passed. Defaults to "GRAY16_LE".
        """

        self.input_shape = input_shape

        # Set up the gstreamer pipeline
        self._pipeline = None
        Gst.init(None)
        self._mainloop = GLib.MainLoop()
        self.input_format = input_format
        self.output_shape = output_shape

    def build_pipeline(self):
        # Set input and output caps
        height, width = self.input_shape[0:2]
        output_caps = f"video/x-raw,width={self.output_shape[0]},height={self.output_shape[1]}"
        input_caps = f"video/x-raw,format={self.input_format},width={width},height={height}, framerate= 60/1"

        # Build and start pipeline
        command = f"appsrc name=src block=True emit-signals=true is_live=True ! queue ! capsfilter ! {input_caps} ! videoconvert ! videoscale ! {output_caps} ! autovideosink sync=False"
        self._pipeline = Gst.parse_launch(command)
        self.bus = self._pipeline.get_bus()
        self.appsrc = self._pipeline.get_by_name("src")
        assert self._pipeline.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.FAILURE

        # give the pipeline time to start up, otherwise we will read as not running and exit
        time.sleep(0.1)

    def window_closed(self):
        # gst posts error messages to the bus when the window exits.
        # wait 5000 ns for messages to make sure we never miss one.
        msg = self.bus.timed_pop_filtered(5000, Gst.MessageType.ERROR)
        if msg:
            t = msg.type
            if t == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                if "Output window was closed" in err.message:
                    logging.info("Output window was closed")
                    self._pipeline.set_state(Gst.State.NULL)
                    return True
        return False

    def put(self, frame):
        """Add a frame to the pipeline for display

        Args:
            frame (np.ndarray): Frame to display dtype must match the "input_format" specified on class instanciation.
        """

        # Startup the pipeline if we haven't
        # We do this here so we can infer input_shape from the frame
        if self._pipeline is None:
            if self.input_shape is None:
                self.input_shape = frame.shape
            if self.output_shape is None:
                self.output_shape = self.input_shape
            self.build_pipeline()
            gst_buffer = Gst.Buffer.new_wrapped(frame.tobytes())
            self.appsrc.emit("push-buffer", gst_buffer)

        # bail out if window is closed
        if self.window_closed():
            return

        # Only try and push data onto the pipeline if it is running.
        if self._pipeline.get_state(Gst.State.PLAYING).state == Gst.State.PLAYING:
            gst_buffer = Gst.Buffer.new_wrapped(frame.tobytes())
            self.appsrc.emit("push-buffer", gst_buffer)

    @property
    def running(self):
        return self._pipeline.get_state(Gst.State.PLAYING).state == Gst.State.PLAYING

    def stop(self):
        assert self._pipeline.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.FAILURE


def main():
    """main entry point"""

    from lo.sdk.api.camera.camera import LOCamera
    from lo.sdk.helpers.platform import isjetsonagx

    # get an image viewer for each sensor
    iv0 = LOGstImageView(output_shape=(640, 480))
    iv1 = LOGstImageView(output_shape=(640, 480))

    # Simulated Camera unless on jetson
    f = None if isjetsonagx() else ""
    with LOCamera(file=f) as cam:
        # change some settings using camera level api
        cam.frame_rate = 20000000
        cam.gain = 100
        cam.exposure = 33334
        print(f"{cam.frame_rate=}, {cam.exposure=}, {cam.gain=}")

        for (encoded_info, encoded_frame), (scene_info, scene_frame) in cam:
            # put frames on the image queue
            iv0.put(encoded_frame)
            iv1.put(scene_frame)

            if not (iv0.running or iv1.running):
                break


if __name__ == "__main__":
    main()
