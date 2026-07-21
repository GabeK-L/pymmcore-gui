from __future__ import annotations

from typing import TYPE_CHECKING

import ndv
from ndv.models import RingBuffer

from pymmcore_gui._qt.QtWidgets import QApplication, QVBoxLayout, QWidget
from pymmcore_gui.widgets.image_preview._preview_base import ImagePreviewBase

if TYPE_CHECKING:
    import numpy as np
    from pymmcore_plus import CMMCorePlus


# Live preview only needs the most recent frame.
BUFFER_SIZE = 1


# Camera display orientation
# (2, 1) corrects a camera physically mounted 90 degrees
DISPLAY_AXES = (2, 1)


# Mirror correction:
# None = no flip
# 0 = flip vertically
# 1 = flip horizontally
#
# Try changing this if the image is mirrored.
FLIP_AXIS = 0
# FLIP_AXIS = 0
# FLIP_AXIS = 1


class NDVPreview(ImagePreviewBase):
    def __init__(
        self,
        mmcore: CMMCorePlus,
        parent: QWidget | None = None,
        *,
        use_with_mda: bool = False,
    ):
        super().__init__(
            parent,
            mmcore,
            use_with_mda=use_with_mda,
        )

        self._viewer = ndv.ArrayViewer()
        self._buffer: RingBuffer | None = None
        self._core_dtype: tuple[str, tuple[int, ...]] | None = None
        self._is_rgb: bool = False
        self.process_events_on_update = True

        qwdg = self._viewer.widget()
        qwdg.setParent(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(qwdg)

    def append(self, data: np.ndarray) -> None:
        needs_setup = self._buffer is None

        if needs_setup:
            self._init_buffer()

        if self._buffer is not None:

            # Correct mirror orientation if needed
            if FLIP_AXIS is not None:
                data = np.flip(data, axis=FLIP_AXIS)
                data = np.ascontiguousarray(data)

            self._buffer.append(data)

            if needs_setup:
                self._apply_viewer_settings()

            self._viewer.display_model.current_index.update(
                {0: len(self._buffer) - 1}
            )

            self._viewer.data_wrapper.data_changed.emit()

            if self.process_events_on_update:
                QApplication.process_events()

    @property
    def dtype_shape(self) -> tuple[str, tuple[int, ...]] | None:
        return self._core_dtype

    def _get_core_dtype_shape(self) -> tuple[str, tuple[int, ...]] | None:
        if (core := self._mmc) is not None:

            if bits := core.getImageBitDepth():

                img_width = core.getImageWidth()
                img_height = core.getImageHeight()

                if core.getNumberOfComponents() > 1:
                    shape = (img_height, img_width, 3)
                else:
                    shape = (img_height, img_width)

                # Convert packed bits to byte-aligned numpy dtype
                if bits <= 8:
                    bits = 8
                elif bits <= 16:
                    bits = 16
                elif bits <= 32:
                    bits = 32

                return (f"uint{bits}", shape)

        return None

    def _init_buffer(self) -> None:
        if (core_dtype := self._get_core_dtype_shape()) is None:
            return

        self._core_dtype = core_dtype

        # RGB images have height, width, channels
        self._is_rgb = len(core_dtype[1]) == 3

        self._buffer = RingBuffer(
            max_capacity=BUFFER_SIZE,
            dtype=core_dtype,
        )

    def _apply_viewer_settings(self) -> None:
        self._viewer.data = self._buffer

        # Correct 90 degree camera orientation
        self._viewer.display_model.visible_axes = DISPLAY_AXES

        if self._is_rgb:
            self._viewer.display_model.channel_axis = 3
            self._viewer.display_model.channel_mode = (
                ndv.models.ChannelMode.RGBA
            )
        else:
            self._viewer.display_model.channel_mode = (
                ndv.models.ChannelMode.GRAYSCALE
            )
            self._viewer.display_model.channel_axis = None

    def _setup_viewer(self) -> None:
        # Recreate buffer after camera ROI changes
        self._buffer = None
        self._core_dtype = None

        self._init_buffer()

        if self._buffer is not None:
            self._apply_viewer_settings()

    def _on_system_config_loaded(self) -> None:
        self._setup_viewer()

    def _on_roi_set(self) -> None:
        self._setup_viewer()
