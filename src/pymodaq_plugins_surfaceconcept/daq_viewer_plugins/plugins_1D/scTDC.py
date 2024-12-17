# -*- coding: utf-8 -*-

# Copyright 2018-2022 Surface Concept GmbH
#
# This file is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This file is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this file.  If not, see <https://www.gnu.org/licenses/>.
#
# ----------------------------------------------------------------------------
#
# This module uses docstrings formatted for the Sphinx/autodoc module which
# helps with the generation of documentation in html form and pdf form. The
# documentation contains an API reference and offers better readability.
#
# ----------------------------------------------------------------------------
# Changes:
# 2019-09-19 : added structures for USER_CALLBACKS pipe
# 2019-10-15 : higher-level Device and Pipe classes
# 2020-05-14 : added wrapper for a separate library to save DLD events to HDF5
# version 1.2.0 : added support for BUFFERED_DATA_CALLBACKS pipe
#                 (requires scTDC1 library version >= 1.3010.0)
# version 1.3.0 : added camera support, added Sphinx-based documentation,
#                 removed HDF5 related classes and functions
# version 1.4.0 : added camera blob mode support, see example_camera_blobs.py

__version__ = "1.4.0"

import ctypes
import os
import time
import traceback
try: # pipes and buffered data callbacks do require numpy
    import numpy as np
except:
    pass

# pipe types
TDC_HISTO         = 0 #: pipe type, TDC time histogram for one TDC channel
DLD_IMAGE_XY      = 1 #: pipe type, image mapping the detector area
DLD_IMAGE_XT      = 2 #: pipe type, image mapping the detector x axis and the TDC time axis
DLD_IMAGE_YT      = 3 #: pipe type, image mapping the detector y axis and the TDC time axis
DLD_IMAGE_3D      = 4 #: pipe type, 3D matrix mapping the detector area the the TDC time axis
DLD_SUM_HISTO     = 5 #: pipe type, 1D histogram for DLDs, counts vs time axis
STATISTICS        = 6 #: pipe type, statistics data delivered at the end of measurements
TMSTAMP_TDC_HISTO = 7 #: pipe type, rarely used and currently undocumented
TDC_STATISTICS    = 8 #: pipe type, rarely used and currently undocumented
DLD_STATISTICS    = 9 #: pipe type, rarely used and currently undocumented
USER_CALLBACKS    = 10 #: pipe type, TDC and DLD event data, slow in python
DLD_IMAGE_XY_EXT  = 11 #: pipe type, currently not supported in Python
BUFFERED_DATA_CALLBACKS = 12 #: pipe type, TDC and DLD event data, more efficient variant of USER_CALLBACKS
PIPE_CAM_FRAMES = 13 #: pipe type, provides camera frame raw image data and frame meta data
PIPE_CAM_BLOBS = 14  #: pipe type, provides camera blob coordinates

# bitsizes for depth parameter in
#  sc_pipe_dld_image_xyt_params_t  and  sc_pipe_tdc_histo_params_t
BS8   = 0 #: pixel data format, unsigned 8-bit integer
BS16  = 1 #: pixel data format, unsigned 16-bit integer
BS32  = 2 #: pixel data format, unsigned 32-bit integer
BS64  = 3 #: pixel data format, unsigned 64-bit integer
BS_FLOAT32 = 4 #: pixel data format, single-precision floating point number
BS_FLOAT64 = 5 #: pixel data format, double-precision floating point number

# callback reasons for end-of-measurement callback in conjunction with
# sc_tdc_set_complete_callback2
CBR_COMPLETE    = 1 #: callback reason, regular completion of measurement
CBR_USER_ABORT  = 2 #: callback reason, user aborted the measurement
CBR_BUFFER_FULL = 3 #: callback reason, measurement stopped due to full buffer
CBR_EARLY_NOTIF = 4 #: callback reason, device idle but PC-side processing not finished, yet
CBR_DICT = {CBR_COMPLETE : "Measurement and data processing completed.",
    CBR_USER_ABORT : "Measurement was interrupted by user.",
    CBR_BUFFER_FULL : "Measurement was aborted because buffers were full.",
    CBR_EARLY_NOTIF : "Acquisition finished, not all data processed yet."}

# enum sc_data_field_t
SC_DATA_FIELD_SUBDEVICE          = 0x0001 #: used in buffered_data_callbacks_pipe
SC_DATA_FIELD_CHANNEL            = 0x0002 #: used in buffered_data_callbacks_pipe
SC_DATA_FIELD_START_COUNTER      = 0x0004 #: used in buffered_data_callbacks_pipe
SC_DATA_FIELD_TIME_TAG           = 0x0008 #: used in buffered_data_callbacks_pipe
SC_DATA_FIELD_DIF1               = 0x0010 #: used in buffered_data_callbacks_pipe
SC_DATA_FIELD_DIF2               = 0x0020 #: used in buffered_data_callbacks_pipe
SC_DATA_FIELD_TIME               = 0x0040 #: used in buffered_data_callbacks_pipe
SC_DATA_FIELD_MASTER_RST_COUNTER = 0x0080 #: used in buffered_data_callbacks_pipe
SC_DATA_FIELD_ADC                = 0x0100 #: used in buffered_data_callbacks_pipe
SC_DATA_FIELD_SIGNAL1BIT         = 0x0200 #: used in buffered_data_callbacks_pipe

_FUNCTYPE = None
if os.name == 'nt':
    _FUNCTYPE = ctypes.WINFUNCTYPE
else:
    _FUNCTYPE = ctypes.CFUNCTYPE

class sc3du_t(ctypes.Structure):
    _fields_ = [("x",ctypes.c_uint),
                ("y",ctypes.c_uint),
                ("time", ctypes.c_uint64)]

class sc3d_t(ctypes.Structure):
    _fields_ = [("x",ctypes.c_int),
                ("y",ctypes.c_int),
                ("time", ctypes.c_int64)]

class roi_t(ctypes.Structure):
    _fields_ = [("offset", sc3d_t),
                ("size", sc3du_t)]

ALLOCATORFUNC = _FUNCTYPE(ctypes.c_int, ctypes.POINTER(None),
                          ctypes.POINTER(ctypes.POINTER(None)))

class sc_pipe_dld_image_xyt_params_t(ctypes.Structure):
    _fields_ = [("depth",    ctypes.c_int),
                ("channel",  ctypes.c_int),
                ("modulo",   ctypes.c_uint64),
                ("binning",  sc3du_t),
                ("roi",      roi_t),
                ("accumulation_ms", ctypes.c_uint),
                ("allocator_owner", ctypes.c_char_p),
                ("allocator_cb",    ALLOCATORFUNC)]

class sc_pipe_tdc_histo_params_t(ctypes.Structure):
    _fields_ = [("depth",     ctypes.c_int),
                ("channel",   ctypes.c_uint),
                ("modulo",    ctypes.c_uint64),
                ("binning",   ctypes.c_uint),
                ("offset",    ctypes.c_uint64),
                ("size",      ctypes.c_uint),
                ("accumulation_ms", ctypes.c_uint),
                ("allocator_owner", ctypes.c_char_p),
                ("allocator_cb", ALLOCATORFUNC)]

class sc_pipe_statistics_params_t(ctypes.Structure):
    _fields_ = [("allocator_owner", ctypes.c_char_p),
                ("allocator_cb", ALLOCATORFUNC)]

class statistics_t(ctypes.Structure):
    _fields_ = [("counts_read", ctypes.c_uint * 64),
                ("counts_received", ctypes.c_uint * 64),
                ("events_found", ctypes.c_uint * 4),
                ("events_in_roi", ctypes.c_uint * 4),
                ("events_received", ctypes.c_uint * 4),
                ("counters", ctypes.c_uint * 64),
                ("reserved", ctypes.c_uint * 52)]

class tdc_event_t(ctypes.Structure):
    _fields_ = [("subdevice",     ctypes.c_uint),
                ("channel",       ctypes.c_uint),
                ("start_counter", ctypes.c_ulonglong),
                ("time_tag",      ctypes.c_ulonglong),
                ("time_data",     ctypes.c_ulonglong),
                ("sign_counter",  ctypes.c_ulonglong)]

class dld_event_t(ctypes.Structure):
    _fields_ = [("start_counter",      ctypes.c_ulonglong),
                ("time_tag",           ctypes.c_ulonglong),
                ("subdevice",          ctypes.c_uint),
                ("channel",            ctypes.c_uint),
                ("sum",                ctypes.c_ulonglong),
                ("dif1",               ctypes.c_ushort),
                ("dif2",               ctypes.c_ushort),
                ("master_rst_counter", ctypes.c_uint),
                ("adc",                ctypes.c_ushort),
                ("signal1bit",         ctypes.c_ushort)]

class sc_pipe_buf_callback_args(ctypes.Structure):
    _fields_ = [("event_index",        ctypes.c_ulonglong),
                ("som_indices",        ctypes.POINTER(ctypes.c_ulonglong)),
                ("ms_indices",         ctypes.POINTER(ctypes.c_ulonglong)),
                ("subdevice",          ctypes.POINTER(ctypes.c_uint)),
                ("channel",            ctypes.POINTER(ctypes.c_uint)),
                ("start_counter",      ctypes.POINTER(ctypes.c_ulonglong)),
                ("time_tag",           ctypes.POINTER(ctypes.c_uint)),
                ("dif1",               ctypes.POINTER(ctypes.c_uint)),
                ("dif2",               ctypes.POINTER(ctypes.c_uint)),
                ("time",               ctypes.POINTER(ctypes.c_ulonglong)),
                ("master_rst_counter", ctypes.POINTER(ctypes.c_uint)),
                ("adc",                ctypes.POINTER(ctypes.c_int)),
                ("signal1bit",         ctypes.POINTER(ctypes.c_ushort)),
                ("som_indices_len",    ctypes.c_uint),
                ("ms_indices_len",     ctypes.c_uint),
                ("data_len",           ctypes.c_uint),
                ("reserved",           ctypes.c_char * 12)]

### ----    user callbacks   ---------------------------------------------------
CB_STARTMEAS = _FUNCTYPE(None, ctypes.POINTER(None))
CB_ENDMEAS = CB_STARTMEAS
CB_MILLISEC = CB_STARTMEAS
CB_STATISTICS = _FUNCTYPE(None, ctypes.POINTER(None),
                          ctypes.POINTER(statistics_t))
CB_TDCEVENT = _FUNCTYPE(None, ctypes.POINTER(None),
                        ctypes.POINTER(tdc_event_t), ctypes.c_size_t)
CB_DLDEVENT = _FUNCTYPE(None, ctypes.POINTER(None),
                        ctypes.POINTER(dld_event_t), ctypes.c_size_t)
### ----    measurement complete callback   ------------------------------------
CB_COMPLETE = _FUNCTYPE(None, ctypes.c_void_p, ctypes.c_int)
### ----    callbacks for buffered_data_callbacks_pipe   -----------------------
CB_BUFDATA_DATA = _FUNCTYPE(None, ctypes.c_void_p,
                            ctypes.POINTER(sc_pipe_buf_callback_args))
CB_BUFDATA_END_OF_MEAS = _FUNCTYPE(ctypes.c_bool, ctypes.c_void_p)
### ----------------------------------------------------------------------------

class sc_pipe_callbacks(ctypes.Structure):
    _fields_ = [("priv",                ctypes.POINTER(None)),
                ("start_of_measure",    CB_STARTMEAS),
                ("end_of_measure",      CB_ENDMEAS),
                ("millisecond_countup", CB_MILLISEC),
                ("statistics",          CB_STATISTICS),
                ("tdc_event",           CB_TDCEVENT),
                ("dld_event",           CB_DLDEVENT)]

class sc_pipe_callback_params_t(ctypes.Structure):
    _fields_ = [("callbacks", ctypes.POINTER(sc_pipe_callbacks))]


class sc_pipe_buf_callbacks_params_t(ctypes.Structure):
  _fields_ = [("priv",                  ctypes.POINTER(None)),
              ("data",                  CB_BUFDATA_DATA),
              ("end_of_measurement",    CB_BUFDATA_END_OF_MEAS),
              ("data_field_selection",  ctypes.c_uint),
              ("max_buffered_data_len", ctypes.c_uint),
              ("dld_events",            ctypes.c_int),
              ("version",               ctypes.c_int),
              ("reserved",              ctypes.c_ubyte * 24)]

class sc_cam_frame_meta_t(ctypes.Structure):
    _fields_ = [
        ("data_offset",   ctypes.c_uint),
        ("frame_idx",     ctypes.c_uint),
        ("frame_time",    ctypes.c_uint64),
        ("width",         ctypes.c_uint16),
        ("height",        ctypes.c_uint16),
        ("roi_offset_x",  ctypes.c_uint16),
        ("roi_offset_y",  ctypes.c_uint16),
        ("adc",           ctypes.c_uint16),
        ("pixelformat",   ctypes.c_uint8),
        ("flags",         ctypes.c_uint8),
        ("reserved",      ctypes.c_uint8 * 4)]

class sc_cam_blob_meta_t(ctypes.Structure):
    _fields_ = [
        ("data_offset", ctypes.c_uint),
        ("nr_blobs",    ctypes.c_uint)
    ]

class sc_cam_blob_position_t(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float)
    ]

# values used in sc_cam_frame_meta_t.pixelformat
SC_CAM_PIXELFORMAT_UINT8 = 0
SC_CAM_PIXELFORMAT_UINT16 = 1

# flags used in sc_cam_frame_meta_t.flags
SC_CAM_FRAME_HAS_IMAGE_DATA = 1
SC_CAM_FRAME_IS_LAST_FRAME = 2

class ShutterMode:
    """Defines the values that represent the available shutter modes"""
    START_AND_STOP_BY_WIRE = 0          #:
    START_AND_STOP_BY_SOFTWARE = 1      #:
    START_BY_WIRE_STOP_BY_SOFTWARE = 2  #:
    START_BY_SOFTWARE_STOP_BY_WIRE = 3  #:
    to_str = {
        START_AND_STOP_BY_WIRE : "Start and stop by wire",
        START_AND_STOP_BY_SOFTWARE : "Start and stop by software",
        START_BY_WIRE_STOP_BY_SOFTWARE : "Start by wire, stop by software",
        START_BY_SOFTWARE_STOP_BY_WIRE : "Start by software, stop by wire"
    }
    from_str = {
        "Start and stop by wire" : START_AND_STOP_BY_WIRE,
        "Start and stop by software" : START_AND_STOP_BY_SOFTWARE,
        "Start by wire, stop by software" : START_BY_WIRE_STOP_BY_SOFTWARE,
        "Start by software, stop by wire" : START_BY_SOFTWARE_STOP_BY_WIRE
    }

def copy_statistics(s):
    assert(type(s)==statistics_t)
    r = statistics_t()
    ctypes.memmove(ctypes.byref(r), ctypes.byref(s), ctypes.sizeof(s))
    return r

def version_at_least(v, vmin):
    """check whether version v is at least version vmin

    :param v: the version to be checked, must be a tuple with 3 elements
    :type v: tuple
    :param vmin: the version that defines the minimum requirement, must be a
      tuple with 3 elements
    :type vmin: tuple
    :return: True if v is at least vmin
    :rtype: bool
    """
    return v[0] > vmin[0] or \
        (v[0] == vmin[0] and v[1] > vmin[1]) or \
            (v[0] == vmin[0] and v[1] == vmin[1] and v[2] >= vmin[2])

class scTDClib:
    """low-level wrapper of the C interface in the dynamically loaded library
    scTDC"""
    def __init__(self, libfilepath=None):
        """loads the library scTDC (scTDC1.dll or libscTDC.so) from hard disk
        and adds the correct signatures to the library functions so they can be
        used from Python.

        :param libfilepath: optionally specify the full path including file name
            to the shared library file, defaults to None
        :type libfilepath: str
        """
        if os.name == 'nt':
            if libfilepath is None:
                try:
                    self.lib = ctypes.WinDLL("scTDC1.dll")
                except OSError:
                    self.lib = ctypes.WinDLL(r".\scTDC1.dll")
                    # if exception happens here, let it traverse to the user
            else:
                self.lib = ctypes.WinDLL(libfilepath)
        else:
            if libfilepath is None:
                self.lib = ctypes.CDLL("libscTDC.so.1")
            else:
                self.lib = ctypes.CDLL(libfilepath)
        i32 = ctypes.c_int
        i32ptr = ctypes.POINTER(i32)
        u32 = ctypes.c_uint
        u32ptr = ctypes.POINTER(u32)
        size_t = ctypes.c_size_t
        sizeptr = ctypes.POINTER(size_t)
        charptr = ctypes.c_char_p
        voidptr = ctypes.c_void_p
        voidptrptr = ctypes.POINTER(ctypes.c_void_p)
        self.lib.sc_tdc_init_inifile.argtypes = [charptr]
        self.lib.sc_tdc_init_inifile.restype = i32
        self.lib.sc_get_err_msg.argtypes = [i32, charptr]
        self.lib.sc_get_err_msg.restype = None
        self.lib.sc_tdc_deinit2.argtypes = [i32]
        self.lib.sc_tdc_deinit2.restype = i32
        self.lib.sc_tdc_start_measure2.argtypes = [i32, i32]
        self.lib.sc_tdc_start_measure2.restype = i32
        self.lib.sc_tdc_interrupt2.argtypes = [i32]
        self.lib.sc_tdc_interrupt2.restype = i32
        self.lib.sc_pipe_open2.argtypes = [i32, i32, voidptr]
        self.lib.sc_pipe_open2.restype = i32
        self.lib.sc_pipe_close2.argtypes = [i32, i32]
        self.lib.sc_pipe_close2.restype = i32
        self.lib.sc_tdc_get_status2.argtypes = [i32, i32ptr]
        self.lib.sc_tdc_get_status2.restype = i32
        self.lib.sc_pipe_read2.argtypes = [i32, i32, voidptrptr, u32]
        self.lib.sc_pipe_read2.restype = i32
        self.lib.sc_tdc_get_statistics2.argtypes = [
            i32, ctypes.POINTER(statistics_t)]
        self.lib.sc_tdc_set_complete_callback2.argtypes = [
            i32, voidptr, CB_COMPLETE]
        self.lib.sc_tdc_set_complete_callback2.restype = i32
        self.lib.sc_tdc_config_get_library_version.argtypes = [u32 * 3]
        self.lib.sc_tdc_config_get_library_version.restype = None
        self.libversion = self.sc_tdc_config_get_library_version()
        self.lib.sc_tdc_zero_master_reset_counter.argtypes = [i32]
        self.lib.sc_tdc_zero_master_reset_counter.restype = i32
        self.lib.sc_tdc_get_device_properties.argtypes = [i32, i32, voidptr]
        self.lib.sc_tdc_get_device_properties.restype = i32
        if version_at_least(self.libversion, (1, 3013, 13)):
            self.lib.sc_tdc_get_device_properties2.argtypes = [i32, i32, voidptr]
            self.lib.sc_tdc_get_device_properties2.restype = i32
        if version_at_least(self.libversion, (1, 3017, 5)):
            # camera support
            self.lib.sc_tdc_set_blob.argtypes = [i32, charptr]
            self.lib.sc_tdc_set_blob.restype = i32
            self.lib.sc_tdc_get_blob.argtypes = [i32, charptr, size_t, sizeptr]
            self.lib.sc_tdc_get_blob.restype = i32
            self.lib.sc_tdc_cam_set_roi.argtypes = [i32, u32, u32, u32, u32]
            self.lib.sc_tdc_cam_set_roi.restype = i32
            self.lib.sc_tdc_cam_get_roi.argtypes = [
                i32, u32ptr, u32ptr, u32ptr, u32ptr]
            self.lib.sc_tdc_cam_get_roi.restype = i32
            self.lib.sc_tdc_cam_set_exposure.argtypes = [i32, u32, u32]
            self.lib.sc_tdc_cam_set_exposure.restype = i32
            self.lib.sc_tdc_cam_get_maxsize.argtypes = [i32, u32ptr, u32ptr]
            self.lib.sc_tdc_cam_get_maxsize.restype = i32
            self.lib.sc_tdc_cam_set_fanspeed.argtypes = [i32, i32]
            self.lib.sc_tdc_cam_set_fanspeed.restype = i32
            self.lib.sc_tdc_cam_set_parameter.argtypes = [i32, charptr, charptr]
            self.lib.sc_tdc_cam_set_parameter.restype = i32
            self.lib.sc_tdc_cam_get_parameter.argtypes = [
                i32, charptr, charptr, sizeptr]
            self.lib.sc_tdc_cam_get_parameter.restype = i32
            # override registry support
            self.lib.sc_tdc_overrides_create.argtypes = []
            self.lib.sc_tdc_overrides_create.restype = i32
            self.lib.sc_tdc_overrides_close.argtypes = [i32]
            self.lib.sc_tdc_overrides_close.restype = i32
            self.lib.sc_tdc_overrides_add_entry.argtypes = [
                i32, charptr, charptr, charptr]
            self.lib.sc_tdc_overrides_add_entry.restype = i32
            self.lib.sc_tdc_init_inifile_override.argtypes = [charptr, i32]
            self.lib.sc_tdc_init_inifile_override.restype = i32

    def sc_tdc_init_inifile(self, inifile_path="tdc_gpx3.ini"):
        """Initializes the hardware and loads the initial settings from the
        specified ini file.

        :param inifile_path: the name of or full path to the configuration file,
          defaults to "tdc_gpx3.ini"
        :type inifile_path: str
        :return: Returns a non-negative device descriptor on success or a
          negative error code in case of failure. The device descriptor is needed
          for all functions that involve the initialized device
        :rtype: int
        """
        return self.lib.sc_tdc_init_inifile(inifile_path.encode('utf-8'))

    def sc_tdc_init_inifile_overrides(self, inifile_path="tdc_gpx3.ini",
        overrides = None):
        """Initializes the hardware and loads the initial settings from the
        specified ini file. Enables overriding of parameters from the ini file
        without modification of the ini file on hard disk (the override
        entries reside in memory and are evaluated by the scTDC library).

        :param inifile_path: the name of or full path to the configuration file,
          defaults to "tdc_gpx3.ini"
        :type inifile_path: str
        :param overrides: a list of 3-tuples (section_name, parameter_name,
          parameter_value), where the section_name is specified without square
          brackets ([]). Spelling of names is case sensitive.
          defaults to [] (empty list)
        :type overrides: list
        :return: a non-negative device descriptor on success, or, a negative
          error code in case of failure. The device descriptor is needed for all
          functions that involve the initialized device
        :rtype: int
        """
        if overrides is None or len(overrides) == 0:
            return self.sc_tdc_init_inifile(inifile_path)
        ovr = self.lib.sc_tdc_overrides_create()
        if ovr < 0:
            return ovr
        try:
            for entry in overrides:
                assert len(entry)==3
                conv = [x.encode('utf-8') for x in entry]
                self.lib.sc_tdc_overrides_add_entry(ovr, *conv)
            return self.lib.sc_tdc_init_inifile_override(
                inifile_path.encode('utf-8'), ovr)
        finally:
            self.lib.sc_tdc_overrides_close(ovr)
            # this does not overwrite the return from the try clause with a None
            # and, in case of an exception, there is no return value but an
            # exception

    def sc_get_err_msg(self, errcode):
        """Returns an error message to the given error code.

        :param errcode: a negative error code returned by one of the library
          functions
        :type errcode: int
        :return: the error message describing the reason of the error code
        :rtype: str
        """
        if errcode>=0:
            return ""
        sbuf = ctypes.create_string_buffer(1024)
        self.lib.sc_get_err_msg(errcode, sbuf)
        if type(sbuf.value)==type(b''):
            return sbuf.value.decode('utf-8')
        else:
            return sbuf.value

    def sc_tdc_config_get_library_version(self):
        """Query the version of the scTDC library.

        :return: a 3-tuple containing the version separated into major, minor
          and patch parts, e.g. version 1.3017.5 becomes (1, 3017, 5)
        :rtype: tuple
        """
        libversion = (ctypes.c_uint * 3)()
        self.lib.sc_tdc_config_get_library_version(libversion)
        return tuple(libversion)

    def sc_tdc_deinit2(self, dev_desc):
        """Deinitialize the hardware.

        :param dev_desc: device descriptor as retrieved from
          :any:`sc_tdc_init_inifile` or :any:`sc_tdc_init_inifile_overrides`
        :type dev_desc: int
        :return: 0 on success or negative error code
        :rtype: int
        """
        return self.lib.sc_tdc_deinit2(dev_desc)

    def sc_tdc_start_measure2(self, dev_desc, exposure_ms):
        """Start a measurement (asynchronously/non-blocking)

        :param dev_desc: device descriptor as returned by one of the
          initialization functions
        :type dev_desc: int
        :param exposure_ms: The exposure time in milliseconds
        :type exposure_ms: int
        :return: 0 on success or negative error code
        :rtype: int
        """
        return self.lib.sc_tdc_start_measure2(dev_desc, exposure_ms)

    def sc_tdc_interrupt2(self, dev_desc):
        """Interrupts a measurement asynchronously (non-blocking).
        Asynchronously means, the function may return before the device actually
        reaches idle state. :any:`sc_tdc_set_complete_callback2` may be used to
        be notified when the device has stopped the measurement.

        :param dev_desc: device descriptor
        :type dev_desc: int
        :return: 0 on success or negative error code
        :rtype: int
        """
        return self.lib.sc_tdc_interrupt2(dev_desc)

    def sc_pipe_open2(self, dev_desc, pipe_type, pipe_params):
        """Open a pipe for reading data from the device. The available pipe
        types with their corresponding pipe_params types are

        * :py:const:`TDC_HISTO` : :py:class:`sc_pipe_tdc_histo_params_t`
        * :py:const:`DLD_IMAGE_XY` : :py:class:`sc_pipe_dld_image_xyt_params_t`
        * :py:const:`DLD_IMAGE_XT` : :py:class:`sc_pipe_dld_image_xyt_params_t`
        * :py:const:`DLD_IMAGE_YT` : :py:class:`sc_pipe_dld_image_xyt_params_t`
        * :py:const:`DLD_IMAGE_3D` : :py:class:`sc_pipe_dld_image_xyt_params_t`
        * :py:const:`DLD_SUM_HISTO` : :py:class:`sc_pipe_dld_image_xyt_params_t`
        * :py:const:`STATISTICS` : :py:class:`sc_pipe_statistics_params_t`
        * :py:const:`USER_CALLBACKS` : :py:class:`sc_pipe_callback_params_t`
        * :py:const:`BUFFERED_DATA_CALLBACKS` : :py:class:`sc_pipe_buf_callbacks_params_t`
        * :py:const:`PIPE_CAM_FRAMES` : None (no configuration parameters)
        * :py:const:`PIPE_CAM_BLOBS` : None (no configuration parameters)

        :param dev_desc: device descriptor
        :type dev_desc: int
        :param pipe_type: one of the pipe type constants
        :type pipe_type: int
        :param pipe_params: various types of structures depending on pipe_type.
          If a structure is needed, it should be passed by value, not by pointer.
        :type pipe_params: Any
        :return: a non-negative pipe handle on success or a negative error code
        :rtype: int
        """
        assert (pipe_type==DLD_IMAGE_XY and
                isinstance(pipe_params, sc_pipe_dld_image_xyt_params_t)) \
            or (pipe_type==DLD_IMAGE_XT and
                isinstance(pipe_params, sc_pipe_dld_image_xyt_params_t)) \
            or (pipe_type==DLD_IMAGE_YT and
                isinstance(pipe_params, sc_pipe_dld_image_xyt_params_t)) \
            or (pipe_type==DLD_IMAGE_3D and
                isinstance(pipe_params, sc_pipe_dld_image_xyt_params_t)) \
            or (pipe_type==DLD_SUM_HISTO and
                isinstance(pipe_params, sc_pipe_dld_image_xyt_params_t)) \
            or (pipe_type==TDC_HISTO and
                isinstance(pipe_params, sc_pipe_tdc_histo_params_t)) \
            or (pipe_type==STATISTICS and
                isinstance(pipe_params, sc_pipe_statistics_params_t)) \
            or (pipe_type==USER_CALLBACKS and
                isinstance(pipe_params, sc_pipe_callback_params_t)) \
            or (pipe_type==BUFFERED_DATA_CALLBACKS and
                isinstance(pipe_params, sc_pipe_buf_callbacks_params_t)) \
            or (pipe_type==PIPE_CAM_FRAMES and pipe_params is None) \
            or (pipe_type==PIPE_CAM_BLOBS and pipe_params is None)
        if pipe_params is None:
            return self.lib.sc_pipe_open2(dev_desc, pipe_type, None)
        else:
            return self.lib.sc_pipe_open2(dev_desc, pipe_type,
                                        ctypes.addressof(pipe_params))

    def sc_pipe_close2(self, dev_desc, pipe_handle):
        """Close a pipe.

        :param dev_desc: device descriptor
        :type dev_desc: int
        :param pipe_handle: the pipe handle as returned by :any:`sc_pipe_open2`
        :type pipe_handle: int
        :return: 0 on success or negative error code
        :rtype: int
        """
        return self.lib.sc_pipe_close2(dev_desc, pipe_handle)

    def sc_pipe_read2(self, dev_desc, pipe_handle, timeout):
        """Read from a pipe. The functions waits until either data is available
        or the timeout is reached.

        :param dev_desc: device descriptor
        :type dev_desc: int
        :param pipe_handle: pipe handle as returned by :any:`sc_pipe_open2`
        :type pipe_handle: int
        :param timeout: the timeout in milliseconds
        :type timeout: int
        :return: a tuple containing the return code and a ctypes.POINTER to
          the data buffer
        :rtype: tuple
        """
        bufptr = ctypes.POINTER(None)()
        retcode = self.lib.sc_pipe_read2(dev_desc, pipe_handle,
                                         ctypes.byref(bufptr), timeout)
        return (retcode, bufptr)


    def sc_tdc_get_status2(self, dev_desc):
        """Query whether the device is idle or in measurement.

        :param dev_desc: device descriptor
        :type dev_desc: int
        :return: 0 (idle) or 1 (exposure) or negative error code
        :rtype: int
        """
        statuscode = ctypes.c_int()
        #statuscodeptr = ctypes.POINTER(ctypes.c_int)(statuscode)
        retcode = self.lib.sc_tdc_get_status2(dev_desc,
                                              ctypes.byref(statuscode))
        if retcode < 0:
            return retcode
        else:
            return 1 if statuscode.value==0 else 0

    def sc_tdc_get_statistics2(self, dev_desc):
        """This function is deprecated. Use the statistics pipe, instead.
        This function is kept for older scTDC library versions."""
        stat1 = statistics_t()
        retcode = self.lib.sc_tdc_get_statistics2(dev_desc,
                                                  ctypes.byref(stat1))
        if retcode < 0:
          return retcode
        else:
          return stat1

    def sc_tdc_set_complete_callback2(self, dev_desc, privptr, callback):
        """Sets a callback to be notified about completed measurements or other
        events regarding the transition from measurement state to idle state.

        :param dev_desc: device descriptor
        :type dev_desc: int
        :param privptr: a private pointer that is passed back into the callback
        :type privptr: ctypes.POINTER(void)
        :param callback: the function to be called for notifications
        :type callback: function
        :return: 0 on success or negative error code
        :rtype: int
        """
        return self.lib.sc_tdc_set_complete_callback2(dev_desc, privptr,
                                                      callback)

class buffered_data_callbacks_pipe(object):
    """ Base class for using the ``BUFFERED_DATA_CALLBACKS`` interface which
    provides DLD or TDC events in a list-of-events form.
    Requires scTDC1 library version >= 1.3010.0.
    In comparison to the ``USER_CALLBACKS`` pipe, this pipe reduces the number
    of callbacks into python, buffering a higher number of events within the
    library before invoking the callbacks. Thereby, the number of Python lines
    of code that need to be executed can be drastically reduced if you stick to
    numpy vector operations rather than iterating through the events one by one.
    The :any:`on_data` callback receives a dictionary containing 1D numpy arrays
    where the size of these arrays can be as large as specified by the
    max_buffered_data_len parameter. To use this interface, write a class that
    derives from this class and override the methods

    * :any:`on_data`
    * :any:`on_end_of_meas`
    """
    def __init__(self,
                 lib,
                 dev_desc,
                 data_field_selection=SC_DATA_FIELD_TIME,
                 max_buffered_data_len=(1<<16),
                 dld_events=True):
        """Creates the pipe which will be immediately active until closed.
        Requires an already initialized device.

        :param lib: an :any:`scTDClib` object
        :type lib: :any:`scTDClib`
        :param dev_desc: device descriptor as returned by
          :any:`sc_tdc_init_inifile` or :any:`sc_tdc_init_inifile_overrides`
        :type dev_desc: int
        :param data_field_selection: a 'bitwise or' combination of
          SC_DATA_FIELD_xyz constants, defaults to :py:const:`SC_DATA_FIELD_TIME`
        :type data_field_selection: int
        :param max_buffered_data_len: The number of events that are buffered
          before invoking the on_data callback. Less events can also be received
          in the on_data callback, when the user chooses to return True from
          the on_end_of_meas callback. defaults to (1<<16)
        :type max_buffered_data_len: int
        :param dld_events: if True, receive DLD events. If False, receive TDC
          events. Depending on the configuration in the tdc_gpx3.ini file, only
          one type of events may be available. defaults to True
        :type dld_events: bool
        """
        self.dev_desc = dev_desc
        self.lib = lib
        self._pipe_desc = None
        self._open_pipe(data_field_selection, max_buffered_data_len,
                        dld_events)

    def _open_pipe(self, data_field_selection, max_buffered_data_len,
                   dld_events):
        p = sc_pipe_buf_callbacks_params_t()
        p.priv = None
        self._cb_data = CB_BUFDATA_DATA(lambda x, y : self._data_cb(y))
        self._cb_eom = CB_BUFDATA_END_OF_MEAS(lambda x : self.on_end_of_meas())
        p.data = self._cb_data
        p.end_of_measurement = self._cb_eom
        p.data_field_selection = data_field_selection
        p.max_buffered_data_len = max_buffered_data_len
        p.dld_events = 1 if dld_events else 0
        p.version = 0
        reservedlist = [0]*24
        p.reserved = (ctypes.c_ubyte * 24)(*reservedlist)
        self._pipe_args = p # prevent garbage collection!
        self._pipe_desc = self.lib.sc_pipe_open2(
            self.dev_desc, BUFFERED_DATA_CALLBACKS, p)

    def _data_cb(self, dptr):
        d = dptr.contents
        x = {"event_index" : d.event_index, "data_len" : d.data_len}
        f = np.ctypeslib.as_array
        if d.subdevice:
            x["subdevice"] = f(d.subdevice, shape=(d.data_len,))
        if d.channel:
            x["channel"] = f(d.channel, shape=(d.data_len,))
        if d.start_counter:
            x["start_counter"] = f(d.start_counter, shape=(d.data_len,))
        if d.time_tag:
            x["time_tag"] = f(d.time_tag, shape=(d.data_len,))
        if d.dif1:
            x["dif1"] = f(d.dif1, shape=(d.data_len,))
        if d.dif2:
            x["dif2"] = f(d.dif2, shape=(d.data_len,))
        if d.time:
            x["time"] = f(d.time, shape=(d.data_len,))
        if d.master_rst_counter:
            x["master_rst_counter"] = f(d.master_rst_counter, shape=(d.data_len,))
        if d.adc:
            x["adc"] = f(d.adc, shape=(d.data_len,))
        if d.signal1bit:
            x["signal1bit"] = f(d.signal1bit, shape=(d.data_len,))
        if d.som_indices:
            x["som_indices"] = f(d.som_indices, shape=(d.som_indices_len,))
        if d.ms_indices:
            x["ms_indices"] = f(d.ms_indices, shape=(d.ms_indices_len,))
        self.on_data(x)

    def on_data(self, data):
        """Override this method to process the data.

        :param data: A dictionary containing several numpy arrays. The selection
          of arrays depends on the data_field_selection value used during
          initialization of the class. The following key names are always
          present in this dictionary:

          * event_index
          * data_len

          Keywords related to regular event data are:

          * subdevice
          * channel
          * start_counter
          * time_tag
          * dif1
          * dif2
          * time
          * master_rst_counter
          * adc
          * signal1bit

          Keywords related to indexing arrays are:

          * som_indices (start of a measurement)
          * ms_indices (millisecond tick as tracked by the hardware)

          These contain event indices that mark the occurence of what is described
          in parentheses in the above list.
        :return: None
        """
        pass

    def on_end_of_meas(self):
        """Override this method to trigger actions at the end of the measurement.
        Do not call methods that start the next measurement from this callback.
        This cannot succeed. Use a signalling mechanism into your main thread,
        instead.

        :return: True indicates that the pipe should transfer the remaining
          buffered events immediately after returning from this callback.
          False indicates that the pipe may continue buffering the next
          measurements until the max_buffered_data_len threshold is reached.
        :rtype: bool
        """
        return True # True signalizes that all buffered data shall be emitted

    def close(self):
        """Close the pipe.
        """
        self.lib.sc_pipe_close2(self.dev_desc, self._pipe_desc)

    def start_measurement_sync(self, time_ms):
        """Start a measurement and wait until it is finished.

        :param time_ms: the duration of the measurement in milliseconds.
        :type time_ms: int
        :return: 0 on success or a negative error code.
        :rtype: int
        """
        retcode = self.lib.sc_tdc_start_measure2(self.dev_desc, time_ms)
        if retcode < 0:
            return retcode
        time.sleep(time_ms/1000.0) # sleep expects floating point seconds
        while self.lib.sc_tdc_get_status2(self.dev_desc) == 1:
            time.sleep(0.01)
        return 0

    def start_measurement(self, time_ms, retries=3):
        """Start a measurement 'in the background', i.e. don't wait for it
        to finish.

        :param time_ms: the duration of the measurement in milliseconds.
        :type time_ms: int
        :param retries: in an asynchronous scheme of measurement sequences,
          trying to start the next measurement can occasionally result in a
          "NOT READY" error. Often some thread of the scTDC1 library just
          needs a few more cycles to reach the "idle" state again, where
          the start of the next measurement will be accepted.
          The retries parameter specifies how many retries with 0.001 s
          sleeps in between will be made before giving up, defaults to 3
        :type retries: int
        :return: 0 on success or a negative error code.
        :rtype: int
        """
        while True:
            retcode = self.lib.sc_tdc_start_measure2(self.dev_desc, time_ms)
            if retcode != -11: # "not ready" error
                return retcode
            retries -= 1
            if retries <= 0:
                return -11
            time.sleep(0.001)

class usercallbacks_pipe(object):
    """ Base class for user implementations of the "USER_CALLBACKS" interface.
    Derive from this class and override some or all of the methods

    *  on_start_of_meas
    *  on_end_of_meas
    *  on_millisecond
    *  on_statistics
    *  on_tdc_event
    *  on_dld_event

    The lib argument in the constructor expects a scTDClib object.
    The dev_desc argument in the constructor expects the device descriptor
    as returned by sc_tdc_init_inifile(...)."""
    def __init__(self, lib, dev_desc):
        self.dev_desc = dev_desc
        self.lib = lib
        self._pipe_desc = None
        self._open_pipe()

    def _open_pipe(self):
        p = sc_pipe_callbacks()
        p.priv = None
        p.start_of_measure = CB_STARTMEAS(lambda x : self.on_start_of_meas())
        p.end_of_measure = CB_ENDMEAS(lambda x : self.on_end_of_meas())
        p.millisecond_countup = CB_MILLISEC(lambda x : self.on_millisecond())
        p.tdc_event = CB_TDCEVENT(lambda x, y, z : self.on_tdc_event(y, z))
        p.dld_event = CB_DLDEVENT(lambda x, y, z : self.on_dld_event(y, z))
        p.statistics = CB_STATISTICS(lambda x, y : self.on_statistics(y))
        self.struct_callbacks = p
        p2 = sc_pipe_callback_params_t()
        p2.callbacks = ctypes.pointer(self.struct_callbacks)
        self._pipe_args = p
        self._pipe_args2 = p2
        self._pipe_desc = self.lib.sc_pipe_open2(self.dev_desc, USER_CALLBACKS,
                                                 p2)

    def do_measurement(self, time_ms):
        self.lib.sc_tdc_start_measure2(self.dev_desc, time_ms)
        time.sleep(time_ms/1000.0) # sleep expects floating point seconds
        while self.lib.sc_tdc_get_status2(self.dev_desc) == 1:
            time.sleep(0.01)

    def on_start_of_meas(self):
        pass

    def on_end_of_meas(self):
        pass

    def on_millisecond(self):
        pass

    def on_statistics(self, stats):
        pass

    def on_tdc_event(self, tdc_events, nr_tdc_events):
        pass

    def on_dld_event(self, dld_events, nr_dld_events):
        pass

    def close(self):
        self.lib.sc_pipe_close2(self.dev_desc, self._pipe_desc)



def _get_voxel_type(depth):
    if depth==BS8:
        return ctypes.c_uint8
    elif depth==BS16:
        return ctypes.c_uint16
    elif depth==BS32:
        return ctypes.c_uint32
    elif depth==BS64:
        return ctypes.c_uint64
    elif depth==BS_FLOAT32:
        return ctypes.c_float
    elif depth==BS_FLOAT64:
        return ctypes.c_double
    else:
        return -1

# * 0x1 start counter, 0x2 time tag, 0x4 subdevice, 0x8 channel,
# * 0x10 time since start pulse ("sum"), 0x20 "x" detector coordinate ("dif1"),
# * 0x40 "y" detector coordinate ("dif2"), 0x80 master reset counter,
# * 0x100 ADC value, 0x200 signal bit. If this function is not called, the


class Device(object):
    """A higher-level interface for TDCs and DLDs for applications that use
    pre-computed histograms from the scTDC library."""
    def __init__(self, inifilepath="tdc_gpx3.ini", autoinit=True, lib=None):
        """Creates a device object.

        :param inifilepath: the name of or full path to the configuration/ini
          file, defaults to "tdc_gpx3.ini"
        :type inifilepath: str, optional
        :param autoinit: if True, initialize the hardware immediately, defaults
          to True
        :type autoinit: bool, optional
        :param lib: if not None, reuse the specified scTDClib object, else the
          Device class creates its own scTDClib object internally, defaults
          to None
        :type lib: :py:class:`scTDClib`, optional
        """
        self.inifilepath = inifilepath
        self.dev_desc = None
        self.pipes = {}
        self.eomcb = {} # end of measurement callbacks
        if lib is None:
            self.lib = scTDClib()
        else:
            self.lib = lib
        if autoinit:
            self.initialize()

    def initialize(self):
        """Initialize the hardware.

        :return: a tuple containing an error code and a human-readable error
          message (zero and empty string in case of success)
        :rtype: tuple(int, str)
        """
        retcode = self.lib.sc_tdc_init_inifile(self.inifilepath)
        if retcode < 0:
            return (retcode, self.lib.sc_get_err_msg(retcode))
        else:
            self.dev_desc = retcode
            # register end of measurement callback
            if not hasattr(self, "_eomcbfobj"):
                def _eomcb(privptr, reason):
                    for i in self.eomcb.keys():
                        self.eomcb[i](reason)
                self._eomcbfobj = CB_COMPLETE(_eomcb) # extend lifetime!
            ret2 = self.lib.sc_tdc_set_complete_callback2(self.dev_desc, None,
                                                   self._eomcbfobj)
            if ret2 < 0:
                print("Registering measurement-complete callback failed")
                print(" message:", self.lib.sc_get_err_msg(ret2))
            return (0, "")

    def deinitialize(self):
        """Deinitialize the hardware.

        :return: a tuple containing an error code and a human-readable error
          message (zero and empty string in case of success)
        :rtype: tuple(int, str)
        """
        if self.dev_desc is None or self.dev_desc < 0:
            return (0, "") # don't argue if there is nothing to do
        retcode = self.lib.sc_tdc_deinit2(self.dev_desc)
        if retcode < 0:
            return (retcode, self.lib.sc_get_err_msg(retcode))
        else:
            try:
                pipekeys = [x for x in self.pipes.keys()]
                for p in pipekeys:
                    del self.pipes[p]
            except:
                traceback.print_exc()
                traceback.print_stack()
            self.dev_desc = None
            return (0, "")

    def is_initialized(self):
        """Query whether the device is initialized

        :return: True if the device is initialized
        :rtype: bool
        """
        return self.dev_desc is not None

    def do_measurement(self, time_ms=100, synchronous=False):
        """Start a measurement.

        :param time_ms: the measurement time in milliseconds, defaults to 100
        :type time_ms: int, optional
        :param synchronous: if True, block until the measurement has finished.
          defaults to False.
        :type synchronous: bool, optional
        :return: a tuple (0, "") in case of success, or a negative error code
          and a string with the error message
        :rtype: tuple(int, str)
        """
        retcode = self.lib.sc_tdc_start_measure2(self.dev_desc, time_ms)
        if retcode < 0:
            return (retcode, self.lib.sc_get_err_msg(retcode))
        else:
            if synchronous:
                time.sleep(time_ms/1000.0)
                while self.lib.sc_tdc_get_status2(self.dev_desc) == 1:
                    time.sleep(0.01)
            return (0, "")

    def interrupt_measurement(self):
        """Interrupt a measurement that was started with synchronous=False.

        :return: a tuple (0, "") in case of success, or a negative error code
          and a string with the error message
        :rtype: tuple(int, str)
        """
        retcode = self.lib.sc_tdc_interrupt2(self.dev_desc)
        return (retcode, self.lib.sc_get_err_msg(retcode))

    def add_end_of_measurement_callback(self, cb):
        """Adds a callback function for the end of measurement. The callback
        function needs to accept one ``int`` argument which indicates the reason
        for the callback. Notification via callback is useful if you want to use
        do_measurement(...) with synchronous=False, for example in GUIs that
        need to be responsive during measurement.

        :param cb: the callback function
        :type cb: Callable
        :return: non-negative ID of the callback (for later removal)
        :rtype: int
        """
        for i in range(len(self.eomcb),-1,-1):
            if not i in self.eomcb:
                self.eomcb[i] = cb
                return i
        return -1

    def remove_end_of_measurement_callback(self, id_of_cb):
        """Removes a previously added callback function for the end of
        measurement.

        :param id_of_cb: the ID as previously returned by
          :any:`add_end_of_measurement_callback`.
        :type id_of_cb: int
        :return: 0 on success, -1 if the id_of_cb is unknown
        :rtype: int
        """
        if id_of_cb in self.eomcb:
            del self.eomcb[id_of_cb]
            return 0
        else:
            return -1

    def _make_new_pipe(self, typestr, par, parent):
        for i in range(1000):
            if i not in self.pipes:
                self.pipes[i] = Pipe(typestr, par, parent)
                return (i,self.pipes[i])
        return None

    def _add_img_pipe_impl(self, depth, modulo, binning, roi, typestr):
        par = sc_pipe_dld_image_xyt_params_t()
        par.depth = depth
        par.channel = -1
        par.modulo = modulo
        par.binning.x = binning[0]
        par.binning.y = binning[1]
        par.binning.time = binning[2]
        par.roi.offset.x = roi[0][0]
        par.roi.offset.y = roi[1][0]
        par.roi.offset.time = roi[2][0]
        par.roi.size.x = roi[0][1]
        par.roi.size.y = roi[1][1]
        par.roi.size.time = roi[2][1]
        par.accumulation_ms = 1 << 31
        pipe = self._make_new_pipe(typestr, par, self)
        if pipe is None:
            return (-1, "Too many pipes open")
        else:
            return pipe # return id and object

    def add_3d_pipe(self, depth, modulo, binning, roi):
        """Adds a 3D pipe (x, y, time) with static buffer. The 3D buffer
        retrieved upon reading is organized such that a point (x, y, time_slice)
        is addressed by x + y * size_x + time_slice * size_x * size_y. When
        getting a numpy array view/copy of the buffer, the 'F' (Fortran)
        indexing order can be chosen, such that the indices are intuitively
        ordered as x, y, time.

        :param depth: one of BS8, BS16, BS32, BS64, BS_FLOAT32, BS_FLOAT64
        :type depth: int
        :param modulo: If 0, no effect. If > 0, a module operation is applied to
          the time values of events before sorting events into the 3D buffer.
          The unit of the module value is the time bin divided by 32, i.e.
          a modulo value of 32 corresponds to one time bin.
        :type modulo: int
        :param binning: a 3-tuple specifying the binning in x, y, and time,
          where all values need to be a power of 2.
        :type binning: (int,int,int)
        :param roi: a 3-tuple of (offset, size) pairs specifying the ranges along
          the x, y, and time axes.
        :type roi: ((int, int), (int, int), (int, int))
        :return: a tuple containing a non-negative pipe ID and the Pipe object
          in case of success. A tuple containing the negative error code and the
          error message in case of failure.
        :rtype: tuple(int, Pipe) | tuple(int, str)
        """
        return self._add_img_pipe_impl(depth, modulo, binning, roi,
                                       typestr="3d")

    def add_xy_pipe(self, depth, modulo, binning, roi):
        """Adds a 2D pipe (x,y) with static buffer. The 2D buffer retrieved upon
        reading is organized such that a point (x,y) is addressed by
        x + y * size_x. When getting a numpy array view/copy of the buffer, the
        'F' (Fortran) indexing order can be chosen, such that the indices are
        intuitively ordered x, y. The binning in time has an influence only on
        the time units in the roi. The time part in the roi specifies the
        integration range, such that only events inside this time range are
        inserted into the data buffer.

        :param depth: one of BS8, BS16, BS32, BS64, BS_FLOAT32, BS_FLOAT64
        :type depth: int
        :param modulo: If 0, no effect. If > 0, a module operation is applied to
          the time values of events before sorting events into the 2D buffer.
          The unit of the module value is the time bin divided by 32, i.e.
          a modulo value of 32 corresponds to one time bin.
        :type modulo: int
        :param binning: a 3-tuple specifying the binning in x, y, and time,
          where all values need to be a power of 2.
        :type binning: (int,int,int)
        :param roi: a 3-tuple of (offset, size) pairs specifying the ranges along
          the x, y, and time axes.
        :type roi: ((int, int), (int, int), (int, int))
        :return: a tuple containing a non-negative pipe ID and the Pipe object
          in case of success. A tuple containing the negative error code and the
          error message in case of failure.
        :rtype: tuple(int, Pipe) | tuple(int, str)
        """
        return self._add_img_pipe_impl(depth, modulo, binning, roi,
                                       typestr="xy")

    def add_xt_pipe(self, depth, modulo, binning, roi):
        """Adds a 2D pipe (x,t) with static buffer. The 2D buffer retrieved upon
        reading is organized such that a point (x,time) is addressed by
        x + time * size_x. When getting a numpy array view/copy of the buffer,
        the 'F' (Fortran) indexing order can be chosen, such that the indices
        are intuitively ordered x, time. The binning in y has an influence only
        on the y units in the roi. The y part in the roi specifies the
        integration range, such that only events inside this y range are
        inserted into the data buffer.

        :param depth: one of BS8, BS16, BS32, BS64, BS_FLOAT32, BS_FLOAT64
        :type depth: int
        :param modulo: If 0, no effect. If > 0, a module operation is applied to
          the time values of events before sorting events into the 2D buffer.
          The unit of the module value is the time bin divided by 32, i.e.
          a modulo value of 32 corresponds to one time bin.
        :type modulo: int
        :param binning: a 3-tuple specifying the binning in x, y, and time,
          where all values need to be a power of 2.
        :type binning: (int,int,int)
        :param roi: a 3-tuple of (offset, size) pairs specifying the ranges along
          the x, y, and time axes.
        :type roi: ((int, int), (int, int), (int, int))
        :return: a tuple containing a non-negative pipe ID and the Pipe object
          in case of success. A tuple containing the negative error code and the
          error message in case of failure.
        :rtype: tuple(int, Pipe) | tuple(int, str)
        """
        return self._add_img_pipe_impl(depth, modulo, binning, roi,
                                       typestr="xt")

    def add_yt_pipe(self, depth, modulo, binning, roi):
        """Adds a 2D pipe (y,t) with static buffer. The 2D buffer retrieved upon
        reading is organized such that a point (y,time) is addressed by
        y + time * size_y. When getting a numpy array view/copy of the buffer,
        the 'F' (Fortran) indexing order can be chosen, such that the indices
        are intuitively ordered y, time. The binning in x has an influence only
        on the x units in the roi. The x part in the roi specifies the
        integration range, such that only events inside this x range are
        inserted into the data buffer.

        :param depth: one of BS8, BS16, BS32, BS64, BS_FLOAT32, BS_FLOAT64
        :type depth: int
        :param modulo: If 0, no effect. If > 0, a module operation is applied to
          the time values of events before sorting events into the 2D buffer.
          The unit of the module value is the time bin divided by 32, i.e.
          a modulo value of 32 corresponds to one time bin.
        :type modulo: int
        :param binning: a 3-tuple specifying the binning in x, y, and time,
          where all values need to be a power of 2.
        :type binning: (int,int,int)
        :param roi: a 3-tuple of (offset, size) pairs specifying the ranges along
          the x, y, and time axes.
        :type roi: ((int, int), (int, int), (int, int))
        :return: a tuple containing a non-negative pipe ID and the Pipe object
          in case of success. A tuple containing the negative error code and the
          error message in case of failure.
        :rtype: tuple(int, Pipe) | tuple(int, str)
        """
        return self._add_img_pipe_impl(depth, modulo, binning, roi,
                                       typestr="yt")

    def add_t_pipe(self, depth, modulo, binning, roi):
        """Adds a 1D time histogram pipe, integrated over a rectangular region
        in the (x,y) plane (for delay-line detectors) with static buffer.
        The buffer received upon reading is a 1D array of the
        intensity values for all resolved time bins. The binning in x and y has
        an influence only on the x and y units in the roi. The x and y parts in
        the roi specify the integration ranges, such that only events inside
        the x and y ranges are inserted into the data buffer.

        :param depth: one of BS8, BS16, BS32, BS64, BS_FLOAT32, BS_FLOAT64
        :type depth: int
        :param modulo: If 0, no effect. If > 0, a module operation is applied to
          the time values of events before sorting events into the 1D array.
          The unit of the module value is the time bin divided by 32, i.e.
          a modulo value of 32 corresponds to one time bin.
        :type modulo: int
        :param binning: a 3-tuple specifying the binning in x, y, and time,
          where all values need to be a power of 2.
        :type binning: (int,int,int)
        :param roi: a 3-tuple of (offset, size) pairs specifying the ranges along
          the x, y, and time axes.
        :type roi: ((int, int), (int, int), (int, int))
        :return: a tuple containing a non-negative pipe ID and the Pipe object
          in case of success. A tuple containing the negative error code and the
          error message in case of failure.
        :rtype: tuple(int, Pipe) | tuple(int, str)
        """
        return self._add_img_pipe_impl(depth, modulo, binning, roi,
                                       typestr="t")

    def add_statistics_pipe(self):
        """Adds a pipe for statistics data (sometimes referred to as rate
        meters). The statistics data is only updated at the end of each
        measurement.

        :return: a tuple containing a non-negative pipe ID and the Pipe object
          in case of success. A tuple containing the negative error code and the
          error message in case of failure.
        :rtype: tuple(int, Pipe) | tuple(int, str)
        """
        par = sc_pipe_statistics_params_t()
        pipe = self._make_new_pipe("stat", par, self)
        if pipe is None:
            return (-1, "Too many pipes open")
        else:
            return pipe # return id and object

    def add_tdc_histo_pipe(self, depth, channel, modulo, binning, offset,
                           size):
        """Adds a pipe for time histograms from a stand-alone TDC. TDC events
        are filtered by the specified channel, and their time values are
        transformed first by modulo, then by binning, then by subtraction of the
        offset, and finally, by clipping to the size value. The resulting value
        (if not clipped) is the array index of the histogram which is
        incremented by one.

        :param depth: one of BS8, BS16, BS32, BS64
        :type depth: int
        :param channel: selects the TDC channel
        :type channel: int
        :param modulo: If 0, no effect. If > 0, a modulo operation is applied to
          the time values of TDC events before sorting them into the 1D array.
          The unit of the module value is the time bin divided by 32, i.e.
          a modulo value of 32 corresponds to one time bin.
        :type modulo: int
        :param binning: divides the time value by the specified binning before
          sorting into the 1D array. Must be a power of 2. Binning 1 is
          equivalent to no binning.
        :type binning: int
        :param offset: The offset / lower boundary of the accepted range on the
          time axis.
        :type offset: int
        :param size: The size / length of the accepted range on the time axis.
          The size is also directly the number of entries in the array/histogram
        :type size: int
        :return: a tuple containing a non-negative pipe ID and the Pipe object
          in case of success. A tuple containing the negative error code and the
          error message in case of failure.
        :rtype: tuple(int, Pipe) | tuple(int, str)
        """
        par = sc_pipe_tdc_histo_params_t()
        par.depth = depth
        par.channel = channel
        par.modulo = modulo
        par.binning = binning
        par.offset = offset
        par.size = size
        par.accumulation_ms = 1 << 31
        pipe = self._make_new_pipe("tdch", par, self)
        if pipe is None:
            return (-1, "Too many pipes open")
        else:
            return pipe # return id and object

    def remove_pipe(self, pipeid):
        """Remove a pipe. Manual removal of pipes may be unnecessary if you are
        using all created pipes until the deinitialization of the device.

        :param pipeid: the pipe ID as returned in the first element of the tuple
          by all add_XYZ_pipe functions
        :type pipeid: int
        :return: 0 on success, -1 if pipe id unknown or error
        :rtype: int
        """
        try:
            self.pipes[pipeid].close()
        except KeyError:
            return -1
        try:
            del self.pipes[pipeid]
        except KeyError:
            return -1
        return 0

class Pipe(object):
    """This class handles various types of data received from scTDC library
    pipes, such as

    * 1D, 2D, 3D histograms from DLD (detected particle) events
    * statistics data at the end of measurements
    * time histograms from stand-alone TDCs.

    Instantiation of this class should only happen through calls to the
    add_XYZ_pipe functions from the :any:`Device` class. Use methods of this
    class to access the data produced by the Pipe.
    """
    def __init__(self, typestr, par, parent):
        """Constructs a Pipe object. Creates the data buffer and opens the pipe
        in the scTDC library for the parent Device.

        :param typestr: one of '3d', 'xy', 'xt', 'yt', 't', 'stat'
        :type typestr: str
        :param par: pipe configuration parameters
        :type par: sc_pipe_dld_image_xyt_params_t | sc_pipe_statistics_params_t
        :param parent: a :any:`Device` object
        :type parent: :any:`Device`
        """
        self.par = par
        self.parent = parent
        self.typestr = typestr
        self.handle = None
        self.buf = None
        self.bufptr = None
        self.bufsize = None
        self.par.allocator_owner = None
        self.pipetypeconst = None
        # ---------------------------------------------------------------------
        # ---     statistics case     -----------------------------------------
        # ---------------------------------------------------------------------
        if self.typestr == 'stat':
            self.pipetypeconst = STATISTICS
            self.par.allocator_cb = self._get_stat_allocator()
            retcode, errmsg = self.reopen()
            if retcode < 0:
                print("scTDC.Pipe.__init__ : error during creation:\n"
                    + "  ({}) {}".format(errmsg, retcode))
            return
        # ---------------------------------------------------------------------
        self.nrvoxels = None
        self.voxeltype = _get_voxel_type(self.par.depth)
        if self.typestr == '3d':
            self.nrvoxels = par.roi.size.x * par.roi.size.y * par.roi.size.time
            self.pipetypeconst = DLD_IMAGE_3D
        elif self.typestr == 'xy':
            self.nrvoxels = par.roi.size.x * par.roi.size.y
            self.pipetypeconst = DLD_IMAGE_XY
        elif self.typestr == 'xt':
            self.nrvoxels = par.roi.size.x * par.roi.size.time
            self.pipetypeconst = DLD_IMAGE_XT
        elif self.typestr == 'yt':
            self.nrvoxels = par.roi.size.y * par.roi.size.time
            self.pipetypeconst = DLD_IMAGE_YT
        elif self.typestr == 't':
            self.nrvoxels = par.roi.size.time
            self.pipetypeconst = DLD_SUM_HISTO
        elif self.typestr == 'tdch':
            self.nrvoxels = par.size
            self.pipetypeconst = TDC_HISTO
        if self.nrvoxels is not None:
            self.par.allocator_cb = self._get_allocator(self.nrvoxels,
                                                        self.voxeltype)
            retcode, errmsg = self.reopen()
            if retcode < 0:
                print("scTDC.Pipe.__init__ : error during creation:\n"
                    + "  ({}) {}".format(errmsg, retcode))


    def _get_allocator(self, nrvoxels, voxeltype):
        if self.buf is not None and self.nrvoxels != nrvoxels:
            return None # already have a buffer, cannot change
        elif self.buf is None:
            self.buf = (voxeltype*nrvoxels)() # fixed-size array of voxeltype
            self.bufptr = ctypes.POINTER(type(self.buf))(self.buf)
            self.nrvoxels = nrvoxels
            self.bufsize = nrvoxels * ctypes.sizeof(voxeltype)
        if not hasattr(self, '_allocatorfunc'):
            def _allocator(privptr, bufptrptr):
                bufptrptr[0] = ctypes.cast(self.bufptr, ctypes.c_void_p)
                return 0
            self._allocatorfunc = ALLOCATORFUNC(_allocator)
        return self._allocatorfunc

    def _get_stat_allocator(self):
        if not hasattr(self, '_stat_allocfunc'):
            self.buf = statistics_t()
            self.bufptr = ctypes.POINTER(type(self.buf))(self.buf)
            self.bufsize = ctypes.sizeof(statistics_t)
            def _stat_alloc(privptr, bufptrptr):
                bufptrptr[0] = ctypes.cast(self.bufptr, ctypes.c_void_p)
                return 0
            self._stat_allocfunc = ALLOCATORFUNC(_stat_alloc)
        return self._stat_allocfunc

    def is_open(self):
        """Query whether the pipe is active / open.

        :return: True, if the pipe is active in the scTDC library (if so, the
          library writes to the data buffer during measurements and increments
          histogram entries on incoming events).
        :rtype: bool
        """
        return self.handle is not None

    def reopen(self, force=False):
        """Open a pipe with previous parameters, if not currently open.

        :param force: set this to True, if the pipe has not been explicitly
          closed, but the device was deinitialized, causing an implicit
          destruction of the pipe (implicit desctruction only happens through
          low-level API calls, whereas Device.deinitialize will close all pipe
          objects and delete references to them), defaults to False
        :type force: bool, optional
        :return: None if nothing to do, (0, "") on success,
          (error code, message) on failure
        :rtype: None | (int, str)
        """
        if self.handle is not None and not force:
            return
        retcode = self.parent.lib.sc_pipe_open2(
            self.parent.dev_desc, self.pipetypeconst, self.par)
        if retcode < 0:
            return (retcode, self.parent.lib.sc_get_err_msg(retcode))
        else:
            self.handle = retcode
            return (0, "")

    def close(self):
        """ Close the pipe such that no events are sorted into the data buffer
        anymore. The data buffer remains unchanged. In that sense, closing acts
        more like setting the pipe inactive the pipe can be reopened later. The
        data buffer can only be garbage-collected after deleting the pipe
        object via the parent device and discarding all other references to the
        Pipe object.

        :return: (0, "") if success, (error code, message) on failure
        :rtype: (int, str)
        """
        retcode = self.parent.lib.sc_pipe_close2(self.parent.dev_desc,
                                                 self.handle)
        if retcode < 0:
            return (retcode, self.parent.lib.sc_get_err_msg(retcode))
        else:
            self.handle = None
            return (0, "")

    def _reshape(self, a):
        if self.typestr=='3d':
            return np.reshape(a, (self.par.roi.size.x, self.par.roi.size.y,
                               self.par.roi.size.time), order='F')
        elif self.typestr=='xy':
            return np.reshape(a, (self.par.roi.size.x, self.par.roi.size.y),
                           order='F')
        elif self.typestr=='xt':
            return np.reshape(a, (self.par.roi.size.x, self.par.roi.size.time),
                           order='F')
        elif self.typestr=='yt':
            return np.reshape(a, (self.par.roi.size.y, self.par.roi.size.time),
                           order='F')
        elif self.typestr=='t' or self.typestr=='tdch':
            #return np.reshape(a, (self.par.roi.size.time,))
            return a # buffer is already 1D, needs no reshaping

    def get_buffer_view(self):
        """For 1D, 2D, 3D pipes, get a numpy array of the data buffer,
        constructed without copying. As a consequence, changes to the data
        buffer, made by the scTDC library after getting the buffer view, will
        be visible to the numpy array returned from this function.
        The indexing is in Fortran order, i.e. x, y, time.
        If the pipe is a statistics pipe, get the statistics_t object which
        may be modified subsequently by the scTDC.

        :return: a view of the static data buffer
        :rtype: numpy.ndarray | statistics_t
        """
        if self.typestr=='stat':
            return self.buf
        else:
            return self._reshape(np.ctypeslib.as_array(self.buf))

    def get_buffer_copy(self):
        """For 1D, 2D, 3D pipes, get a numpy array of a copy of the data
        buffer. The indexing is in Fortran order, i.e. x, y, time.
        If the pipe is a statistics pipe, return a copy of the statistics_t
        object.

        :return: a copy of the data buffer
        :rtype: numpy.ndarray | statistics_t
        """
        if self.typestr=='stat':
            return copy_statistics(self.buf)
        else:
            return self._reshape(np.array(self.buf, copy=True))

    def clear(self):
        """Set all voxels of the data buffer to zero"""
        ctypes.memset(self.buf, 0, self.bufsize)

class CamFramePipe(object):
    """A pipe for reading camera image frames and frame meta information
    synchronously. Do not instantiate this class by hand. Use
    Camera.add_frame_pipe, instead."""
    def __init__(self, device):
        self.dev_desc = device.dev_desc
        self.lib = device.lib
        ret = self.lib.sc_pipe_open2(
            self.dev_desc, PIPE_CAM_FRAMES, None)
        self.pipe_desc = ret if ret >= 0 else None

    def is_active(self):
        """Query whether the pipe is active / open.

        :return: True if the pipe is active.
        :rtype: bool
        """
        return self.pipe_desc is not None

    def close(self):
        """Close the pipe, release memory associated with the pipe"""
        if self.pipe_desc is not None:
            self.lib.sc_pipe_close2(self.dev_desc, self.pipe_desc)
            self.pipe_desc = None

    def read(self, timeout_ms = 500):
        """Wait until the next camera frame becomes available or timeout is
        reached. Return meta data and, if available, image data of the next
        camera frame. If image data is returned, access to it is only allowed
        until the next time that this read function is called. Perform a copy
        of the image data if you need to keep it for longer. As soon as a
        CamFramePipe is opened and measurements are started, the pipe allocates
        memory for storing the frame data until this frame data is read. Not
        reading the pipe frequently enough can exhaust the memory.

        :param timeout_ms: the timeout in milliseconds, defaults to 500
        :type timeout_ms: int, optional
        :return: Returns a tuple (meta, image_data) where meta is a dictionary
          containing the frame meta data, and image_data is a numpy array. If an
          error occurs, returns a tuple (error_code, error_message).
        :rtype: (dict, numpy.ndarray) | (int, str)
        """
        (ret, bufptr) = self.lib.sc_pipe_read2(
            self.dev_desc, self.pipe_desc, timeout_ms)
        if ret >= 0 and bufptr:
            dataptr = ctypes.cast(bufptr, ctypes.POINTER(sc_cam_frame_meta_t))
            d = dataptr.contents
            meta = dict((field,
                getattr(d, field)) for field, _ in d._fields_ \
                    if field not in (
                        'data_offset',
                        'pixelformat',
                        'reserved',
                        'flags'))
            meta['last_frame'] = bool(d.flags & SC_CAM_FRAME_IS_LAST_FRAME)
            if d.flags & SC_CAM_FRAME_HAS_IMAGE_DATA:
                voxeltype = ctypes.c_uint8
                if d.pixelformat == SC_CAM_PIXELFORMAT_UINT16:
                    voxeltype = ctypes.c_uint16
                imgptr = ctypes.cast(
                    bufptr.value + d.data_offset,
                    ctypes.POINTER(voxeltype))
                img = np.ctypeslib.as_array(imgptr,
                    shape=(d.height, d.width))
                return (meta, img)
            else:
                return (meta, None)
        else:
            return (ret, self.lib.sc_get_err_msg(ret))

class CamBlobsPipe(object):
    """A pipe for reading camera blob data synchronously. Do not instantiate
    this class by hand. Use Camera.add_blobs_pipe, instead."""
    def __init__(self, device):
        self.dev_desc = device.dev_desc
        self.lib = device.lib
        ret = self.lib.sc_pipe_open2(
            self.dev_desc, PIPE_CAM_BLOBS, None)
        self.pipe_desc = ret if ret >= 0 else None

    def is_active(self):
        """Query whether the pipe is active / open.

        :return: True if the pipe is active.
        :rtype: bool
        """
        return self.pipe_desc is not None

    def close(self):
        """Close the pipe, release memory associated with the pipe"""
        if self.pipe_desc is not None:
            self.lib.sc_pipe_close2(self.dev_desc, self.pipe_desc)
            self.pipe_desc = None

    def read(self, timeout_ms = 500):
        """Wait until blob data for the next camera frame becomes available or
        timeout is reached and read it.
        Returns blob data of the next camera frame. If data is returned, access
        is only allowed until the next time that this read function is called.
        Perform a copy of the data if you need to keep it for longer. As soon as
        a CamBlobsPipe is opened and measurements are started, the pipe
        allocates memory for storing data which is released by reading. Not
        reading the pipe frequently enough can exhaust the memory.

        :param timeout_ms: the timeout in milliseconds, defaults to 500
        :type timeout_ms: int, optional
        :return: Returns an array of blob positions in case of success. If an
          error occurs, returns a tuple (error_code, error_message).
        :rtype: numpy.ndarray | (int, str)
        """
        (ret, bufptr) = self.lib.sc_pipe_read2(
            self.dev_desc, self.pipe_desc, timeout_ms)
        if ret >= 0 and bufptr:
            dataptr = ctypes.cast(bufptr, ctypes.POINTER(sc_cam_blob_meta_t))
            d = dataptr.contents
            d.nr_blobs
            if d.nr_blobs > 0:
                blobptr = ctypes.cast(
                    bufptr.value + d.data_offset,
                    ctypes.POINTER(sc_cam_blob_position_t))
                blobs = np.ctypeslib.as_array(blobptr,
                    shape=(d.nr_blobs,))
                return blobs
            else:
                return np.zeros((0,), dtype=sc_cam_blob_position_t)
        else:
            return (ret, self.lib.sc_get_err_msg(ret))

class Camera(Device):
    """A specialization of the :any:`Device` class that offers additional,
    camera-specific functions"""
    def add_frame_pipe(self):
        """Add a CamFramePipe pipe for receiving meta data and image data for
        individual camera frames.

        :return: a tuple containing a pipe ID and the :any:`CamFramePipe` object
            if successful; a tuple containing the error code and an error
            message in case of failure.
        :rtype: (int, CamFramePipe) | (int, str)
        """
        if self.dev_desc is None:
            return (-10, "Device not initialized")
        for i in range(1000):
            if i not in self.pipes:
                self.pipes[i] = CamFramePipe(self)
                return (i,self.pipes[i])
        return (-10000, "Too many pipes open")

    def add_blobs_pipe(self):
        """Add a CamBlobsPipe for receiving blob data for individual camera
        frames.

        :return: a tuple containing a pipe ID and the :any:`CamBlobsPipe` object
            if successful; a tuple containing the error code and an error
            message in case of failure.
        :rtype: (int, CamBlobsPipe) | (int, str)
        """
        if self.dev_desc is None:
            return (-10, "Device not initialized")
        for i in range(1000):
            if i not in self.pipes:
                self.pipes[i] = CamBlobsPipe(self)
                return (i,self.pipes[i])
        return (-10000, "Too many pipes open")

    def set_exposure_and_frames(self, exposure, nrframes):
        """Set the exposure per frame in microseconds and the number of frames

        :param exposure: the exposure per frame in microseconds
        :type exposure: int
        :param nrframes: the number of frames
        :type nrframes: int
        :return: (0, "") on success or a tuple with negative error code and
          error message
        :rtype: (int, str)
        """
        if self.dev_desc is None:
            return (-10, "Device not initialized")
        exposure = max(exposure, 2)
        nrframes = max(nrframes, 1)
        ret = self.lib.lib.sc_tdc_cam_set_exposure(
            self.dev_desc, exposure, nrframes)
        if ret < 0:
            return (ret, self.lib.sc_get_err_msg(ret))
        else:
            return (0, "")

    def get_max_size(self):
        """Get the maximum possible width and height for regions of interest
        (the width and height in pixels of the sensor area).

        :return: a tuple (0, roi) where roi is a dictionary with keys 'width'
          and 'height' if successful, a tuple (error_code, error_message) in
          case of failure
        :rtype: (int, dict) | (int, str)
        """
        if self.dev_desc is None:
            return (-10, "Device not initialized")
        width, height = ctypes.c_uint(), ctypes.c_uint()
        ret = self.lib.lib.sc_tdc_cam_get_maxsize(self.dev_desc,
            ctypes.byref(width), ctypes.byref(height))
        if ret < 0:
            return (ret, self.lib.sc_get_err_msg(ret))
        else:
            return (0, {'width' : width.value, 'height' : height.value})

    def set_region_of_interest(self, xmin, xmax, ymin, ymax):
        """Set the region of interest.

        :param xmin: the position of the boundary to the left
        :type xmin: int
        :param xmax: the position of the boundary to the right
        :type xmax: int
        :param ymin: the position of the top boundary
        :type ymin: int
        :param ymax: the position of the bottom boundary
        :type ymax: int
        :return: a tuple (0, "") in case of success, or a tuple containing a
          negative error code and an error message
        :rtype: (int, str)
        """
        if self.dev_desc is None:
            return (-10, "Device not initialized")
        ret = self.lib.lib.sc_tdc_cam_set_roi(
            self.dev_desc, xmin, xmax, ymin, ymax)
        if ret < 0:
            return (ret, self.lib.sc_get_err_msg(ret))
        else:
            return (0, "")

    def get_region_of_interest(self):
        """Get the currently set region of interest

        :return: a tuple (0, roi) where roi is a dict with keywords 'xmin',
          'xmax', 'ymin', 'ymax' if successful, a tuple containing error code
          and error message in case of failure
        :rtype: (int, dict) | (int, str)
        """
        if self.dev_desc is None:
            return (-10, "Device not initialized")
        xmin, xmax, ymin, ymax = (ctypes.c_uint(), ctypes.c_uint(),
            ctypes.c_uint(), ctypes.c_uint())
        ret = self.lib.lib.sc_tdc_cam_get_roi(self.dev_desc,
            ctypes.byref(xmin), ctypes.byref(xmax), ctypes.byref(ymin),
            ctypes.byref(ymax))
        if ret < 0:
            return (ret, self.lib.sc_get_err_msg(ret))
        else:
            return (0, {'xmin':xmin.value, 'xmax':xmax.value, 'ymin':ymin.value,
                'ymax':ymax.value})

    def set_fanspeed(self, fanspeed):
        """Set the fan speed

        :param fanspeed: the fan speed on a scale from 0 (off) to 255 (maximum)
        :type fanspeed: int
        :return: (0, "") if successful; (error_code, error_message) in case of
          failure
        :rtype: (int, str)
        """
        if self.dev_desc is None:
            return (-10, "Device not initialized")
        ret = self.lib.lib.sc_tdc_cam_set_fanspeed(self.dev_desc, fanspeed)
        if ret < 0:
            return (ret, self.lib.sc_get_err_msg(ret))
        else:
            return (0, "")

    def set_blob_mode(self, blob_dif_min_top=1, blob_dif_min_bottom=3):
        """Activate blob mode. Refer to
        https://www.surface-concept.com/sctdc-sdk-doc/05_reconflex_cameras.html#blob-recognition-criteria,
        (condition 3 + 4) for explanation of the blob_dif_min_top/bottom
        parameters.

        :param blob_dif_min_top: allowed values range from 0 to 63
        :type blob_dif_min_top: int
        :param blob_dif_min_bottom: allowed values range from 0 to 63
        :type blob_dif_min_bottom: int
        :return: (0, "") if successful; (error_code, error_message) in case of
          failure
        :rtype: (int, str)
        """
        ret1 = self._set_param_wrapper("BlobDifMinTop", blob_dif_min_top)
        if ret1[0] != 0:
            return ret1
        return self._set_param_wrapper("BlobDifMinBottom", blob_dif_min_bottom)

    def set_image_mode(self):
        """Deactivate blob mode."""
        ret1 = self._set_param_wrapper("BlobDifMinTop", 0)
        if ret1[0] != 0:
            return ret1
        return self._set_param_wrapper("BlobDifMinBottom", 0)

    def set_smoother_masks_square(self, size1=1, size2=1):
        """Set the smoother pixel masks to filled squares of specified sizes.
        Specifying both sizes as 1 results in no smoothing.
        Allowed values for each of the size parameters are 1, 2, 3, 4, 5.
        """
        def size_to_mask_string(s):
            if s < 2:
                return "1"
            elif s == 2:
                return "{}".format(0x0303)
            elif s == 3:
                return "{}".format(0x070707)
            elif s == 4:
                return "{}".format(0x0F0F0F0F)
            else:
                return "{}".format(0x1F1F1F1F1F)
        ret1 = self._set_param_wrapper("SmootherPixelMask", size_to_mask_string(size1))
        if ret1[0] != 0:
            return ret1
        return self._set_param_wrapper("SmootherPixelMask2", size_to_mask_string(size2))

    def set_smoother_bit_shifts(self, shift1=0, shift2=0):
        """Set the smoother bit shifts applied to the intensity value after
        convolution with the smoother pixel mask after smoothing stages 1 and 2,
        respectively. Recommended shifts for square sizes used in
        `set_smoother_masks_square` are shift 0 for size 1, shift 2 for size 2,
        shift 3 for size 3, shift 4 for size 4, shift 4 or 5 for size 5.
        """
        ret1 = self._set_param_wrapper("SmootherShift", shift1)
        if ret1[0] != 0:
            return ret1
        return self._set_param_wrapper("SmootherShift2", shift2)

    def _set_param_wrapper(self, name, value):
        if self.dev_desc is None:
            return (-10, "Device not initialized")

        ret = self.lib.lib.sc_tdc_cam_set_parameter(self.dev_desc,
            name.encode('utf-8'), str(value).encode('utf-8'))
        if ret < 0:
            return (ret, self.lib.sc_get_err_msg(ret))
        else:
            return (0, "")

    def _get_param_wrapper(self, name):
        if self.dev_desc is None:
            return (-10, "Device not initialized")
        vsize = ctypes.c_size_t(63)
        value = ctypes.create_string_buffer(vsize.value)
        ret = self.lib.lib.sc_tdc_cam_get_parameter(self.dev_desc,
            name.encode('utf-8'), value, ctypes.byref(vsize))
        if ret < 0:
            return (ret, self.lib.sc_get_err_msg(ret))
        else:
            try:
                vstr = value.value.decode('utf-8')
                return (0, int(vstr))
            except ValueError:
                return (-10000, "cannot convert result: {}".format(vstr))

    def set_analog_gain(self, value):
        """Set the analog gain (a property of the sensor)

        :param value: the analog gain value ranging from 0 to 480
        :type value: int
        :return: (0, "") in case of success; (error_code, error_message) in case
          of failure
        :rtype: (int, str)
        """
        return self._set_param_wrapper("AnalogGain", value)

    def get_analog_gain(self):
        """Get the currently set analog gain (a property of the sensor)

        :return: A tuple (0, analog_gain) if successful; a tuple
          (error_code, error_message) in case of failure
        :rtype: (int, int) | (int, str)
        """
        return self._get_param_wrapper("AnalogGain")

    def set_black_offset(self, value):
        """Set the black offset (a property of the sensor)

        :param value: the black offset value ranging from 0 to 255 (in
          BitMode 8) or from 0 to 4095 (in BitMode 12).
        :type value: int
        :return: (0, "") in case of success; (error_code, error_message) in case
          of failure
        :rtype: (int, str)
        """
        return self._set_param_wrapper("BlackOffset", value)

    def get_black_offset(self):
        """Get the currently set black offset (a property of the sensor)

        :return: A tuple (0, black_offset) if successful; a tuple
          (error_code, error_message) in case of failure
        :rtype: (int, int) | (int, str)
        """
        return self._get_param_wrapper("BlackOffset")

    def set_white_pixel_min(self, value):
        """Set the 'White Pixel Min' parameter, a threshold criterion for
        filtering white pixels.

        :param value: the white pixel minimum value ranging from 0 to 255.
          0 turns white pixel remover off. 1 is the lowest threshold (removes
          the most white pixels).
        :type value: int
        :return: (0, "") in case of success; (error_code, error_message) in case
          of failure
        :rtype: (int, str)
        """
        return self._set_param_wrapper("WhitePixelMin", value)

    def get_white_pixel_min(self):
        """Get the current value of the 'White Pixel Min' parameter

        :return: A tuple (0, white_pixel_min) if successful; a tuple
          (error_code, error_message) in case of failure
        :rtype: (int, int) | (int, str)
        """
        return self._get_param_wrapper("WhitePixelMin")

    def set_white_pixel_relax(self, value):
        """Set the 'White Pixel Relax' parameter, which controls a ratio
        between the center pixel and its horizontal and vertical neighbours such
        that if the ratio is exceeded, the center pixel is considered as a white
        pixel

        :param value: the white pixel relax value, one of 0, 1, 2, 3.

          * white pixel relax == 0 : ratio 2;
          * white pixel relax == 1 : ratio 1.5;
          * white pixel relax == 2 : ratio 1.25;
          * white pixel relax == 3 : ratio 1;

        :type value: int
        :return: (0, "") in case of success; (error_code, error_message) in case
          of failure
        :rtype: (int, str)
        """
        return self._set_param_wrapper("WhitePixelRelax", value)

    def get_white_pixel_relax(self):
        """Get the current value of the 'White Pixel Relax' parameter

        :return: A tuple (0, white_pixel_relax) if successful; a tuple
          (error_code, error_message) in case of failure
        :rtype: (int, int) | (int, str)
        """
        return self._get_param_wrapper("WhitePixelRelax")

    def set_shutter_mode(self, value):
        """Set the shutter mode

        :param value: one of the values defined in the :py:class:`ShutterMode`
          class
        :type value: int
        :return: (0, "") in case of success; (error_code, error_message) in case
          of failure
        :rtype: (int, str)
        """
        return self._set_param_wrapper("ShutterMode", value)

    def get_shutter_mode(self):
        """Get the currently active shutter mode

        :return: A tuple (0, shutter_mode) if successful; a tuple
          (error_code, error_message) in case of failure. The shutter mode value
          is one of the constants defined in the :py:class:`ShutterMode` class
        :rtype: (int, int) | (int, str)
        """
        return self._get_param_wrapper("ShutterMode")