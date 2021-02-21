
# Definition of trackvis header structure:
# http://www.trackvis.org/docs/?subsect=fileformat

import os
import glob
import s3fs
import struct
import string
import warnings
import threading
from copy import deepcopy

import numpy as np
from numpy.compat.py3k import asstr

import nibabel as nib

from nibabel.openers import Opener
from nibabel.volumeutils import (native_code, swapped_code, endian_codes)
from nibabel.orientations import (aff2axcodes, axcodes2ornt)

from .rolling_prefetch import prefetch, get_block, cached_read, pf_read

from .array_sequence import create_arraysequences_from_generator
from .tractogram_file import TractogramFile
from .tractogram_file import DataError, HeaderError, HeaderWarning
from .tractogram import TractogramItem, Tractogram, LazyTractogram
from .header import Field
from .utils import peek_next
from .trk import *




class S3TrkFile(TrkFile):
    """ Convenience class to encapsulate TRK file format.

    Notes
    -----
    TrackVis (so its file format: TRK) considers the streamline coordinate
    (0,0,0) to be in the corner of the voxel whereas NiBabel's streamlines
    internal representation (Voxel space) assumes (0,0,0) to be in the
    center of the voxel.

    Thus, streamlines are shifted by half a voxel on load and are shifted
    back on save.
    """

    # Constants
    MAGIC_NUMBER = b"TRACK"
    HEADER_SIZE = 1000
    SUPPORTS_DATA_PER_POINT = True
    SUPPORTS_DATA_PER_STREAMLINE = True

    def __init__(self, tractogram, header=None):
        """
        Parameters
        ----------
        tractogram : :class:`Tractogram` object
            Tractogram that will be contained in this :class:`TrkFile`.

        header : dict, optional
            Metadata associated to this tractogram file.

        Notes
        -----
        Streamlines of the tractogram are assumed to be in *RAS+*
        and *mm* space where coordinate (0,0,0) refers to the center
        of the voxel.
        """

        super(S3TrkFile, self).__init__(tractogram, header)

    @classmethod
    def is_correct_format(cls, fileobj):
        """ Check if the file is in TRK format.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TRK file (and ready to read from the beginning
            of the TRK header data). Note that calling this function
            does not change the file position.

        Returns
        -------
        is_correct_format : {True, False}
            Returns True if `fileobj` is compatible with TRK format,
            otherwise returns False.
        """
        with Opener(fileobj) as f:
            magic_len = len(cls.MAGIC_NUMBER)
            magic_number = f.read(magic_len)
            f.seek(-magic_len, os.SEEK_CUR)
            return magic_number == cls.MAGIC_NUMBER and "s3://" in fileobj.bucket


    @classmethod
    #@profile
    def load(cls, fileobj, lazy_load=False, caches={ "/dev/shm": 7*1024 }, prefetch_size=32*1024**2):
        """ Loads streamlines from a filename or file-like object.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TRK file (and ready to read from the beginning
            of the TRK header). Note that calling this function
            does not change the file position.
        lazy_load : {False, True}, optional
            If True, load streamlines in a lazy manner i.e. they will not be
            kept in memory. Otherwise, load all streamlines in memory.

        Returns
        -------
        trk_file : :class:`TrkFile` object
            Returns an object containing tractogram data and header
            information.

        Notes
        -----
        Streamlines of the returned tractogram are assumed to be in *RAS*
        and *mm* space where coordinate (0,0,0) refers to the center of the
        voxel.
        """
        hdr = cls._read_header(fileobj)

        # create rolling prefetch thread
        t = threading.Thread(target=prefetch, args=(fileobj.path, deepcopy(caches), prefetch_size))
        t.start()
        # Find scalars and properties name
        data_per_point_slice = {}
        if hdr[Field.NB_SCALARS_PER_POINT] > 0:
            cpt = 0
            for scalar_field in hdr['scalar_name']:
                scalar_name, nb_scalars = decode_value_from_name(scalar_field)

                if nb_scalars == 0:
                    continue

                slice_obj = slice(cpt, cpt + nb_scalars)
                data_per_point_slice[scalar_name] = slice_obj
                cpt += nb_scalars

            if cpt < hdr[Field.NB_SCALARS_PER_POINT]:
                slice_obj = slice(cpt, hdr[Field.NB_SCALARS_PER_POINT])
                data_per_point_slice['scalars'] = slice_obj

        data_per_streamline_slice = {}
        if hdr[Field.NB_PROPERTIES_PER_STREAMLINE] > 0:
            cpt = 0
            for property_field in hdr['property_name']:
                results = decode_value_from_name(property_field)
                property_name, nb_properties = results

                if nb_properties == 0:
                    continue

                slice_obj = slice(cpt, cpt + nb_properties)
                data_per_streamline_slice[property_name] = slice_obj
                cpt += nb_properties

            if cpt < hdr[Field.NB_PROPERTIES_PER_STREAMLINE]:
                slice_obj = slice(cpt, hdr[Field.NB_PROPERTIES_PER_STREAMLINE])
                data_per_streamline_slice['properties'] = slice_obj

        def _read():
            for pts, scals, props in cls._read(fileobj, hdr, caches):
                items = data_per_point_slice.items()
                data_for_points = dict((k, scals[:, v]) for k, v in items)
                items = data_per_streamline_slice.items()
                data_for_streamline = dict((k, props[v]) for k, v in items)
                yield TractogramItem(pts,
                                     data_for_streamline,
                                     data_for_points)

        tractogram = LazyTractogram.from_data_func(_read)

        tractogram.affine_to_rasmm = get_affine_trackvis_to_rasmm(hdr)
        tractogram = tractogram.to_world()

        return cls(tractogram, header=hdr)


    @staticmethod
    #@profile
    def _read(fileobj, header, caches=None):
        """ Return generator that reads TRK data from `fileobj` given `header`

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TRK file (and ready to read from the beginning
            of the TRK header). Note that calling this function
            does not change the file position.
        header : dict
            Metadata associated with this tractogram file.

        Yields
        ------
        data : tuple of ndarrays
            Length 3 tuple of streamline data of form (points, scalars,
            properties), where:

            * points: ndarray of shape (n_pts, 3)
            * scalars: ndarray of shape (n_pts, nb_scalars_per_point)
            * properties: ndarray of shape (nb_properties_per_point,)
        """
        i4_dtype = np.dtype(header[Field.ENDIANNESS] + "i4")
        f4_dtype = np.dtype(header[Field.ENDIANNESS] + "f4")

        # keep S3 open (not sure if necessary)
        with Opener(fileobj) as f:
            start_position = f.tell()
            

            nb_pts_and_scalars = int(3 +
                                     header[Field.NB_SCALARS_PER_POINT])
            pts_and_scalars_size = int(nb_pts_and_scalars * f4_dtype.itemsize)
            nb_properties = header[Field.NB_PROPERTIES_PER_STREAMLINE]
            properties_size = int(nb_properties * f4_dtype.itemsize)

            fn_prefix = os.path.basename(fileobj.path)
            cf_, fidx = get_block(fn_prefix, caches, header["_offset_data"])

            # Set the file position at the beginning of the data.
            if cf_ is None:
                f.seek(header["_offset_data"], os.SEEK_SET)

            # If 'count' field is 0, i.e. not provided, we have to loop
            # until the EOF.
            nb_streamlines = header[Field.NB_STREAMLINES]
            if nb_streamlines == 0:
                nb_streamlines = np.inf

            count = 0
            nb_pts_dtype = i4_dtype.str[:-1]
            while count < nb_streamlines:

                nb_pts_str, cf_, fidx = pf_read(f, i4_dtype.itemsize, fn_prefix, caches, cf_, fidx)

                # Check if we reached EOF
                if len(nb_pts_str) == 0:
                    break

                # Read number of points of the next streamline.
                nb_pts = struct.unpack(nb_pts_dtype, nb_pts_str)[0]

                br = nb_pts * pts_and_scalars_size

                data, cf_, fidx = pf_read(f, br, fn_prefix, caches, cf_, fidx)

                # Read streamline's data
                points_and_scalars = np.ndarray(
                    shape=(nb_pts, nb_pts_and_scalars),
                    dtype=f4_dtype,
                    buffer=data)

                points = points_and_scalars[:, :3]
                scalars = points_and_scalars[:, 3:]

                data, cf_, fidx = pf_read(f, properties_size, fn_prefix, caches, cf_, fidx)

                # Read properties
                properties = np.ndarray(
                    shape=(nb_properties,),
                    dtype=f4_dtype,
                    buffer=data)

                yield points, scalars, properties
                count += 1

            # In case the 'count' field was not provided.
            header[Field.NB_STREAMLINES] = count

            # Set the file position where it was (in case it was already open).
            f.seek(start_position, os.SEEK_CUR)


