
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
from shutil import disk_usage
from pathlib import Path

import numpy as np
from numpy.compat.py3k import asstr

import nibabel as nib

from nibabel.openers import Opener
from nibabel.volumeutils import (native_code, swapped_code, endian_codes)
from nibabel.orientations import (aff2axcodes, axcodes2ornt)

from .array_sequence import create_arraysequences_from_generator
from .tractogram_file import TractogramFile
from .tractogram_file import DataError, HeaderError, HeaderWarning
from .tractogram import TractogramItem, Tractogram, LazyTractogram
from .header import Field
from .utils import peek_next
from .trk import *


# NOTE: Need to pass s3fs conditions over so that they can be used here
def rolling_prefetch(filename, caches, block_size=43036684):
    fn_prefix = os.path.basename(filename)
    fs = s3fs.S3FileSystem()
    offset = 0

    # Loop until all data has been read
    while True:
        # remove files flagged for deletion
        for c in caches.keys():
            for p in Path(c).glob("*.nibtodelete"):
                p.unlink()

        # NOTE: will use a bit of memory to read/write file. Need to warn user
        # Prefetch to cache
        for path,space in caches.items():
            
            space *= 1024**2 # convert to bytes from megabytes
            avail_cache = disk_usage(path).free - space

            while avail_cache >= block_size:
                try:

                    data = fs.read_block(filename, offset, block_size)

                    # if offsethas exceed available data
                    if len(data) == 0 :
                        return

                    # only write to final path when data copy is complete
                    tmp_path = os.path.join(path, f".{fn_prefix}.{offset}.tmp")
                    final_path = os.path.join(path, f"{fn_prefix}.{offset}")
                    with open(tmp_path, "wb") as f:
                        f.write(data)
                    os.rename(tmp_path, final_path)
                    offset += block_size
                    avail_cache = disk_usage(path).free - space

                except Exception as e:
                    # Assuming file has finished being read here
                    return

        if fs.du(filename) <= offset:
            return



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

    def __init__(self, tractogram, header=None, caches={ "/dev/shm": 7*1024 }):
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

        self.caches = caches
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
    def load(cls, fileobj, lazy_load=False, caches={ "/dev/shm": 7*1024 }):
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
        #TODO: remove
        cls.caches = caches
        hdr = cls._read_header(fileobj, cls.caches)

        # create rolling prefetch thread
        t = threading.Thread(target=rolling_prefetch, args=(fileobj.path, deepcopy(caches)))
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
            for pts, scals, props in cls._read(fileobj, hdr, cls.caches):
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
    def _read_header(fileobj, caches):
        """ Reads a TRK header from a file.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TRK file (and ready to read from the beginning
            of the TRK header). Note that calling this function
            does not change the file position.

        Returns
        -------
        header : dict
            Metadata associated with this tractogram file.
        """
        # Record start position if this is a file-like object
        start_position = fileobj.tell() if hasattr(fileobj, 'tell') else None

        with Opener(fileobj) as f:
            # Reading directly from a file into a (mutable) bytearray enables a zero-copy
            # cast to a mutable numpy object with frombuffer
            header_buf = bytearray(header_2_dtype.itemsize)
            f.readinto(header_buf)
            header_rec = np.frombuffer(buffer=header_buf, dtype=header_2_dtype)
            # Check endianness
            endianness = native_code
            if header_rec['hdr_size'] != TrkFile.HEADER_SIZE:
                endianness = swapped_code

                # Swap byte order
                header_rec = header_rec.newbyteorder()
                if header_rec['hdr_size'] != TrkFile.HEADER_SIZE:
                    msg = (f"Invalid hdr_size: {header_rec['hdr_size']} "
                           f"instead of {TrkFile.HEADER_SIZE}")
                    raise HeaderError(msg)

            if header_rec['version'] == 1:
                # There is no 4x4 matrix for voxel to RAS transformation.
                header_rec[Field.VOXEL_TO_RASMM] = np.zeros((4, 4))
            elif header_rec['version'] == 2:
                pass  # Nothing more to do.
            else:
                raise HeaderError('NiBabel only supports versions 1 and 2 of '
                                  'the Trackvis file format')

            # Convert the first record of `header_rec` into a dictionnary
            header = dict(zip(header_rec.dtype.names, header_rec[0]))
            header[Field.ENDIANNESS] = endianness

            # If vox_to_ras[3][3] is 0, it means the matrix is not recorded.
            if header[Field.VOXEL_TO_RASMM][3][3] == 0:
                header[Field.VOXEL_TO_RASMM] = np.eye(4, dtype=np.float32)
                warnings.warn(("Field 'vox_to_ras' in the TRK's header was"
                               " not recorded. Will continue assuming it's"
                               " the identity."), HeaderWarning)

            # Check that the 'vox_to_ras' affine is valid, i.e. should be
            # able to determine the axis directions.
            axcodes = aff2axcodes(header[Field.VOXEL_TO_RASMM])
            if None in axcodes:
                msg = ("The 'vox_to_ras' affine is invalid! Could not"
                       " determine the axis directions from it.\n"
                       f"{header[Field.VOXEL_TO_RASMM]}")
                raise HeaderError(msg)

            # By default, the voxel order is LPS.
            # http://trackvis.org/blog/forum/diffusion-toolkit-usage/interpretation-of-track-point-coordinates
            if header[Field.VOXEL_ORDER] == b"":
                msg = ("Voxel order is not specified, will assume 'LPS' since"
                       " it is Trackvis software's default.")
                warnings.warn(msg, HeaderWarning)
                header[Field.VOXEL_ORDER] = b"LPS"

            # Keep the file position where the data begin.
            header['_offset_data'] = f.tell()

        # Set the file position where it was, if it was previously open.
        if start_position is not None:
            fileobj.seek(start_position, os.SEEK_SET)

        return header

    @staticmethod
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
            fns = get_cached(fn_prefix, caches)
            
            is_cached, cf_, offset_, fidx = get_updated_offset(fns, header["_offset_data"])

            # Set the file position at the beginning of the data.
            if is_cached:
                cf_.seek(offset_, os.SEEK_SET)
            else:
                f.seek(header["_offset_data"], os.SEEK_SET)

            # If 'count' field is 0, i.e. not provided, we have to loop
            # until the EOF.
            nb_streamlines = header[Field.NB_STREAMLINES]
            if nb_streamlines == 0:
                nb_streamlines = np.inf

            count = 0
            nb_pts_dtype = i4_dtype.str[:-1]
            while count < nb_streamlines:
                if is_cached:
                    is_cached, nb_pts_str, cf_, fidx  = cached_read(f, cf_, i4_dtype.itemsize, fidx, fn_prefix, caches)
                else:
                    nb_pts_str = f.read(i4_dtype.itemsize)

                # Check if we reached EOF
                if len(nb_pts_str) == 0:
                    break

                # Read number of points of the next streamline.
                nb_pts = struct.unpack(nb_pts_dtype, nb_pts_str)[0]

                br = nb_pts * pts_and_scalars_size
                if is_cached:
                    is_cached, data, cf_, fidx  = cached_read(f, cf_, br, fidx, fn_prefix, caches)
                else:
                    data = f.read(br)

                # Read streamline's data
                points_and_scalars = np.ndarray(
                    shape=(nb_pts, nb_pts_and_scalars),
                    dtype=f4_dtype,
                    buffer=data)

                points = points_and_scalars[:, :3]
                scalars = points_and_scalars[:, 3:]

                if is_cached:
                    is_cached, buffer, cf_, fidx  = cached_read(f, cf_, properties_size, fidx, fn_prefix, caches)
                else:
                    buffer = f.read(properties_size)

                # Read properties
                properties = np.ndarray(
                    shape=(nb_properties,),
                    dtype=f4_dtype,
                    buffer=buffer)

                yield points, scalars, properties
                count += 1

            # In case the 'count' field was not provided.
            header[Field.NB_STREAMLINES] = count

            # Set the file position where it was (in case it was already open).
            f.seek(start_position, os.SEEK_CUR)


def get_cached(fn_prefix, caches):
    return [fn for fs in caches.keys() for fn in glob.glob(os.path.join(fs, f"{fn_prefix}.*"))]

def get_cached_idx(fns):
    return {(int(fn.split('.')[-1]), int(fn.split('.')[-1]) + os.stat(fn).st_size) : fn for fn in fns}

def get_updated_offset(fns, offset):
    cidx = get_cached_idx(fns) 

    for k,v in cidx.items():
        if offset >= k[0] and offset < k[1]:
            return True, open(v, "rb"), offset - k[0], k

    return False, None, offset, None

def cached_read(f, cf_, nbytes, fidx, fn_prefix, caches):
    b_remaining = fidx[1] - fidx[0] - cf_.tell()
    remainder_bytes = max(nbytes - b_remaining, 0)
    data = cf_.read(min([nbytes, b_remaining]))
    offset = fidx[1]

    is_cached = True
    path = cf_.name
    while remainder_bytes > 0:
        cf_.close()
        fns = get_cached(fn_prefix, caches)
        is_cached, cf_, offset, fidx = get_updated_offset(fns, offset)

        path = cf_.name

        if is_cached:
            cf_.seek(offset, os.SEEK_SET)
            b_remaining = fidx[1] - fidx[0] - cf_.tell()
            nbytes = remainder_bytes
            remainder_bytes = max(nbytes - b_remaining, 0)
            data += cf_.read(min(nbytes, b_remaining))
            offset = cf_.tell() + fidx[0]
        else:
            f.seek(offset, os.SEEK_SET)
            # assumes all data can just be read from long term storage
            data += f.read(remainder_bytes)
            remainder_bytes = 0
            is_cached = False
            cf_ = None
            fidx = None

        if remainder_bytes > 0 and cf_ is not None:
            cf_.close()
            os.rename(path, f"{path}.nibtodelete")

    return is_cached, data, cf_, fidx

