#!/usr/bin/env python
import asyncio
import s3fs
import glob
import os
import concurrent.futures

from shutil import disk_usage
from pathlib import Path

# NOTE: Need to pass s3fs conditions over so that they can be used here

# extension of files flagged for deletion
DELETE_STR = ".nibtodelete"

# @profile
def prefetch(
    filename, caches, block_size=64*1024**2, **s3_kwargs
):
    """Concurrently fetch data from S3 in blocks and store in cache

    Parameters
    ----------
    filename : str
        S3 file to prefetch from
    caches : dict
        Dictionary containing cache paths as keys and available space (in MB) as values
    block_size: int
        Number of bytes to prefect at a time (default: 32MB)
    s3_kwargs: kwargs
        Keyword arguments to pass to s3fs object

    """
    fn_prefix = os.path.basename(filename)
    fs = s3fs.S3FileSystem(**s3_kwargs)

    # try / except as filesystem may be closed by read thread
    try:
        total_bytes = fs.du(filename)
        offset = 0

        fetch = True

        # Loop until all data has been read
        while fetch:
            # remove files flagged for deletion
            for c in caches.keys():
                for p in Path(c).glob(f"*{DELETE_STR}"):
                    p.unlink()

            # NOTE: will use a bit of memory to read/write file. Need to warn user
            # Prefetch to cache
            for path, space in caches.items():

                space *= 1024 ** 2  # convert to bytes from megabytes
                avail_cache = min(disk_usage(path).free, space)

                while avail_cache >= block_size and total_bytes > offset:

                    data = fs.read_block(filename, offset, block_size)

                    # only write to final path when data copy is complete
                    tmp_path = os.path.join(path, f".{fn_prefix}.{offset}.tmp")
                    final_path = os.path.join(path, f"{fn_prefix}.{offset}")

                    with open(tmp_path, "wb") as f:
                        f.write(data)

                    os.rename(tmp_path, final_path)
                    offset += block_size
                    avail_cache = disk_usage(path).free - space

                # if we have already read the entire file terminate prefetching
                if total_bytes <= offset:
                    fetch = False
                    break
    except:
        pass


# @profile
def get_block(fn_prefix, caches, offset):
    """Open the cached block fileobj at the necessary file offset

    Parameters
    ----------
    fn_prefix : str
        The basename of the original file
    caches : dict
        Dictionary containing cache paths as keys and available space (in MB) as values
    offset : int
        byte offset in main file

    Returns
    -------
    cf_ : fileobj
        Fileobj of the cached file (returns None if it does not exist)
    k : tuple (int, int)
        The positioning of the opened block respective to the original file
    """
    # Get the list of files in cache
    cached_files = [
        fn
        for fs in caches.keys()
        for fn in glob.glob(os.path.join(fs, f"{fn_prefix}.[0-9]*"))
        if DELETE_STR not in fn
    ]

    # Iterate through the cached files/offsets
    for f in cached_files:

        # Get position of cached block relative to original file
        b_start = int(f.split(".")[-1])
        b_end = b_start + os.stat(f).st_size

        if offset >= b_start and offset < b_end:
            cf_ = open(f, "rb")
            c_offset = offset - b_start
            cf_.seek(c_offset, os.SEEK_SET)
            return cf_, (b_start, b_end)

    return None, None


# @profile
def cached_read(f, nbytes, fn_prefix, caches, cf_, fidx):
    """Read necessary bytes from cached blocks. If remainder of data is not in cache, read from original
    location.

    Parameters
    ----------
    f : fileobj
        The original file fileobj
    nbytes: int
        The number of bytes needed to be read
    cf_: fileobj
        The fileobj corresponding to the cached file
    fn_prefix: str
        The original file basename
    fidx: tuple (int, int)
        The start and end index of the cached file
    caches : dict
        Dictionary containing cache paths as keys and available space (in MB) as values

    Returns
    -------
    data: bytestring
        The total read data of size nbytes
    cf_ : fileobj
        Fileobj of the cached file (returns None if it does not exist)
    fidx: tuple (int, int)
        The start and end index of the cached file
    """

    # get the number of bytes remaining in file relative to current offset
    b_remaining = fidx[1] - fidx[0] - cf_.tell()

    # get the number of bytes that will remain after reading from this cached block
    remainder_bytes = max(nbytes - b_remaining, 0)

    # read necessary bytes provided they're available
    data = cf_.read(max(min([nbytes, b_remaining]), 0))

    # update global offset to point to next byte
    offset = cf_.tell() + fidx[0]

    # Loop through cached files or read from original file remaining bytes
    while remainder_bytes > 0:

        # flag previous cached file for deletion and close it
        os.rename(cf_.name, f"{cf_.name}{DELETE_STR}")
        cf_.close()

        # check to see if offset is in cache
        cf_, fidx = get_block(fn_prefix, caches, offset)

        # read remaining from cached blocks, if possible
        if cf_ is not None:
            b_remaining = fidx[1] - fidx[0] - cf_.tell()
            nbytes = remainder_bytes
            remainder_bytes = max(nbytes - b_remaining, 0)

            # appending additional data to previous data collected
            data += cf_.read(min(nbytes, b_remaining))

            # update global offset
            offset = cf_.tell() + fidx[0]
        else:
            f.seek(offset, os.SEEK_SET)
            data += f.read(remainder_bytes)
            remainder_bytes = 0
            is_cached = False
            cf_ = None
            fidx = None

    else:
        # Make sure to update global file offset
        if cf_ is not None:
            f.seek(offset, os.SEEK_SET)

    return data, cf_, fidx

def pf_read(f, nbytes, fn_prefix, caches={}, cf_=None, fidx=None):
    """Determine whether to read bytes from cache or original file

    Parameters
    ----------
    f : fileobj
        The original file fileobj
    nbytes: int
        The number of bytes needed to be read
    fn_prefix: str
        The original file basename
    caches : dict
        Dictionary containing cache paths as keys and available space (in MB) as values
    cf_: fileobj
        The fileobj corresponding to the cached file
    fidx: tuple (int, int)
        The start and end index of the cached file

    Returns
    -------
    data: bytestring
        The total read data of size nbytes
    cf_ : fileobj
        Fileobj of the cached file (returns None if it does not exist)
    fidx: tuple (int, int)
        The start and end index of the cached file
    """

    if cf_ is not None:
        data, cf_, fidx = cached_read(f, nbytes, fn_prefix, caches, cf_, fidx)
    else:
        data = f.read(nbytes)
        get_block(fn_prefix, caches, f.tell())

    return data, cf_, fidx

