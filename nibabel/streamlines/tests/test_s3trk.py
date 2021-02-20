#!/usr/bin/env python
import os
import pytest

import s3fs
import boto3
from moto import mock_s3

from pathlib import Path
from nibabel.streamlines import rolling_prefetch as rp


CACHE_DIR = "/dev/shm"
CACHE_SIZE = 1024**2
CACHES = { CACHE_DIR : CACHE_SIZE }
DELETE_STR = ".nibtodelete"
BUCKET_NAME = "s3trk"

@pytest.fixture
@mock_s3
def create_main_file():
    # create main file
    fname = "random.bin"

    # generate a random bytestring
    data = os.urandom(CACHE_SIZE)

    conn = boto3.resource('s3')
    conn.create_bucket(Bucket=BUCKET_NAME)
    conn.Bucket(BUCKET_NAME).put_object(Key=fname, Body=data)

    s3_path = os.path.join(BUCKET_NAME, fname)

    return s3_path

@pytest.fixture
@mock_s3
def create_cached(create_main_file):

    fname = os.path.basename(create_main_file)

    # take a chunk and save it to tmpfs space
    f = open(fname, "rb")

    conn = boto3.resource('s3')
    conn.create_bucket(Bucket=BUCKET_NAME)

    cfname = os.path.join(CACHE_DIR, f"{fname}.0")
    csize = int(CACHE_SIZE / 2)

    with open(cfname, 'wb') as cf_:
        cf_.write(f.read(csize))

    f.seek(0, os.SEEK_SET)

    cf_ = open(cfname, "rb")

    return { "f": f, "nbytes": csize + 256, "fn_prefix": fname, "cf_": cf_, "fidx": (0, csize) }

    cleanup([cfname])


def cleanup(cfnames=[]):
    for c in cfnames:
        os.unlink(c)

    
def test_cached_read(create_cached):

    data, cf_, fidx = rp.cached_read(caches=CACHES, **create_cached)
    create_cached["f"].seek(0, os.SEEK_SET)

    assert(data == create_cached["f"].read(create_cached["nbytes"]))
    assert(cf_ is None)
    assert(fidx is None)
    assert(os.path.exists(f"{create_cached['cf_'].name}{DELETE_STR}"))

def test_prefetch(create_main_file):
    fname = create_main_file

    bs = CACHE_SIZE / 2
    rp.prefetch(fname, CACHES, block_size=bs)

    f_bn = os.path.basename(fname)

    # test prefetching
    cached_files = Path(CACHE_DIR).glob(os.path.basename(f_bn) + "*")
    assert(len(list(cached_files)) == 2), cached_files

    # file removal
    to_remove = os.path.join(CACHE_DIR, f"{f_bn}.1{DELETE_STR}")
    os.rename(cached_files[0], to_remove)

    rp.prefetch(fname, CACHES, block_size=CACHE_SIZE)
    assert(not os.path.exists(to_remove))
    
