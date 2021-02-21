#!/usr/bin/env python
import os
import pytest

import s3fs
import boto3
from moto import mock_s3

from pathlib import Path
from nibabel.streamlines import rolling_prefetch as rp


CACHE_DIR = "/dev/shm"
CACHE_SIZE = 1024 ** 2
CACHES = {CACHE_DIR: CACHE_SIZE}
DELETE_STR = ".nibtodelete"
BUCKET_NAME = "s3trk"

port = 5555
endpoint_uri = "http://127.0.0.1:%s/" % port

# taken from s3fs tests https://github.com/dask/s3fs/blob/main/s3fs/tests/test_s3fs.py#L57
@pytest.fixture()
def s3_base():
    # writable local S3 system
    import shlex
    import subprocess
    import requests
    import time

    proc = subprocess.Popen(shlex.split("moto_server s3 -p %s" % port))

    timeout = 5
    while timeout > 0:
        try:
            r = requests.get(endpoint_uri)
            if r.ok:
                break
        except:
            pass
        timeout -= 0.1
        time.sleep(0.1)
    yield
    proc.terminate()
    proc.wait()


@pytest.fixture()
def s3(s3_base):
    from botocore.session import Session

    # NB: we use the sync botocore client for setup
    session = Session()
    client = session.create_client("s3", endpoint_url=endpoint_uri)
    client.create_bucket(Bucket=BUCKET_NAME)

    yield


@pytest.fixture
def create_main_file(s3):
    # create main file
    fname = "random.bin"

    # generate a random bytestring
    data = os.urandom(CACHE_SIZE)

    s3_path = os.path.join(BUCKET_NAME, fname)
    fs = s3fs.S3FileSystem()

    with fs.open(s3_path, "wb") as f:
        f.write(data)

    return s3_path


@pytest.fixture
def create_cached(create_main_file):

    fname = os.path.basename(create_main_file)

    # take a chunk and save it to tmpfs space
    f = open(fname, "rb")

    csize = int(CACHE_SIZE / 4)
    cfname_0 = os.path.join(CACHE_DIR, f"{fname}.0")
    cfname_1 = os.path.join(CACHE_DIR, f"{fname}.{csize}")

    with open(cfname_0, "wb") as cf_:
        cf_.write(f.read(csize))

    with open(cfname_1, "wb") as cf_:
        cf_.write(f.read(csize))

    f.seek(0, os.SEEK_SET)

    cf_ = open(cfname_0, "rb")

    return {
        "f": f,
        "nbytes": csize * 2 + 256,
        "fn_prefix": fname,
        "cf_": cf_,
        "fidx": (0, csize),
    }


def cleanup(fn_prefix):
    for c in Path(CACHE_DIR).glob(fn_prefix + "*"):
        c.unlink()


def test_prefetch(create_main_file):
    fname = create_main_file

    bs = CACHE_SIZE / 2
    rp.prefetch(fname, CACHES, block_size=bs)

    f_bn = os.path.basename(fname)

    # test prefetching
    cached_files = Path(CACHE_DIR).glob("*")
    cf = list(cached_files)
    assert len(cf) == 2

    # file removal
    to_remove = os.path.join(CACHE_DIR, f"{f_bn}.1{DELETE_STR}")
    os.rename(cf[0], to_remove)

    rp.prefetch(fname, CACHES, block_size=CACHE_SIZE)
    assert not os.path.exists(to_remove)
    cleanup(f_bn)


def test_cached_read(create_cached):

    data, cf_, fidx = rp.cached_read(caches=CACHES, **create_cached)
    create_cached["f"].seek(0, os.SEEK_SET)

    assert data == create_cached["f"].read(create_cached["nbytes"])
    assert cf_ is None
    assert fidx is None
    assert os.path.exists(f"{create_cached['cf_'].name}{DELETE_STR}")

    cleanup(create_cached["fn_prefix"])


def test_pf_read_cached(create_cached):

    data, cf_, fidx = rp.pf_read(caches=CACHES, **create_cached)
    create_cached["f"].seek(0, os.SEEK_SET)

    assert data == create_cached["f"].read(create_cached["nbytes"])
    assert cf_ is None
    assert fidx is None
    assert os.path.exists(f"{create_cached['cf_'].name}{DELETE_STR}")

    cleanup(create_cached["fn_prefix"])


def test_pf_read_uncached(create_main_file):

    fn_prefix = os.path.basename(create_main_file)
    fs = s3fs.S3FileSystem()
    nbytes = 1024

    with fs.open(create_main_file, "rb") as f:
        data, cf_, fidx = rp.pf_read(
            caches=CACHES, f=f, nbytes=1024, fn_prefix=fn_prefix
        )
        f.seek(0, os.SEEK_SET)
        assert data == f.read(nbytes)

    assert cf_ is None
    assert fidx is None

    cleanup(fn_prefix)
