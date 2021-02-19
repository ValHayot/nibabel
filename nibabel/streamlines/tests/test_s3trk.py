#!/usr/bin/env python
from nibabel.streamlines import s3trk
import os

def prep(cache_dir="/dev/shm"):

    # create main file
    fname = "random.bin"
    size = 1024**2
    with open(fname, 'wb') as fout:
        fout.write(os.urandom(size)) 

    # take a chunk and save it to tmpfs space
    f = open(fname, "rb")

    cfname = os.path.join(cache_dir, f"{fname}.0")
    csize = int(size / 2)

    with open(cfname, 'wb') as cf_:
        cf_.write(f.read(csize))

    f.seek(0, os.SEEK_SET)

    cf_ = open(cfname, "rb")

    caches = { cache_dir: size }

    return f, cf_, csize + 256, (0, csize), fname, caches 

    
def test_cached_read():

    f, cf_, nbytes, fidx, fn_prefix, caches = prep()
    is_cached, data, cf_, fidx = s3trk.cached_read(f, cf_, nbytes, fidx, fn_prefix, caches)

    f.seek(0, os.SEEK_SET)
    assert(is_cached is False)
    assert(data == f.read(nbytes))
    assert(cf_ is None)
    assert(fidx is None)


