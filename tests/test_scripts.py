import os
import tables
import numpy as np
import shapefile
import rasterio
from click import Context


from uncoverml import geoio
from uncoverml.scripts.maketargets import main as maketargets
from uncoverml.scripts.cvindexer import main as cvindexer
from uncoverml.scripts.extractfeats import main as extractfeats


def test_make_targets(make_shp_gtiff):

    fshp, _ = make_shp_gtiff
    field = "lon"

    ctx = Context(maketargets)
    ctx.forward(maketargets, shapefile=fshp, fieldname=field)

    fhdf5 = os.path.splitext(fshp)[0] + "_" + field + ".hdf5"

    assert os.path.exists(fhdf5)

    with tables.open_file(fhdf5, mode='r') as f:
        lon = f.root.targets.read()
        Longitude = f.root.Longitude.read().flatten()

    assert np.allclose(lon, Longitude)


def test_cvindexer_shp(make_shp_gtiff):

    fshp, _ = make_shp_gtiff
    folds = 6
    field = "lon"
    fshp_hdf5 = os.path.splitext(fshp)[0] + ".hdf5"
    fshp_targets = os.path.splitext(fshp)[0] + "_" + field + ".hdf5"

    # Make target file
    ctx = Context(maketargets)
    ctx.forward(maketargets, shapefile=fshp, fieldname=field)

    # Create Indices as if extractfeats had been called with an image
    permutation = np.arange(10)  # number of features in fixture
    geoio.writeback_target_indices(permutation, fshp_targets)

    # Make crossval with hdf5
    ctx = Context(cvindexer)
    ctx.forward(cvindexer, targetfile=fshp_targets, outfile=fshp_hdf5, folds=6)

    # Read in resultant HDF5
    with tables.open_file(fshp_hdf5, mode='r') as f:
        hdfcoords = np.hstack((f.root.Longitude.read(),
                               f.root.Latitude.read()))
        finds = f.root.FoldIndices.read()

    # Validate order is consistent with shapefile
    f = shapefile.Reader(fshp)
    shpcoords = np.array([p.points[0] for p in f.shapes()])

    assert np.allclose(shpcoords, hdfcoords)

    # Test we have the right number of folds
    assert finds.min() == 0
    assert finds.max() == (folds - 1)


def test_extractfeats(make_shp_gtiff, make_ipcluster4):

    fshp, ftif = make_shp_gtiff
    chunks = 4
    outdir = os.path.dirname(fshp)
    name = "fchunk_worker"

    # Extract features from gtiff
    ctx = Context(extractfeats)
    ctx.forward(extractfeats, geotiff=ftif, name=name, outputdir=outdir)

    ffiles = []
    for i in range(chunks):
        fname = os.path.join(outdir, "{}.part{}.hdf5".format(name, i))
        assert os.path.exists(fname)
        ffiles.append(fname)

    # Now compare extracted features to geotiff
    with rasterio.open(ftif, 'r') as f:
        I = np.transpose(f.read(), [2, 1, 0])

    efeats = []
    for fname in ffiles:
        print(fname)
        with tables.open_file(fname, 'r') as f:
            strip = [fts for fts in f.root.features]
            efeats.append(np.reshape(strip, (I.shape[0], -1, I.shape[2])))

    efeats = np.concatenate(efeats, axis=1)

    assert I.shape == efeats.shape
    assert np.allclose(I, efeats)


def test_extractfeats_targets(make_shp_gtiff, make_ipcluster4):

    fshp, ftif = make_shp_gtiff
    outdir = os.path.dirname(fshp)
    name = "fpatch"

    # Make target file
    field = "lat"
    fshp_targets = os.path.splitext(fshp)[0] + "_" + field + ".hdf5"
    ctx = Context(maketargets)
    ctx.forward(maketargets, shapefile=fshp, fieldname=field,
                outfile=fshp_targets)

    # Extract features from gtiff
    ctx = Context(extractfeats)
    ctx.forward(extractfeats, geotiff=ftif, name=name, outputdir=outdir,
                targets=fshp_targets)

    # Get the 4 parts
    feat_list = []
    for i in range(4):
        fname = name + ".part{}.hdf5".format(i)
        with tables.open_file(os.path.join(outdir, fname), 'r') as f:
            feat_list.append(f.root.features[:])
    feats = np.concatenate(feat_list, axis=0)

    # Read lats and lons from targets
    with tables.open_file(fshp_targets, mode='r') as f:
        lonlat = np.hstack((f.root.Longitude.read(),
                            f.root.Latitude.read()))
        permutation = f.root.Indices.read()

    lonlat_p = lonlat[permutation]

    assert np.allclose(feats, lonlat_p)