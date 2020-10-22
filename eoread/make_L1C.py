#!/usr/bin/env python3
# -*- coding: utf-8 -*-




from pathlib import Path
import os
from sys import argv
from tempfile import TemporaryDirectory
from .misc import safe_move


def makeL1C(filename_l1a, dirname=None):
    """
    Generate L1C product using SeaDAS

    dirname: path to target directory (default None: same directory as l1a)

    Returns the path to the new product
    """
    l1a = Path(filename_l1a)
    assert l1a.exists()
    if dirname is None:
        dname = l1a.parent
    else:
        dname = Path(dirname)

    # sensor switch
    if l1a.name.startswith('A'):
        return makeL1C_MODIS(l1a, dname)
    elif l1a.name.startswith('V'):
        return makeL1C_VIIRS(l1a, dname)
    elif l1a.name.startswith('S'):
        return makeL1C_SeaWiFS(l1a, dname)
    else:
        raise Exception(f'Invalid sensor in genL1C ({l1a.name})')


def makeL1C_MODIS(l1a, dirname):
    assert str(l1a).endswith('.L1A_LAC')
    with TemporaryDirectory('tmp_eoread_L1C_') as tmpdir:
        geo = Path(tmpdir)/(l1a.stem+'.GEO')
        l1b = Path(tmpdir)/(l1a.stem+'.L1B_LAC')
        l1c = dirname/(l1a.stem+'.L1C')

        if l1c.exists():
            print(f'Skipping existing {l1c}')
        else:
            # gen GEO
            cmd = f'modis_GEO.py --output={geo} {l1a}'
            print(cmd)
            if os.system(cmd):
                raise Exception('Error in modis_GEO')
            assert geo.exists()

            # gen l1b
            cmd = f'modis_L1B.py -y -z --okm={l1b} {l1a} {geo}'
            print(cmd)
            if os.system(cmd):
                raise Exception('Error in modis_L1B.py')
            assert l1b.exists()

            # gen L1C
            run_l2gen_L1C(ifile=l1b,
                          l1c=l1c,
                          geofile=geo,
                          nbands=16)

    return l1c


def makeL1C_VIIRS(l1a, dirname):
    with TemporaryDirectory(prefix='tmp_eoread_L1C_') as tmpdir:
        if str(l1a).endswith('.L1A_SNPP.nc'):
            l1c = dirname/(l1a.name.replace('.L1A_SNPP.nc', '.L1C'))
            geo = Path(tmpdir)/(l1a.name.replace('.L1A_SNPP.nc', '.GEO-M_SNPP.nc'))
        elif str(l1a).endswith('.L1A_JPSS1.nc'):
            l1c = dirname/(l1a.name.replace('.L1A_JPSS1.nc', '.L1C'))
            geo = Path(tmpdir)/(l1a.name.replace('.L1A_JPSS1.nc', '.GEO-M_JPSS1.nc'))
        else:
            raise Exception(f'genL1C_VIIRS: invalid file name {l1a}')

        if l1c.exists():
            print(f'Skipping existing {l1c}')
        else:
            # gen GEO
            cmd = f'geolocate_viirs ifile={l1a} geofile_mod={geo}'
            if os.system(cmd):
                raise Exception('Error in genL1C_VIIRS')

            run_l2gen_L1C(ifile=l1a,
                          l1c=l1c,
                          geofile=geo,
                          nbands=10)

    return l1c


def makeL1C_SeaWiFS(l1a, dirname):
    l1c = dirname/(l1a.stem+'.L1C')
    if l1c.exists():
        print(f'Skipping existing {l1c}')
    else:
        run_l2gen_L1C(ifile=l1a,
                      l1c=l1c,
                      nbands=8)
    
    return l1c


def run_l2gen_L1C(ifile, l1c, nbands, geofile=None):
    with TemporaryDirectory(prefix='tmp_eoread_L1C_') as tmpdir:
        gains = ' '.join(['1.0']*nbands)
        l1c_tmp = Path(tmpdir)/l1c.name

        cmd = f'l2gen ifile="{ifile}" ofile="{l1c_tmp}" oformat="netcdf4" '
        if geofile is not None:
            cmd += f'geofile="{geofile}" '
        cmd += f'l2prod="rhot_nnn polcor_nnn sena senz sola solz latitude longitude" '
        cmd += f'gain="{gains}" atmocor=0 aer_opt=-99 brdf_opt=0'

        # run the command
        print('L1A/B:', ifile)
        print('L1C:', l1c)
        print('CMD:', cmd)
        if os.system(cmd):
            raise Exception(f'Error running command "{cmd}"')

        assert l1c_tmp.exists()

        safe_move(l1c_tmp, l1c)
    
    assert l1c.exists()


if __name__ == "__main__":
    for l1a in argv[1:]:
        makeL1C(l1a)
