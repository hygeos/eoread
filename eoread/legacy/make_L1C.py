#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from sys import argv
import shutil
import re
import subprocess
from tempfile import TemporaryDirectory
from typing import Literal, Union
from core.files.fileutils import filegen
from core.env import getdir


"""
NASA level1 1A or 1B products (MODIS, SeaWiFS, VIIRS) are lacking radiometric
information (polariztion correction). This modules runs l2gen
(https://oceancolor.gsfc.nasa.gov/resources/docs/tutorials/notebooks/oci-ocssw-processing/)
without atmospheric correction, to generate so-called L1C products including the
polarization correction.

l2gen runs either as a standard shell command, or in a docker container.
"""


def makeL1C(
    l1a: Path | str,
    dirname: Path | None = None,
    method: Literal["docker", "shell", "auto"] = "shell",
    **kwargs
) -> Path:
    """
    Generate a Level-1C product from NASA L1A/L1B data using SeaDAS OCSSW tools.
    
    This function automatically detects the sensor type from the filename and
    dispatches to the appropriate sensor-specific L1C generation function. L1C
    products include polarization-corrected reflectances without atmospheric
    correction, generated using NASA's l2gen processor.
    
    Supported sensors:
    - MODIS (Aqua/Terra): Files starting with 'A'
    - VIIRS (SNPP/JPSS-1): Files starting with 'V' or 'JPSS1_VIIRS'
    - HAWKEYE (SeaHawk-1): Files starting with 'SEAHAWK1_HAWKEYE'
    - SeaWiFS: Files starting with 'S'

    Parameters
    ----------
    l1a : Path or str
        Path to the Level-1A or Level-1B input product.
    dirname : Path, optional
        Directory for output L1C file. If None, uses same directory as input.
    method : str, optional
        Execution method - "shell" runs OCSSW commands directly (requires
        OCSSWROOT environment variable), "docker" runs via container, "auto"
        automatically selects based on executable availability.
    **kwargs : dict
        Additional arguments passed to l2gen processor.

    Returns
    -------
    Path
        Path to the generated L1C product file.
        
    Raises
    ------
    ValueError
        If the sensor type cannot be determined from the filename.
        
    Examples
    --------
    >>> l1c = makeL1C('A2023001000000.L1A_LAC', method='shell')
    >>> print(l1c)
    A2023001000000.L1C
    """
    l1a = Path(l1a)
    # sensor switch
    if l1a.name.startswith("A"):
        return makeL1C_MODIS(l1a=l1a, dirname=dirname, method=method, **kwargs)
    elif l1a.name.startswith("V") or l1a.name.startswith("JPSS1_VIIRS"):
        return makeL1C_VIIRS(l1a=l1a, dirname=dirname, method=method, **kwargs)
    elif l1a.name.startswith("SEAHAWK1_HAWKEYE"):
        return makeL1C_HAWKEYE(l1a=l1a, dirname=dirname, method=method, **kwargs)
    elif l1a.name.startswith("S"):
        return makeL1C_SeaWIFS(l1a=l1a, dirname=dirname, method=method, **kwargs)
    else:
        raise ValueError(f"Invalid sensor in makeL1C ({l1a.name})")


def makeL1C_HAWKEYE(l1a: Path, dirname: Path | None, method: str, **kwargs) -> Path:
    """
    Generate Level-1C product from SeaHawk-1 HAWKEYE L1A data.
    
    Creates a geolocation file (GEO) and runs l2gen to produce polarization-
    corrected reflectances. The GEO file is generated in a temporary directory
    and removed after processing.

    Parameters
    ----------
    l1a : Path
        Path to HAWKEYE Level-1A input file.
    dirname : Path, optional
        Output directory for L1C file, or None to use input directory.
    method : str
        Execution method ("shell", "docker", or "auto").
    **kwargs : dict
        Additional arguments passed to l2gen.

    Returns
    -------
    Path
        Path to the generated L1C product.
    """
    l1c = (dirname or l1a.parent) / l1a.name.replace("L1A", "L1C")
    with TemporaryDirectory() as tmpdir:
        geo = Path(tmpdir) / (l1a.stem + ".GEO")
        
        if not l1c.exists():
            make_HAWKEYE_GEO(l1a=l1a, geo=geo, method=method)
            assert geo.exists()

        run_l2gen_L1C(ifile=l1a, l1c=l1c, nbands=8, method=method, geofile=geo, **kwargs)

    return l1c

def makeL1C_MODIS(l1a: Path, dirname: Path | None, method: str, **kwargs) -> Path:
    """
    Generate Level-1C product from MODIS (Aqua/Terra) L1A data.
    
    Creates both geolocation (GEO) and calibrated radiance (L1B) files as
    intermediate products, then runs l2gen to produce the L1C output with
    polarization-corrected reflectances. Temporary files are created in a
    temporary directory and automatically cleaned up.

    Parameters
    ----------
    l1a : Path
        Path to MODIS Level-1A input file.
    dirname : Path, optional
        Output directory for L1C file, or None to use input directory.
    method : str
        Execution method ("shell", "docker", or "auto").
    **kwargs : dict
        Additional arguments passed to l2gen.

    Returns
    -------
    Path
        Path to the generated L1C product.
    """
    if dirname is None:
        dname = l1a.parent
    else:
        dname = Path(dirname)

    with TemporaryDirectory() as tmpdir:
        geo = Path(tmpdir) / (l1a.stem + ".GEO")
        l1b = Path(tmpdir) / (l1a.stem + ".L1B_LAC")
        l1c = dname / (l1a.stem + ".L1C")

        # Generate GEO and L1B files (temporary)
        if not l1c.exists():
            make_MODIS_GEO(l1a=l1a, geo=geo, method=method)
            assert geo.exists()
            make_L1B_MODIS(l1a=l1a, l1b=l1b, geo=geo, method=method)

        run_l2gen_L1C(
            ifile=l1b,
            l1c=l1c,
            geofile=geo,
            nbands=16,
            method=method,
        )

    return l1c


@filegen(arg="l1b", if_exists="skip")
def make_L1B_MODIS(l1a: Path, l1b: Path, geo: Path, method: str):
    """
    Generate MODIS Level-1B calibrated radiances from L1A data.
    
    Runs the modis_L1B processor from OCSSW to calibrate raw MODIS data.
    Uses the @filegen decorator to skip processing if output already exists.

    Parameters
    ----------
    l1a : Path
        Path to MODIS L1A input file.
    l1b : Path
        Path for output L1B calibrated radiance file.
    geo : Path
        Path to MODIS geolocation file (required input).
    method : str
        Execution method ("shell", "docker", or "auto").
        
    Raises
    ------
    RuntimeError
        If modis_L1B processing fails.
    AssertionError
        If L1B file is not created successfully (shell mode).
    """
    if method == "auto":
        method = get_method_auto("modis_L1B")
    if method == "shell":
        cmd = get_prefix()
        cmd += f"modis_L1B -y -z --okm={l1b} {l1a} {geo}"
        print(cmd)
        result = subprocess.run(cmd, shell=True, executable='/bin/bash')
        if result.returncode != 0:
            raise RuntimeError("Error in modis_L1B")
        assert l1b.exists()
    else:
        # docker
        cmd = "modis_L1B -y -z --okm={l1b} {l1a}"
        print(cmd)
        app = get_ocssw_docker_app()
        app.run(
            cmd=cmd,
            l1a=l1a,
            l1b=l1b,
        )


@filegen(arg="geo", if_exists="skip")
def make_MODIS_GEO(l1a: Path, geo: Path, method: str):
    """
    Generate MODIS geolocation file from L1A data.
    
    Runs the modis_GEO processor from OCSSW to compute pixel-level geolocation
    (latitude, longitude, view angles, solar angles). Uses the @filegen decorator
    to skip processing if output already exists.

    Parameters
    ----------
    l1a : Path
        Path to MODIS L1A input file.
    geo : Path
        Path for output geolocation file.
    method : str
        Execution method ("shell", "docker", or "auto").
        
    Raises
    ------
    RuntimeError
        If modis_GEO processing fails.
    AssertionError
        If GEO file is not created successfully (shell mode).
    """
    if method == "auto":
        method = get_method_auto("modis_GEO")
    if method == "shell":
        cmd = get_prefix()
        cmd += f"modis_GEO --output={geo} {l1a}"
        print(cmd)
        result = subprocess.run(cmd, shell=True, executable='/bin/bash')
        if result.returncode != 0:
            raise RuntimeError("Error in modis_GEO")
        assert geo.exists()
    else:
        # Docker
        cmd = "modis_GEO --output={geo} {l1a}"
        print(cmd)
        app = get_ocssw_docker_app()
        app.run(
            cmd=cmd,
            l1a=l1a,
            geo=geo,
        )


def makeL1C_VIIRS(l1a: Path, dirname: Path | None, method: str, **kwargs) -> Path:
    """
    Generate Level-1C product from VIIRS (SNPP or JPSS-1) L1A data.
    
    Creates a geolocation file (GEO-M) and runs l2gen to produce polarization-
    corrected reflectances. Handles multiple VIIRS filename conventions for
    both SNPP and JPSS-1 platforms. The GEO file is created in a temporary
    directory and automatically cleaned up.

    Parameters
    ----------
    l1a : Path
        Path to VIIRS Level-1A input file (SNPP or JPSS-1).
    dirname : Path, optional
        Output directory for L1C file, or None to use input directory.
    method : str
        Execution method ("shell", "docker", or "auto").
    **kwargs : dict
        Additional arguments passed to l2gen.

    Returns
    -------
    Path
        Path to the generated L1C product.
        
    Raises
    ------
    RuntimeError
        If filename doesn't match expected VIIRS conventions.
    """
    if dirname is None:
        dname = l1a.parent
    else:
        dname = Path(dirname)


    with TemporaryDirectory() as tmpdir:
        
        bname = re.sub(r'\.L1A.*\.nc$', '', l1a.name)
        l1c = dname / (bname + '.L1C.nc')
        l1b = Path(tmpdir) / (bname + '.L1B.nc')
        geo = Path(tmpdir) / (bname + '.GEO.nc')

        # Generate GEO files
        if not l1c.exists():
            make_VIIRS_GEO(l1a=l1a, geo=geo, method=method)
            assert geo.exists()
        
        # Generate VIIRS L1B
        if not l1c.exists():
            make_L1B_VIIRS(l1a=l1a, l1b=l1b, method=method)
            assert l1b.exists()

        run_l2gen_L1C(
            ifile=l1b,
            l1c=l1c,
            geofile=geo,
            nbands=10,
            method=method,
        )

    return l1c


@filegen(arg="geo", if_exists="skip")
def make_VIIRS_GEO(l1a: Path, geo: Path, method: str):
    """
    Generate VIIRS moderate-resolution geolocation file from L1A data.
    
    Runs the geolocate_viirs processor from OCSSW to compute pixel-level
    geolocation for VIIRS moderate resolution bands. Uses the @filegen
    decorator to skip processing if output already exists.

    Parameters
    ----------
    l1a : Path
        Path to VIIRS L1A input file.
    geo : Path
        Path for output GEO-M geolocation file.
    method : str
        Execution method ("shell", "docker", or "auto").
        
    Raises
    ------
    RuntimeError
        If geolocate_viirs processing fails.
    """
    if method == "auto":
        method = get_method_auto("geolocate_viirs")
    if method == "docker":
        app = get_ocssw_docker_app()
        cmd = "geolocate_viirs ifile={l1a} geofile_mod={geo}"
        print(cmd)
        app.run(
            cmd=cmd,
            l1a=l1a,
            geo=geo,
        )
    else:
        cmd = get_prefix()
        cmd += f"geolocate_viirs ifile={l1a} geofile_mod={geo}"
        print(cmd)
        result = subprocess.run(cmd, shell=True, executable='/bin/bash')
        if result.returncode != 0:
            raise RuntimeError("Error in genL1C_VIIRS")


@filegen(arg="l1b", if_exists="skip")
def make_L1B_VIIRS(l1a: Path, l1b: Path, method: str):
    """
    Generate VIIRS Level-1B calibrated radiances from L1A data.
    
    Runs the calibrate_viirs processor from OCSSW to calibrate raw VIIRS data.
    Uses the @filegen decorator to skip processing if output already exists.

    Parameters
    ----------
    l1a : Path
        Path to VIIRS L1A input file.
    l1b : Path
        Path for output L1B calibrated radiance file.
    method : str
        Execution method ("shell", "docker", or "auto").
        
    Raises
    ------
    RuntimeError
        If calibrate_viirs processing fails.
    """
    if method == "auto":
        method = get_method_auto("calibrate_viirs")
    if method == "docker":
        app = get_ocssw_docker_app()
        cmd = "calibrate_viirs ifile={l1a} l1bfile_mod={l1b}"
        print(cmd)
        app.run(
            cmd=cmd,
            l1a=l1a,
            l1b=l1b,
        )
    else:
        cmd = get_prefix()
        cmd += f"calibrate_viirs ifile={l1a} l1bfile_mod={l1b}"
        print(cmd)
        result = subprocess.run(cmd, shell=True, executable='/bin/bash')
        if result.returncode != 0:
            raise RuntimeError("Error in make_L1B_VIIRS")


@filegen(arg="geo", if_exists="skip")
def make_HAWKEYE_GEO(l1a: Path, geo: Path, method: str):
    """
    Generate SeaHawk-1 HAWKEYE geolocation file from L1A data.
    
    Runs the geolocate_hawkeye processor from OCSSW to compute pixel-level
    geolocation. Uses the @filegen decorator to skip processing if output
    already exists.

    Parameters
    ----------
    l1a : Path
        Path to HAWKEYE L1A input file.
    geo : Path
        Path for output geolocation file.
    method : str
        Execution method ("shell", "docker", or "auto").
        
    Raises
    ------
    RuntimeError
        If geolocate_hawkeye processing fails.
    """
    if method == "auto":
        method = get_method_auto("geolocate_hawkeye")
    if method == "docker":
        app = get_ocssw_docker_app()
        cmd = "geolocate_hawkeye {l1a} {geo}"
        print(cmd)
        app.run(
            cmd=cmd,
            l1a=l1a,
            geo=geo,
        )
    else:
        cmd = get_prefix()
        cmd += f"geolocate_hawkeye {l1a} {geo}"
        print(cmd)
        result = subprocess.run(cmd, shell=True, executable='/bin/bash')
        if result.returncode != 0:
            raise RuntimeError("Error in make_HAWKEYE_GEO")


def makeL1C_SeaWIFS(l1a: Path, dirname: Path | None, method: str, **kwargs) -> Path:
    """
    Generate Level-1C product from SeaWiFS L1A data.
    
    Runs l2gen directly on SeaWiFS data to produce polarization-corrected
    reflectances. SeaWiFS L1A files already contain geolocation information,
    so no separate GEO file is needed.

    Parameters
    ----------
    l1a : Path
        Path to SeaWiFS Level-1A input file.
    dirname : Path, optional
        Output directory for L1C file, or None to use input directory.
    method : str
        Execution method ("shell", "docker", or "auto").
    **kwargs : dict
        Additional arguments passed to l2gen.

    Returns
    -------
    Path
        Path to the generated L1C product.
    """
    if dirname is None:
        dname = l1a.parent
    else:
        dname = Path(dirname)
    l1c = dname / (l1a.stem + ".L1C")

    run_l2gen_L1C(ifile=l1a, l1c=l1c, nbands=8, method=method, **kwargs)

    return l1c


def get_method_auto(executable: str) -> str:
    """
    Automatically determine execution method based on executable availability.
    
    Checks if the specified OCSSW executable is available in the system PATH.
    Returns "shell" if found (for direct execution), otherwise returns "docker"
    to run via container.

    Parameters
    ----------
    executable : str
        Name of the OCSSW executable to check (e.g., 'l2gen', 'modis_GEO').

    Returns
    -------
    str
        "shell" if executable is found in PATH, "docker" otherwise.
    """
    if shutil.which(executable) is not None:
        return "shell"
    else:
        return "docker"

def get_ocssw_executable(executable: str) -> str:
    """
    Get the full path to an OCSSW executable from OCSSWROOT.
    
    Constructs the path to an OCSSW binary using the OCSSWROOT environment
    variable and verifies that the executable exists.

    Parameters
    ----------
    executable : str
        Name of the OCSSW executable (e.g., 'l2gen', 'modis_GEO').

    Returns
    -------
    str
        Full path to the executable.
        
    Raises
    ------
    AssertionError
        If the executable is not found or OCSSWROOT is not set.
    """
    exe = getdir("OCSSWROOT") / "bin" / executable
    assert shutil.which(exe) is not None, (
        f"Could not find {executable}, please check the environment variable OCSSWROOT"
    )
    return str(exe)

def get_prefix() -> str:
    """
    Generate shell command prefix to set up OCSSW environment.
    
    Creates a command prefix that sets OCSSWROOT and sources the OCSSW
    environment configuration file. This prefix should be prepended to
    OCSSW commands when running in shell mode.

    Returns
    -------
    str
        Shell command prefix string to initialize OCSSW environment.
    """
    root = getdir("OCSSWROOT")
    cmd = f"OCSSWROOT={root} && source {root}/OCSSW_bash.env && "
    return cmd


def l2gen_cmdline(
    nbands: int,
    geofile: bool,
    prefix: bool,
    **kwargs
) -> str:
    """
    Construct l2gen command line for L1C generation.
    
    Builds the l2gen command with appropriate parameters for generating L1C
    products: polarization-corrected reflectances (rhot_nnn, polcor_nnn),
    geometry (sensor/solar angles), and geolocation, with atmospheric
    correction disabled.

    Parameters
    ----------
    nbands : int
        Number of spectral bands in the sensor.
    geofile : bool
        Whether a separate geolocation file is used.
    **kwargs
        Additional l2gen parameters (key=value pairs).

    Returns
    -------
    str
        Complete l2gen command line string with placeholders for file paths.
        Placeholders: {ifile}, {ofile}, and optionally {geofile}.
    """
    gains = " ".join(["1.0"] * nbands)

    cmd = ''
    if prefix:
        cmd += get_prefix()
    cmd += 'l2gen ifile="{ifile}" ofile="{ofile}" oformat="netcdf4" '
    if geofile:
        cmd += 'geofile="{geofile}" '
    cmd += 'l2prod="rhot_nnn polcor_nnn sena senz sola solz latitude longitude" '
    cmd += f'gain="{gains}" atmocor=0 aer_opt=-99 brdf_opt=0'

    cmd += " " + " ".join([f"{k}={v}" for k, v in kwargs.items()])

    return cmd


def get_ocssw_docker_app(
    version: Union[str, None] = None, tags: Union[list, None] = None
):
    """
    Initialize OCSSW Docker application instance.
    
    Creates an OCSSW hydro application object for running OCSSW processors
    in Docker containers. Requires the hydro package for container management.

    Parameters
    ----------
    version : str, optional
        Specific OCSSW version to use, or None for default.
    tags : list, optional
        List of Docker image tags to use, or None for default.

    Returns
    -------
    OCSSW
        OCSSW hydro application instance configured for the specified version/tags.
    """
    from hydro.apps.OCSSW.OCSSW import OCSSW
    app = OCSSW(version=version, tags=tags)
    return app


@filegen(arg="l1c", if_exists="skip")
def run_l2gen_L1C(
    ifile: Path,
    l1c: Path,
    nbands: int,
    method: str,
    geofile: None | Path = None,
    **kwargs
):
    """
    Run l2gen to generate Level-1C product with polarization correction.
    
    Executes NASA's l2gen processor without atmospheric correction to produce
    L1C products containing polarization-corrected reflectances, viewing/solar
    geometry, and geolocation. Uses the @filegen decorator to skip processing
    if output already exists.

    Parameters
    ----------
    ifile : Path
        Path to input L1A or L1B file.
    l1c : Path
        Path for output L1C file.
    nbands : int
        Number of spectral bands in the sensor.
    method : str
        Execution method ("shell", "docker", or "auto").
    geofile : Path, optional
        Path to geolocation file, or None if geolocation is embedded.
    **kwargs
        Additional parameters passed to l2gen.
        
    Raises
    ------
    ValueError
        If method is not "shell", "docker", or "auto".
    RuntimeError
        If l2gen processing fails (shell mode).
    """
    # run the command
    print("L1A/B:", ifile)
    print("L1C:", l1c)
    if method == "auto":
        method = get_method_auto("l2gen")
    if method == "docker":
        app = get_ocssw_docker_app()
        if geofile is None:
            cmd = l2gen_cmdline(nbands, geofile=False, prefix=False, **kwargs)
            app.run(
                cmd=cmd,
                ifile=ifile,
                ofile=l1c,
            )
        else:
            cmd = l2gen_cmdline(nbands, geofile=True, prefix=False, **kwargs)
            app.run(
                cmd=cmd,
                ifile=ifile,
                ofile=l1c,
                geofile=geofile
            )
        print("CMD:", cmd)

    elif method == "shell":
        if geofile is None:
            cmd = l2gen_cmdline(nbands, geofile=False, prefix=True, **kwargs).format(ifile=ifile, ofile=l1c)
        else:
            cmd = l2gen_cmdline(nbands, geofile=True, prefix=True, **kwargs).format(
                ifile=ifile, ofile=l1c, geofile=geofile
            )
        print("CMD:", cmd)
        result = subprocess.run(cmd, shell=True, executable='/bin/bash')
        if result.returncode != 0:
            raise RuntimeError(f'Error running command "{cmd}"')
    else:
        raise ValueError(f"Invalid method {method}")


if __name__ == "__main__":
    for l1a in argv[1:]:
        makeL1C(l1a)