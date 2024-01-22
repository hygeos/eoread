#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import fs
from fs.osfs import OSFS
from fs.base import FS
from .utils.uncompress import uncompress as uncomp


class Mirror_Uncompress:
    """
    Locally map a remote filesystem, with lazy file access and archive decompression.
    
    remote_fs is opened by fs.open_fs (see docs.pyfilesystem.org)
        Examples:
            'ftp://<user>:<password>@<server>/<path>'
            '<path>'
            'osfs://<path>'
        ... or a FS (FTPFS, OSFS, etc)
    
    uncompress: comma-separated string of extensions to uncompress.
        Set to '' to disactivate decompression.
    
    Example:
        mfs = MirrorFS(FTPFS(...).opendir(...), '.')
        for p in mfs.glob('*.zip'):
            mfs.get(p)
    """
    def __init__(self,
                 remote_fs,
                 local_fs,
                 uncompress: str='.tar.gz,.zip,.gz,.Z',
                 ) -> None:
        self.remote_fs = remote_fs
        self.local_fs = local_fs
        self.local = None
        self.remote = None
        self.uncompress = uncompress.split(',')
        self.uncompress.append('')  # case where file is not compressed !
    
    def get_local(self) -> FS:
        if self.local is None:
            self.local = fs.open_fs(self.local_fs)
        
        return self.local
    
    def get_remote(self) -> FS:
        if self.remote is None:
            self.remote = fs.open_fs(self.remote_fs)
        
        return self.remote
    
    def glob(self, pattern: str):
        """
        pattern: remote pattern
        """
        for p in self.get_remote().glob(pattern):
            yield p.path
        
    def find(self, pattern):
        """
        finds and returns a unique path from pattern
        """
        # find local
        ls = list(self.get_local().glob(pattern))
        if len(ls) == 1:
            return ls[0].path
        
        # find remote
        ls = list(self.get_remote().glob(pattern))
        if len(ls) != 1:
            raise FileNotFoundError(f'Query on {self.remote_fs} did not lead to a single file ({pattern}) -> {ls}')

        return ls[0].path
    
    def get(self, path):
        """
        Get a path, and optionally does decompression if needed
        If path ends by an `uncompress` extension, this extension is stripped.

        Returns the absolute local path
        """
        path_local, path_remote = None, None
        for p in self.uncompress:
            # check whether path has been provided as a remote path
            if path.endswith(p):
                path_remote = path
                path_local = path[:-len(p)]
                break

        # if path has been provided as local
        # (path_remote may still be undetermined)
        path_local = path_local or path
        
        if not self.get_local().exists(path_local):
            # get local path
            if path_remote is None:
                for p in self.uncompress:
                    if self.get_remote().exists(path_local+p):
                        path_remote = path_local+p
                        break
            assert self.get_remote().exists(path_remote), \
                f'{path_remote} does not exist on {self.get_remote()}'
            path_tmp = path_local+'.tmp'

            if path_local == path_remote:
                # no compression
                if self.get_remote().isdir(path_remote):
                    copy = fs.copy.copy_dir
                else:
                    copy = fs.copy.copy_file

                copy(
                    self.get_remote(), path_remote,
                    self.get_local(), path_tmp)
            else:
                # apply decompression
                with TemporaryDirectory() as tmpdir:
                    Path(OSFS(tmpdir).getsyspath(path_remote)).parent.mkdir(parents=True, exist_ok=True)
                    fs.copy.copy_file(
                        self.get_remote(), path_remote,
                        OSFS(tmpdir), path_remote
                        )
                    u = uncomp(OSFS(tmpdir).getsyspath(path_remote),
                               Path(self.get_local().getsyspath(path_tmp)).parent)
                    shutil.move(u, self.get_local().getsyspath(path_tmp))
            shutil.move(self.get_local().getsyspath(path_tmp),
                        self.get_local().getsyspath(path_local))

        path_final = self.get_local().getsyspath(path_local)
        assert Path(path_final).exists()
        return path_final