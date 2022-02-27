#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from contextlib import contextmanager
from ftplib import FTP
import keyring

class Credentials:
    """
    Facilitates the access to credentials through the keyring library

    ex: Credentials('my_ftp_server').password()

    Credentials are stored by keyring, for a given <id>, like so:
        keyring set <id> username
        keyring set <id> password
    """
    def __init__(self, entry_id) -> None:
        self.entry_id = entry_id
    
    def get(self, field):
        """
        Gets a field in the current entry_id

        (field is generally: username, password, url)
        """
        ret = keyring.get_password(self.entry_id, field)

        if ret is None:
            raise ValueError(
                f'{field} is not stored for {self.entry_id}. '
                f'Please provide it by: `keyring set {self.entry_id} {field}`')

        return ret

    def username(self):
        return self.get('username')

    def password(self):
        return self.get('password')

    def url(self):
        return self.get('url')

    def copernicus(self, url=None):
        """
        returns copernicus hub credentials to be passed to sentinelapi:

        SentinelAPI(**Credentials('scihub').copernicus())
        """

        if url is None:
            # if url is not provided, determine it from entry_id
            url = {
                'cophub': 'https://cophub.copernicus.eu/dhus/',
                'coda': 'https://coda.eumetsat.int',
                'codarep': 'https://codarep.eumetsat.int',
                'scihub': 'https://scihub.copernicus.eu/dhus/',
            }.get(self.entry_id)

        return {
            'user': self.username(),
            'password': self.password(),
            'api_url': url or self.url(),
        }

    @contextmanager
    def ftp(self, url=None):
        """
        Returns an ftp instance

        with Credentials('my_ftp_server').ftp('ftp.address.com'):
            ftp.cwd('dir/name/')
            ftp.nlst()
        """
        yield FTP(url or self.url(),
                  user=self.username(),
                  passwd=self.password())
