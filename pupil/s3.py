from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import boto3
import uuid

from boto3.s3.transfer import S3Transfer

class S3Client:
    def __init__(self):
        self.s3 = boto3.resource("s3")
        self.client = boto3.client("s3")
        self.transfer = S3Transfer(self.client)

    def close(self):
        del self.s3
        del self.client
        del self.transfer

cache = {}

def get_s3_client():
    if "client" not in cache:
        cache["client"] = S3Client()

    return cache["client"]

def close_s3_client():
    if "client" in cache:
        client = get_s3_client()
        client.close()
        del cache["client"]