from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import botocore
import csv
import concurrent.futures
import io
import json
import os
import xml.etree.ElementTree as ET
import xmltodict

from pupil.s3 import get_s3_client

def download_assets(iterable):
    client = get_s3_client()

    for item in iterable():
        if item is not None and not os.path.exists(item.filename):
            dirname = os.path.dirname(item.filename)

            if not os.path.exists(dirname):
                print("Making directories to {}".format(dirname))
                os.makedirs(dirname)

    def conditionally_download_item(item):
        if item is not None and not os.path.exists(item.filename):
            print("Downloading asset to {}".format(item.filename))
            client.client.download_file(item.bucket, item.path,item.filename)
        else:
            print("Asset {} already exists, skipping download".format(item.filename))

    thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=min(os.cpu_count() * 2, 24))
    futures = []

    # Download scans in parallel
    for item in iterable():
        future = thread_pool.submit(conditionally_download_item, item)
        futures.append(future)

    _ = concurrent.futures.wait(futures)

    thread_pool.shutdown()

def object_exists_s3(bucket, path):
    client = get_s3_client()
    try:
        response = client.client.head_object(Bucket=bucket, Key=path)
        return response["ContentLength"] > 0
    except botocore.exceptions.ClientError as error:
        if error.response["Error"]["Code"] == "404":
            # The object does not exist.
            print("Warning: s3://{}/{} not found".format(bucket, path))
            return False
        else:
            # Something else has gone wrong.
            raise error

def load_json_from_file(filename):
    with io.open(filename, encoding="utf-8") as json_file:
        return json.load(json_file)

def write_json_to_file(filename, json_data, encoder=None):
    with io.open(filename, "w", encoding="utf-8") as json_file:
        if encoder is not None:
            json.dump(json_data, json_file, indent=4, separators=(',', ': '), cls=encoder)
        else:
            json.dump(json_data, json_file, indent=4, separators=(',', ': '))

def write_json_to_string(json_data):
    return json.dumps(json_data, indent=4, separators=(',', ': '))

def load_json_from_s3(bucket, path):
    client = get_s3_client()

    s3_bucket = client.s3.Bucket(bucket)
    s3_object = s3_bucket.Object(path)

    with io.BytesIO(s3_object.get()["Body"].read()) as stream:
        with io.TextIOWrapper(stream, encoding="utf-8") as json_file:
            return json.load(json_file)

def load_xml_from_file(filename):
    with io.open(filename, "rb") as xml_file:
        return xmltodict.parse(xml_file)

def load_xml_from_s3(bucket, path):
    client = get_s3_client()
    s3_bucket = client.s3.Bucket(bucket)
    s3_object = s3_bucket.Object(path)

    with io.BytesIO(s3_object.get()["Body"].read()) as stream:
        with io.TextIOWrapper(stream, encoding="utf-8") as xml_file:
            return xmltodict.parse(xml_file.read())

def load_svg_from_file(filename):
    return ET.parse(filename)

def load_svg_from_s3(bucket, path):
    client = get_s3_client()
    s3_bucket = client.s3.Bucket(bucket)
    s3_object = s3_bucket.Object(path)

    with io.BytesIO(s3_object.get()["Body"].read()) as stream:
        return ET.parse(stream)

_EXT_METADATA_MAP = {
    ".csv": "text/csv",
    ".exr": "binary/octet-stream",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".json": "application/json",
    ".png": "image/png"
}

def download_file(bucket, path, filename):
    client = get_s3_client()
    client.client.download_file(bucket, path, filename)

def upload_file(filename, bucket, path, **kwargs):
    _, ext = os.path.splitext(path)
    if ext in _EXT_METADATA_MAP:
        if "extra_args" not in kwargs:
            kwargs["extra_args"] = {}

        kwargs["extra_args"]["ContentType"] = _EXT_METADATA_MAP[ext]

    client = get_s3_client()
    client.transfer.upload_file(filename, bucket, path, **kwargs)

def write_to_csv(filename, data):
    with io.open(filename, "w", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, dialect="excel")
        writer.writerows(data)

def append_to_csv(filename, data):
    with io.open(filename, "a", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, dialect="excel")
        writer.writerows(data)