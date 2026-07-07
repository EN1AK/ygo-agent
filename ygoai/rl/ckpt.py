from typing import List

import os
import shutil
import subprocess
from pathlib import Path
import zipfile


class ModelCheckpoint(object):
    """ ModelCheckpoint handler can be used to periodically save objects to disk.

    Args:
        dirname (str):
            Directory path where objects will be saved.
        save_fn (callable):
            Function that will be called to save the object. It should have the signature `save_fn(obj, path)`.
        n_saved (int, optional):
            Number of objects that should be kept on disk. Older files will be removed.
    """

    def __init__(self, dirname, save_fn, n_saved=1):
        self._dirname = Path(dirname).expanduser().absolute()
        self._n_saved = n_saved
        self._save_fn = save_fn
        self._saved: List[Path] = []

    def _check_dir(self):
        self._dirname.mkdir(parents=True, exist_ok=True)

        # Ensure that dirname exists
        if not self._dirname.exists():
            raise ValueError(
                "Directory path '{}' is not found".format(self._dirname))

    def save(self, obj, name):
        self._check_dir()
        path = self._dirname / name
        self._save_fn(obj, str(path))
        self._saved.append(path)
        print(f"Saved model to {path}")

        if len(self._saved) > self._n_saved:
            to_remove = self._saved.pop(0)
            if to_remove != path:
                if to_remove.is_dir():
                    shutil.rmtree(to_remove)
                else:
                    if to_remove.exists():
                        os.remove(to_remove)
    
    def get_latest(self):
        path = self._saved[-1]
        return path


def sync_to_gcs(bucket, source, dest=None):
    if bucket.startswith("gs://"):
        bucket = bucket[5:]
    source_path = Path(source).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"GCS sync source not found: {source_path}")
    if dest is None:
        dest = source_path.name
    gcs_path = "/".join(part.strip("/") for part in (bucket, str(dest)) if part)
    gcs_url = f"gs://{gcs_path}"
    with open(os.devnull, "wb") as devnull:
        subprocess.Popen(
            ["gsutil", "cp", str(source_path), gcs_url],
            stdout=devnull,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
    print(f"Sync to GCS: {gcs_url}")


def zip_files(zip_file_path, files_to_zip):
    """
    Creates a zip file at the specified path, containing the files and directories
    specified in files_to_zip.

    Args:
        zip_file_path (str): The path to the zip file to be created.
        files_to_zip (list): A list of paths to files and directories to be zipped.
    """
    with zipfile.ZipFile(zip_file_path, mode='w') as zip_file:
        for file_path in files_to_zip:
            # Check if the path is a file or a directory
            if os.path.isfile(file_path):
                # If it's a file, add it to the zip file
                zip_file.write(file_path)
            elif os.path.isdir(file_path):
                # If it's a directory, add all its files and subdirectories to the zip file
                for root, dirs, files in os.walk(file_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zip_file.write(file_path)
