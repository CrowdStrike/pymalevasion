#!/usr/bin/env python3

"""
Extraction script for archived samples e.g. for
- https://github.com/lxyeternal/pypi_malregistry/tree/main

Extract sources (of given extension) from *.tar.gz archives, renaming to their sha256.
The actual names are not important, only the contents.

Raw dataset structure is:
    package-manager/package-name/version/package.file

Usage:
    ./pkg_extract.py --input-dir pypi_malregistry --output-dir out --output-json out.json --extension py
"""

import hashlib
import itertools
import json
import tarfile
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path


def sha256(x: bytes) -> str:
    return hashlib.sha256(x).hexdigest()


def read_archive(archive: Path, extension: str) -> list[bytes]:
    """
    Read files from `archive` with `extension` and return their contents as bytes.
    """
    formats = ["r:gz", "r:xz"]
    out = []

    for fmt in formats:
        try:
            with tarfile.open(archive, fmt) as tar:
                # read into bytes
                scripts = (f for f in tar.getmembers() if f.name.endswith(f".{extension}"))
                for script in scripts:
                    data = tar.extractfile(script)
                    if data is None:
                        continue
                    data = data.read()
                    if len(data) > 0:
                        out.append(data)

            break  # we've read the archive successfully with this format

        except tarfile.ReadError:
            continue  # try the next format
    else:
        raise tarfile.ReadError(f"Could not read {archive} as any of {formats}")

    return out


def main(args):
    # collect all archives
    archives = itertools.chain(args.input_dir.glob("**/*.tgz"), args.input_dir.glob("**/*.tar.gz"))

    # make output dir if it doesn't exist
    args.output_dir.mkdir(exist_ok=False)

    pkg2files = defaultdict(list)

    for i, archive in enumerate(archives, start=1):
        assert archive.name.endswith(".tar.gz")
        pkg = archive.name.removesuffix(".tar.gz")
        for data in read_archive(archive, extension=args.extension):
            # write to output dir as <sha256>.<extension>
            filename = f"{sha256(data)}.{args.extension}"
            out_path = args.output_dir / filename

            pkg2files[pkg].append(filename)

            with open(out_path, "wb") as fp:
                fp.write(data)

            print(f"[{i}] {archive} to {out_path}.")

    with open(args.output_json, "wt") as fp:
        json.dump(pkg2files, fp, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, help="Input directory for raw data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for extracted data",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path to dump the JSON mapping pkg name to corresponding scripts",
    )
    parser.add_argument(
        "--extension",
        type=str,
        choices=["py", "js", "ts"],
        help="File extension to extract",
    )
    args = parser.parse_args()

    main(args)
