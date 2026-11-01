#!/usr/bin/env python3
"""Build a basic Debian package with only the Python standard library."""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path


def add_package_tree(archive: tarfile.TarFile, root: Path, relative: str) -> None:
    source = root / relative
    archive.add(source, arcname=relative, recursive=False, filter=normalize_tar_info)
    for path in sorted(source.rglob("*")):
        archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False, filter=normalize_tar_info)


def normalize_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    info.mtime = 0
    info.pax_headers = {}
    return info


def make_tar_gz(root: Path, members: list[str]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz", format=tarfile.GNU_FORMAT) as archive:
        for member in members:
            add_package_tree(archive, root, member)
    return stream.getvalue()


def ar_member(name: str, data: bytes) -> bytes:
    header = (
        f"{name:<16}{0:<12}{0:<6}{0:<6}{0o100644:<8o}{len(data):<10}`\n"
    ).encode("ascii")
    if len(header) != 60:
        raise ValueError(f"Invalid ar header length for {name}: {len(header)}")
    return header + data + (b"\n" if len(data) % 2 else b"")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: build_deb_portable.py PACKAGE_DIRECTORY OUTPUT.deb")
    package_root = Path(sys.argv[1]).resolve()
    output_path = Path(sys.argv[2]).resolve()
    control = make_tar_gz(package_root / "DEBIAN", ["control"])
    data = make_tar_gz(package_root, sorted(p.name for p in package_root.iterdir() if p.name != "DEBIAN"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(
        b"!<arch>\n"
        + ar_member("debian-binary", b"2.0\n")
        + ar_member("control.tar.gz", control)
        + ar_member("data.tar.gz", data)
    )


if __name__ == "__main__":
    main()
