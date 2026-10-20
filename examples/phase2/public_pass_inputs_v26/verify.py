#!/usr/bin/env python3
"""Check portable public fixture bytes/arrays; never runs a compiler or issues a seal."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import struct
import zipfile

MAX_BYTES = 4_500_000


def relative(value: str) -> str:
    path = PurePosixPath(value)
    if (not isinstance(value, str) or not value or path.is_absolute()
            or ".." in path.parts or "\\" in value or path.as_posix() != value):
        raise ValueError("noncanonical relative bundle path")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def read_json(raw: bytes):
    return json.loads(raw, object_pairs_hook=unique_object)


def array(raw: bytes):
    if raw[:6] != b"\x93NUMPY" or tuple(raw[6:8]) not in ((1, 0), (2, 0), (3, 0)):
        raise ValueError("invalid NPY header")
    width = 2 if raw[6] == 1 else 4
    header_size = int.from_bytes(raw[8:8 + width], "little")
    if header_size > 4096:
        raise ValueError("oversized NPY header")
    end = 8 + width + header_size
    header = ast.literal_eval(raw[8 + width:end].decode("utf-8"))
    if set(header) != {"descr", "fortran_order", "shape"} or header["fortran_order"] is not False:
        raise ValueError("unsupported NPY layout")
    shape = header["shape"]
    if not isinstance(shape, tuple) or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("invalid NPY shape")
    formats = {"|i1": (1, "b"), "<i4": (4, "i")}
    if header["descr"] not in formats:
        raise ValueError("only exact signed-i8/i32 fixture arrays are allowed")
    item_size, code = formats[header["descr"]]
    size = 1
    for n in shape:
        size *= n
    payload = raw[end:]
    if size > MAX_BYTES or len(payload) != size * item_size:
        raise ValueError("incomplete or oversized NPY payload")
    return header, payload, code


def flat(value):
    if isinstance(value, list):
        return [item for child in value for item in flat(child)]
    if type(value) is not int:
        raise ValueError("non-integer source golden")
    return [value]


def main():
    root = Path(__file__).resolve().parent
    rows = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        name = relative(name)
        if name in rows or len(digest) != 64:
            raise ValueError("invalid checksum roster")
        rows[name] = digest
    actual = set()
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlink in portable bundle")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            actual.add(name)
            total += path.stat().st_size
            if total > MAX_BYTES:
                raise ValueError("bundle size budget exceeded")
            if name != "SHA256SUMS" and hashlib.sha256(path.read_bytes()).hexdigest() != rows.get(name):
                raise ValueError("bundle byte/hash mismatch: " + name)
    if actual != set(rows) | {"SHA256SUMS"} or len(actual) > 64:
        raise ValueError("extra or missing bundle leaves")
    manifest = read_json((root / "manifest.json").read_bytes())
    if manifest["schema"] != "gemmini.portable_public_pass_inputs.v26.v1":
        raise ValueError("unsupported fixture manifest")
    for name, row in manifest["files"].items():
        raw = (root / relative(name)).read_bytes()
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("payload manifest mismatch")
    if len(manifest["files"]) != 43:
        raise ValueError("incomplete fixture payload")
    package = hashlib.sha256()
    compiler_names = sorted(name for name in manifest["files"] if name.startswith("compiler/"))
    for name in compiler_names:
        package.update(name.removeprefix("compiler/").encode())
        package.update(b"\0")
        package.update((root / name).read_bytes())
        package.update(b"\0")
    if len(compiler_names) != 16 or package.hexdigest() != manifest["compiler"]["sha256"]:
        raise ValueError("original complete compiler package differs")
    outputs = values = parameter_bytes = 0
    types = {"i8": "|i1", "i32": "<i4"}
    for selection in manifest["cases"]:
        descriptor = root / relative(selection["path"])
        case = read_json(descriptor.read_bytes())
        base = descriptor.parent
        expected_names = [row["name"] for row in case["inputs"] + case["outputs"]]
        if case["arg_order"] != expected_names:
            raise ValueError("source argument/output order differs")
        with zipfile.ZipFile(base / relative(case["reference"])) as archive:
            infos = archive.infolist()
            names = [i.filename for i in infos]
            expected = [r["reference_member"] for r in case["inputs"] + case["outputs"]]
            if len(names) != len(set(names)) or set(names) != set(expected):
                raise ValueError("reference omits or adds arrays")
            if sum(i.file_size for i in infos) > MAX_BYTES:
                raise ValueError("reference decompression budget exceeded")
            decoded = {}
            for row in case["inputs"] + case["outputs"]:
                raw = archive.read(row["reference_member"])
                if hashlib.sha256(raw).hexdigest() != row["npy_sha256"]:
                    raise ValueError("complete reference array changed")
                header, payload, code = array(raw)
                if list(header["shape"]) != row["shape"] or header["descr"] != types[row["dtype"]]:
                    raise ValueError("reference/source shape or dtype differs")
                decoded[row["reference_member"]] = [v[0] for v in struct.iter_unpack("<" + code, payload)]
            outputs += len(case["outputs"])
            values += sum(len(decoded[row["reference_member"]]) for row in case["outputs"])
            golden_path = base / "source-capture-golden.json"
            if golden_path.exists():
                golden = read_json(golden_path.read_bytes())
                # Original capture transport unwraps a single source output.
                if len(case["outputs"]) == 1:
                    golden = [golden]
                if len(golden) != len(case["outputs"]):
                    raise ValueError("source golden output arity differs")
                for row, item in zip(case["outputs"], golden, strict=True):
                    if flat(item) != decoded[row["reference_member"]]:
                        raise ValueError("captured output differs from complete independent reference")
        weights = (base / relative(case["weights"])).read_bytes()
        length = int.from_bytes(weights[:8], "little")
        header = read_json(weights[8:8 + length])
        payload = weights[8 + length:]
        spans = []
        for name, row in header.items():
            if name == "__metadata__":
                continue
            count = 1
            for n in row["shape"]:
                count *= n
            begin, end = row["data_offsets"]
            if row["dtype"] != "I8" or end - begin != count:
                raise ValueError("weight dtype/extent differs")
            spans.append((begin, end))
        at = 0
        for begin, end in sorted(spans):
            if begin != at or end < begin:
                raise ValueError("missing/overlapping weight payload")
            at = end
        if at != len(payload):
            raise ValueError("unaccounted weight bytes")
        parameter_bytes += at
    print(json.dumps({"status": "portable_fixture_integrity_verified", "payload_files": 43,
                      "compiler_files": 16, "cases": 3, "output_tensors": outputs,
                      "output_values": values, "parameter_bytes": parameter_bytes,
                      "V26_execution": "NOT_RUN", "admission": "NOT_ISSUED"}, sort_keys=True))


if __name__ == "__main__":
    main()
