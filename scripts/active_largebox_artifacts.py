"""Load durable, losslessly compressed active-largebox model/checkpoint assets."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import shutil
import tempfile

BUNDLE = Path(__file__).resolve().parents[1] / 'logs/active_largebox/accepted_ground_to_ground_v1'

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def materialize(path):
    """Restore gzip bytes without changing the versioned bundle or prior evidence."""
    path = Path(path)
    packed = path.with_name(path.name + '.gz')
    if not packed.exists():
        if path.exists():
            return path
        raise FileNotFoundError(f'Missing model/checkpoint: {path} or {packed}')
    record = None
    for parent in packed.parents:
        manifest = parent / 'storage_manifest.json'
        if manifest.exists():
            record = json.loads(manifest.read_text())['compressed_files'].get(
                packed.relative_to(parent).as_posix())
            break
    if record is None:
        raise ValueError(f'No durable checksum metadata for {packed}')
    if sha256(packed) != record['gzip_sha256']:
        raise ValueError(f'Compressed artifact checksum mismatch: {packed}')
    cache = Path(os.environ.get('ORCS_ACTIVE_MODEL_CACHE',
                                str(Path(tempfile.gettempdir()) / 'orcs-active-largebox-assets')))
    cache.mkdir(parents=True, exist_ok=True)
    restored = cache / (record['raw_sha256'] + path.suffix)
    if restored.exists():
        if sha256(restored) != record['raw_sha256']:
            raise ValueError(f'Restored artifact checksum mismatch: {restored}')
        return restored
    fd, temporary = tempfile.mkstemp(dir=cache)
    try:
        with os.fdopen(fd, 'wb') as output, gzip.open(packed, 'rb') as source:
            shutil.copyfileobj(source, output)
        if sha256(temporary) != record['raw_sha256']:
            raise ValueError(f'Decompressed artifact checksum mismatch: {packed}')
        os.replace(temporary, restored)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return restored
