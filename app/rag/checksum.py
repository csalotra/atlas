import hashlib


def calculate_checksum(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
