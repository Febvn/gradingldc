"""
Sertifikat TLS self-signed untuk mode HTTPS.

Diperlukan karena browser HANYA mengizinkan akses kamera (getUserMedia) pada
secure context (HTTPS) atau localhost. Agar halaman Live dapat membuka kamera
dari HP melalui jaringan lokal (mis. https://192.168.x.x:5000), server harus
melayani HTTPS. Sertifikat dibuat memakai pustaka `cryptography` (tanpa pyOpenSSL).

Pengguna akan melihat peringatan "Not secure" di browser (wajar untuk sertifikat
self-signed) — cukup pilih "Lanjutkan / Advanced → Proceed".
"""
from __future__ import annotations

import os
import socket
import datetime
import ipaddress


def get_lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def ensure_cert(cert_dir: str) -> tuple[str, str]:
    """Pastikan ada (cert.pem, key.pem); buat self-signed bila belum ada."""
    os.makedirs(cert_dir, exist_ok=True)
    cert_path = os.path.join(cert_dir, "cert.pem")
    key_path = os.path.join(cert_dir, "key.pem")
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return cert_path, key_path

    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    lan_ip = get_lan_ip()
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Smart Grading Kopi PoC"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ITERA x PT LDC"),
    ])
    san = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
    ]
    try:
        san.append(x509.IPAddress(ipaddress.ip_address(lan_ip)))
    except ValueError:
        pass

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    return cert_path, key_path
