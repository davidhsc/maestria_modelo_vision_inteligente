"""
exportar_certificados_windows.py  —  Certificados de confianza de Windows -> archivo PEM
=========================================================================================
En redes corporativas, un proxy o antivirus suele "inspeccionar" el tráfico HTTPS con un
certificado propio. Windows confía en él, pero un contenedor Linux NO, y `pip install`
falla con "CERTIFICATE_VERIFY_FAILED ... unable to get local issuer certificate".

Este script exporta TODOS los certificados raíz en los que confía Windows (incluido el de la
empresa) a certs/ca-windows.pem, para pasarlo al build de Docker como secreto:

    python herramientas/exportar_certificados_windows.py
    docker build --secret id=ca_corporativa,src=certs/ca-windows.pem -f Dockerfile.edge -t vision-figuras:1.0-edge .

Solo funciona en Windows (usa el almacén de certificados del sistema).
"""

import ssl
import sys
from pathlib import Path

if not hasattr(ssl, "enum_certificates"):
    sys.exit("Este script solo funciona en Windows.")

certificados = set()
for almacen in ("ROOT", "CA"):                       # raíces e intermedias de confianza
    for datos, codificacion, _ in ssl.enum_certificates(almacen):
        if codificacion == "x509_asn":
            certificados.add(ssl.DER_cert_to_PEM_cert(datos))

destino = Path(__file__).resolve().parents[1] / "certs" / "ca-windows.pem"
destino.parent.mkdir(exist_ok=True)
destino.write_text("".join(sorted(certificados)), encoding="ascii")
print(f"{len(certificados)} certificados exportados a {destino}")
