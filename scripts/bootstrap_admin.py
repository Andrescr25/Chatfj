#!/usr/bin/env python3
"""
Crea o promueve a la primera persona administradora de Chat FJ.

Es el único paso manual del sistema de roles: a partir de aquí, esa persona
puede dar acceso a las demás desde el panel web.

Uso:
    python scripts/bootstrap_admin.py --email persona@poder-judicial.go.cr
    python scripts/bootstrap_admin.py --email persona@x.cr --name "Nombre" --password "clave-larga"
    python scripts/bootstrap_admin.py --list
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import firebase_admin
from firebase_admin import auth, credentials

from src.app.config import settings

ROLE_ADMIN = "admin"


def init_firebase():
    if firebase_admin._apps:
        return
    cred_path = settings.FIREBASE_CREDENTIALS_PATH
    if os.path.exists(cred_path):
        firebase_admin.initialize_app(credentials.Certificate(cred_path))
        print(f"✅ Firebase inicializado con {cred_path}")
    else:
        firebase_admin.initialize_app()
        print("✅ Firebase inicializado con credenciales por defecto de Google Cloud")


def listar():
    print("\nCuentas con rol administrativo:\n")
    encontrados = 0
    for user in auth.list_users().iterate_all():
        claims = user.custom_claims or {}
        email = (user.email or "").lower()
        role = claims.get("role")
        if role == "superadmin":  # rol de la versión anterior, hoy equivale a admin
            role = "admin (rol antiguo)"
        if email in settings.admin_emails:
            role = "admin (por ADMIN_EMAILS)"
        if not role:
            continue
        encontrados += 1
        estado = "deshabilitada" if user.disabled else "activa"
        print(f"  • {email or user.uid}  —  {role}  —  {estado}")
    if not encontrados:
        print("  (ninguna)")
    print()


def asignar(email: str, name: str, password: str):
    email = email.strip().lower()
    try:
        user = auth.get_user_by_email(email)
        print(f"👤 Cuenta existente: {email}")
        if name and not user.display_name:
            auth.update_user(user.uid, display_name=name)
        if password:
            auth.update_user(user.uid, password=password)
            print("🔑 Contraseña actualizada.")
    except auth.UserNotFoundError:
        user = auth.create_user(
            email=email,
            display_name=name or None,
            password=password or None,
            email_verified=False,
        )
        print(f"✅ Cuenta creada: {email}")

    auth.set_custom_user_claims(user.uid, {"role": ROLE_ADMIN})
    print(f"✅ Acceso de administración asignado a {email} (uid: {user.uid})")

    if not password:
        try:
            link = auth.generate_password_reset_link(email)
            print("\n🔗 Enlace para definir la contraseña (envíeselo a la persona):")
            print(f"   {link}\n")
        except Exception as e:
            print(f"⚠️ No se pudo generar el enlace de contraseña: {e}")

    print("ℹ️ Si la persona ya tenía sesión abierta, debe cerrarla y volver a entrar")
    print("   para que el nuevo rol aparezca en su token.")


def main():
    parser = argparse.ArgumentParser(description="Gestión inicial de administradores de Chat FJ")
    parser.add_argument("--email", help="Correo de la persona")
    parser.add_argument("--name", default="", help="Nombre para mostrar")
    parser.add_argument("--password", default="", help="Contraseña inicial (mínimo 8 caracteres)")
    parser.add_argument("--list", action="store_true", help="Lista las cuentas con rol")
    args = parser.parse_args()

    if args.password and len(args.password) < 8:
        print("❌ La contraseña debe tener al menos 8 caracteres.")
        sys.exit(1)

    init_firebase()

    if args.list:
        listar()
        return

    if not args.email:
        parser.error("Indique --email o use --list")

    asignar(args.email, args.name, args.password)


if __name__ == "__main__":
    main()
