"""
Tests de autenticación y acceso al panel de administración.

Hay un único rol: 'admin'.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import HTTPException

from src.app.core.security import ROLE_ADMIN, get_current_user, require_admin


class TestAutenticacion(unittest.TestCase):
    def test_sin_encabezado_rechaza(self):
        with self.assertRaises(HTTPException) as ctx:
            get_current_user(None)
        self.assertEqual(ctx.exception.status_code, 401)

    def test_encabezado_mal_formado_rechaza(self):
        with self.assertRaises(HTTPException) as ctx:
            get_current_user("token-suelto-sin-bearer")
        self.assertEqual(ctx.exception.status_code, 401)

    def test_token_invalido_rechaza(self):
        with patch("firebase_admin.auth.verify_id_token", side_effect=ValueError("token falso")), \
             self.assertRaises(HTTPException) as ctx:
            get_current_user("Bearer token-invalido")
        self.assertEqual(ctx.exception.status_code, 401)

    def test_cuenta_sin_rol_no_es_administradora(self):
        """Una cuenta legítima de Firebase sin rol asignado no puede administrar."""
        decoded = {"uid": "u1", "email": "cualquiera@ejemplo.cr"}
        with patch("firebase_admin.auth.verify_id_token", return_value=decoded), \
             self.assertRaises(HTTPException) as ctx:
            get_current_user("Bearer token-valido")
        self.assertEqual(ctx.exception.status_code, 403)

    def test_rol_admin_es_aceptado(self):
        decoded = {"uid": "u2", "email": "admin@ejemplo.cr", "role": ROLE_ADMIN, "name": "Ana"}
        with patch("firebase_admin.auth.verify_id_token", return_value=decoded):
            user = get_current_user("Bearer token-valido")
        self.assertEqual(user.role, ROLE_ADMIN)
        self.assertEqual(user.display_name, "Ana")

    def test_rol_antiguo_superadmin_sigue_funcionando(self):
        """Nadie queda fuera por tener el claim de la versión de dos roles."""
        decoded = {"uid": "u3", "email": "antiguo@ejemplo.cr", "role": "superadmin"}
        with patch("firebase_admin.auth.verify_id_token", return_value=decoded):
            user = get_current_user("Bearer token-valido")
        self.assertEqual(user.role, ROLE_ADMIN)

    def test_allowlist_por_correo_da_acceso(self):
        decoded = {"uid": "u4", "email": "jefatura@ejemplo.cr"}
        with patch("firebase_admin.auth.verify_id_token", return_value=decoded), \
             patch("src.app.config.settings.ADMIN_EMAILS", "jefatura@ejemplo.cr"):
            user = get_current_user("Bearer token-valido")
        self.assertEqual(user.role, ROLE_ADMIN)

    def test_allowlist_acepta_el_nombre_anterior_de_la_variable(self):
        decoded = {"uid": "u5", "email": "jefatura@ejemplo.cr"}
        with patch("firebase_admin.auth.verify_id_token", return_value=decoded), \
             patch("src.app.config.settings.ADMIN_EMAILS", ""), \
             patch("src.app.config.settings.SUPERADMIN_EMAILS", "jefatura@ejemplo.cr"):
            user = get_current_user("Bearer token-valido")
        self.assertEqual(user.role, ROLE_ADMIN)


class TestAutorizacion(unittest.TestCase):
    def test_administrador_pasa_la_puerta(self):
        decoded = {"uid": "u2", "email": "admin@ejemplo.cr", "role": ROLE_ADMIN}
        with patch("firebase_admin.auth.verify_id_token", return_value=decoded):
            user = get_current_user("Bearer token-valido")
        self.assertEqual(require_admin(user).uid, "u2")


if __name__ == "__main__":
    unittest.main()
