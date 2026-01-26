import logging
import time
import json
import hashlib
import threading
from collections import OrderedDict
from typing import Dict, Any, Optional

try:
    import firebase_admin
    from firebase_admin import firestore
except ImportError:
    firebase_admin = None
    firestore = None

logger = logging.getLogger(__name__)

class SmartCache:
    """Cache inteligente con persistencia en Firestore (Stack Gratuito)."""
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache: OrderedDict = OrderedDict() # Cache en memoria L1
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()
        self.firestore_db = None
        self._init_firestore()

    def _init_firestore(self):
        """Inicializa conexión a Firestore."""
        try:
            if firebase_admin._apps:
                self.firestore_db = firestore.client()
                logger.info("✅ Firestore conectado para Cache L2")
            else:
                logger.warning("⚠️ Firebase no inicializado, usando solo memoria.")
        except Exception as e:
            logger.warning(f"⚠️ Error conectando Firestore: {e}")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Obtener valor del cache (Memoria -> Firestore)."""
        with self.lock:
            # 1. Buscar en memoria (L1)
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    self.hits += 1
                    self.cache.move_to_end(key)
                    return value
                else:
                    del self.cache[key]

            # 2. Buscar en Firestore (L2)
            if self.firestore_db:
                try:
                    doc_ref = self.firestore_db.collection('cache').document(hashlib.md5(key.encode()).hexdigest())
                    doc = doc_ref.get()
                    if doc.exists:
                        data = doc.to_dict()
                        timestamp = data.get('timestamp', 0)
                        
                        if time.time() - timestamp < self.ttl:
                            value = json.loads(data.get('value'))
                            # Restaurar a memoria
                            self.cache[key] = (value, timestamp)
                            self.hits += 1
                            return value
                except Exception as e:
                    logger.warning(f"⚠️ Error leyendo Firestore: {e}")

            self.misses += 1
            return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Guardar valor en cache (Memoria + Firestore)."""
        with self.lock:
            timestamp = time.time()

            # Guardar en memoria
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = (value, timestamp)

            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

            # Guardar en Firestore (Async)
            if self.firestore_db:
                threading.Thread(target=self._save_to_firestore, args=(key, value, timestamp), daemon=True).start()

    def _save_to_firestore(self, key: str, value: Dict[str, Any], timestamp: float):
        try:
            doc_ref = self.firestore_db.collection('cache').document(hashlib.md5(key.encode()).hexdigest())
            doc_ref.set({
                'key': key, # Guardar key original por si acaso
                'value': json.dumps(value, ensure_ascii=False),
                'timestamp': timestamp,
                'created_at': firestore.SERVER_TIMESTAMP
            })
        except Exception as e:
            logger.warning(f"⚠️ Error guardando en Firestore: {e}")

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            # No borramos Firestore completo por seguridad/costos
            
    def stats(self) -> Dict[str, Any]:
        return {
            "size_memory": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "backend": "Firestore" if self.firestore_db else "MemoryOnly"
        }
