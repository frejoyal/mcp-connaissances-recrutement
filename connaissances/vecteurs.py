"""Vecteurs bge-m3 (Ollama), avec un cache publié dans le dépôt.

Les vecteurs des CV et des faits de mémoire viennent des dépôts rag-recrutement-bench et memoire-agent-recrutement.
Une nouvelle requête est vectorisée par Ollama, sauf en mode « cache » (CONNAISSANCES_MODE=cache), où tout doit déjà
être dans le cache : c'est ce que fait la CI, sans modèle ni réseau.
"""
import hashlib
import os
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parent.parent
CACHE = RACINE / 'resultats' / 'cache' / 'embeddings.npz'
OLLAMA = os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11434')
MODELE = 'bge-m3'


def cle(texte):
    return hashlib.sha1(f'{MODELE}\x1f{texte}'.encode('utf-8')).hexdigest()


class Vecteurs:
    def __init__(self, mode=None):
        self.mode = mode or os.environ.get('CONNAISSANCES_MODE', 'direct')
        d = np.load(CACHE)
        self._v = dict(zip(d['cles'].tolist(), d['vecteurs']))
        self.modifie = False

    def __call__(self, textes):
        cles = [cle(t) for t in textes]
        manquants = [(k, t) for k, t in zip(cles, textes) if k not in self._v]
        if manquants:
            if self.mode == 'cache':
                raise RuntimeError('Vecteur absent du cache : relancez avec Ollama démarré (CONNAISSANCES_MODE=direct).')
            import requests
            r = requests.post(f'{OLLAMA}/api/embed', json={'model': MODELE, 'input': [t for _, t in manquants]}, timeout=300)
            r.raise_for_status()
            for (k, _), v in zip(manquants, r.json()['embeddings']):
                self._v[k] = np.asarray(v, dtype=np.float16)
            self.modifie = True
        m = np.stack([self._v[k].astype(np.float32) for k in cles])
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    def enregistrer(self):
        if self.modifie:
            cles = sorted(self._v)
            np.savez_compressed(CACHE, cles=np.array(cles), vecteurs=np.stack([self._v[k] for k in cles]))
            self.modifie = False
