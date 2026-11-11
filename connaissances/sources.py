"""Les trois sources de connaissances, et la recherche dans chacune.

- CV : 50 CV fictifs découpés en 442 sections. Les vecteurs sont dans Qdrant ; la recherche est hybride (vecteurs Qdrant
  et BM25, fusion par rangs réciproques), comme la meilleure méthode sans reclassement de rag-recrutement-bench.
- Règles : les 16 pages du wiki de llm-wiki-recrutement, cherchées par BM25 (la méthode la plus sobre de ce dépôt).
- Mémoire : les faits retenus pour chaque recruteur par la mémoire de faits de memoire-agent-recrutement.

Qdrant tourne en mode local (fichiers sur le disque, sans serveur) par défaut. Avec QDRANT_URL, le code se connecte à un
serveur Qdrant (par exemple lancé avec Docker) ; la CI vérifie que les deux modes donnent les mêmes résultats.
"""
import json
import os
import re
import unicodedata
import uuid

import numpy as np
from rank_bm25 import BM25Okapi

from .vecteurs import RACINE, Vecteurs

DONNEES = RACINE / 'donnees'
COLLECTION = 'cv_sections'
K_RRF = 60
CANDIDATS_VECTEURS = 100
UTILISATEURS = {'julie': 'Julie Ouellet', 'karim': 'Karim Benali', 'sophie': 'Sophie Gendreau'}
SPECIALITES = {
    'cloud': 'architecture infonuagique', 'data': 'ingénierie et analyse de données', 'cybersec': 'cybersécurité',
    'devweb': 'développement web', 'ia': 'intelligence artificielle et apprentissage automatique',
    'productmgmt': 'gestion de produit', 'devops': 'DevOps et fiabilité des plateformes', 'devmobile': 'développement mobile',
}
# Les mêmes mots vides que rag-recrutement-bench, pour que la recherche de CV donne les mêmes résultats.
MOTS_VIDES = set("""
a au aux avec ce ces cet cette d dans de des du elle en et eux il ils je j l la le les leur leurs lui m ma mais me meme mes moi mon
n ne nos notre nous on ou par pas pour qu que qui s sa se ses son sur t ta te tes toi ton tu un une vos votre vous y ete etre ai
as avait avons ont est sont qui quoi quel quelle quels quelles dont the of and for with in on at to an or by is are
""".split())


def jetons(texte):
    texte = ''.join(c for c in unicodedata.normalize('NFD', texte.lower()) if unicodedata.category(c) != 'Mn')
    return [m for m in re.findall(r'[a-z0-9]+(?:[-/][a-z0-9]+)*', texte) if m not in MOTS_VIDES and len(m) > 1]


def rangs(scores):
    ordre = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    r = [0] * len(scores)
    for position, i in enumerate(ordre):
        r[i] = position
    return r


class SourceCV:
    def __init__(self, vecteurs, client=None):
        self.vecteurs = vecteurs
        self.cvs = {cv['id']: cv for cv in json.loads((DONNEES / 'cvs.json').read_text(encoding='utf-8'))}
        self.passages, self.candidats = [], []
        for cv in self.cvs.values():
            for m in re.split(r'\n(?=## )', cv['cv_texte'].strip()):
                if len(m.split()) >= 5:
                    self.passages.append(m.strip())
                    self.candidats.append(cv['id'])
        self.bm25 = BM25Okapi([jetons(p) for p in self.passages])
        self.client = client or connecter_qdrant()
        self._indexer()

    def _indexer(self):
        from qdrant_client import models
        if self.client.collection_exists(COLLECTION) and self.client.count(COLLECTION).count == len(self.passages):
            return
        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)
        m = self.vecteurs(self.passages)
        self.client.create_collection(COLLECTION, vectors_config=models.VectorParams(size=m.shape[1], distance=models.Distance.COSINE))
        points = [models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, f'section-{i}')), vector=m[i].tolist(),
                                     payload={'numero': i, 'candidat': self.candidats[i]}) for i in range(len(self.passages))]
        for debut in range(0, len(points), 128):
            self.client.upsert(COLLECTION, points=points[debut:debut + 128], wait=True)

    def vecteurs_qdrant(self, requete, n=CANDIDATS_VECTEURS):
        """Les n sections les plus proches selon Qdrant : [(numéro de section, score)]."""
        q = self.vecteurs([requete])[0].tolist()
        reponse = self.client.query_points(COLLECTION, query=q, limit=n, with_payload=True)
        return [(p.payload['numero'], round(float(p.score), 6)) for p in reponse.points]

    def chercher(self, requete, nombre=5):
        """Classement hybride des candidats : rangs Qdrant et BM25 fusionnés, meilleur passage par candidat."""
        fusion = {}
        for rang, (i, _) in enumerate(self.vecteurs_qdrant(requete)):
            fusion[i] = fusion.get(i, 0.0) + 1.0 / (K_RRF + 1 + rang)
        bm25 = [round(float(s), 6) for s in self.bm25.get_scores(jetons(requete))]
        for i, r in enumerate(rangs(bm25)):
            fusion[i] = fusion.get(i, 0.0) + 1.0 / (K_RRF + 1 + r)
        meilleurs = {}
        for i, s in fusion.items():
            c = self.candidats[i]
            if c not in meilleurs or s > meilleurs[c][0]:
                meilleurs[c] = (s, i)
        classement = sorted(meilleurs.items(), key=lambda x: (-x[1][0], x[0]))[:nombre]
        return [{'id': c, 'nom': self.cvs[c]['nom'], 'specialite': SPECIALITES[self.cvs[c]['specialite']],
                 'ville': self.cvs[c]['localisation'], 'annees_experience': self.cvs[c]['annees_exp'],
                 'passage': self.passages[i]} for c, (s, i) in classement]


class SourceRegles:
    def __init__(self):
        self.pages = {}
        for f in sorted((DONNEES / 'wiki').glob('*.md')):
            if f.stem not in ('index', 'journal'):
                self.pages[f.stem] = f.read_text(encoding='utf-8')
        self.titres = list(self.pages)
        self.bm25 = BM25Okapi([jetons(t + ' ' + x) for t, x in self.pages.items()])

    def chercher(self, requete, nombre=3):
        scores = [round(float(s), 6) for s in self.bm25.get_scores(jetons(requete))]
        ordre = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [{'titre': self.titres[i], 'texte': re.sub(r'^---\n.*?\n---\n', '', self.pages[self.titres[i]], flags=re.S).strip()}
                for i in ordre[:nombre] if scores[i] > 0]


class SourceMemoire:
    def __init__(self, vecteurs):
        self.vecteurs = vecteurs
        self.faits = {u: x['faits'] for u, x in json.loads((DONNEES / 'memoire-de-faits.json').read_text(encoding='utf-8')).items()}

    def chercher(self, utilisateur, requete, nombre=5):
        faits = self.faits.get(utilisateur)
        if not faits:
            return None
        scores = self.vecteurs([f['texte'] for f in faits]) @ self.vecteurs([requete])[0]
        ordre = sorted(range(len(faits)), key=lambda i: (-round(float(scores[i]), 6), i))
        return [faits[i] for i in ordre[:nombre]]


def connecter_qdrant():
    from qdrant_client import QdrantClient
    url = os.environ.get('QDRANT_URL')
    if url:
        return QdrantClient(url=url)
    chemin = os.environ.get('QDRANT_CHEMIN', str(RACINE / 'index' / 'qdrant'))
    return QdrantClient(location=':memory:') if chemin == ':memory:' else QdrantClient(path=chemin)


def scores_exacts(source, requete, n):
    """Recherche exacte (produit scalaire numpy sur toutes les sections), pour vérifier Qdrant."""
    m = source.vecteurs(source.passages)
    s = m @ source.vecteurs([requete])[0]
    ordre = np.lexsort((np.arange(len(s)), -np.round(s, 6)))
    return [int(i) for i in ordre[:n]]
