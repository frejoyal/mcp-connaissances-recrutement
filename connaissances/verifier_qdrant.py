"""Vérifie la recherche de CV dans Qdrant sur les 60 questions de rag-recrutement-bench.

Usage :
    python -m connaissances.verifier_qdrant              # Qdrant local (fichiers), écrit resultats/verification-qdrant.json
    QDRANT_URL=http://localhost:6333 python -m connaissances.verifier_qdrant --verifier   # serveur Qdrant (CI, Docker)

Trois contrôles :
1. Qdrant contre la recherche exacte : pour chaque question, les 10 sections les plus proches selon Qdrant sont-elles
   les mêmes, dans le même ordre, que celles d'un produit scalaire numpy sur les 442 sections ?
2. L'outil hybride (100 sections tirées de Qdrant, plus BM25) contre l'hybride calculé sur toutes les sections :
   même premier candidat, mêmes cinq premiers ?
3. La qualité de l'outil : bon candidat en premier, ou dans les trois premiers, sur les 45 questions qui ont une réponse.
   On doit retrouver le score de la méthode hybride sans reclassement de rag-recrutement-bench.
Le rapport ne dépend pas du mode de Qdrant : la CI vérifie que le mode local et un vrai serveur donnent le même.
"""
import argparse
import json
import sys

from .sources import K_RRF, SourceCV, jetons, rangs, scores_exacts
from .vecteurs import RACINE, Vecteurs

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
RAPPORT = RACINE / 'resultats' / 'verification-qdrant.json'


def hybride_exact(source, requete, nombre=5):
    """La fusion hybride de rag-recrutement-bench : vecteurs exacts et BM25 sur toutes les sections."""
    m = source.vecteurs(source.passages)
    vec = [round(float(s), 6) for s in m @ source.vecteurs([requete])[0]]
    bm = [round(float(s), 6) for s in source.bm25.get_scores(jetons(requete))]
    fusion = [1.0 / (K_RRF + 1 + a) + 1.0 / (K_RRF + 1 + b) for a, b in zip(rangs(vec), rangs(bm))]
    meilleurs = {}
    for i, s in enumerate(fusion):
        c = source.candidats[i]
        meilleurs[c] = max(meilleurs.get(c, 0.0), s)
    return [c for c, _ in sorted(meilleurs.items(), key=lambda x: (-x[1], x[0]))[:nombre]]


def verifier(source):
    questions = json.loads((RACINE / 'evaluation' / 'questions-cv.json').read_text(encoding='utf-8'))
    memes_10, meme_ordre, meme_premier, memes_cinq = 0, 0, 0, 0
    repondables, premier, trois = 0, 0, 0
    for q in questions:
        qdrant = [i for i, _ in source.vecteurs_qdrant(q['question'], 10)]
        exact = scores_exacts(source, q['question'], 10)
        memes_10 += set(qdrant) == set(exact)
        meme_ordre += qdrant == exact
        outil = [c['id'] for c in source.chercher(q['question'], 5)]
        reference = hybride_exact(source, q['question'], 5)
        meme_premier += outil[:1] == reference[:1]
        memes_cinq += outil == reference
        if q['attendus']:
            repondables += 1
            premier += outil[0] in q['attendus']
            trois += any(c in q['attendus'] for c in outil[:3])
    n = len(questions)
    return {
        'sections': len(source.passages),
        'questions': n,
        'qdrant_memes_10_sections_que_la_recherche_exacte': f'{memes_10}/{n}',
        'qdrant_meme_ordre_que_la_recherche_exacte': f'{meme_ordre}/{n}',
        'outil_meme_premier_candidat_que_l_hybride_complet': f'{meme_premier}/{n}',
        'outil_memes_cinq_candidats_que_l_hybride_complet': f'{memes_cinq}/{n}',
        'outil_bon_candidat_en_premier': f'{premier}/{repondables}',
        'outil_bon_candidat_dans_les_trois_premiers': f'{trois}/{repondables}',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verifier', action='store_true')
    args = parser.parse_args()
    vecteurs = Vecteurs()
    rapport = verifier(SourceCV(vecteurs))
    vecteurs.enregistrer()
    for k, v in rapport.items():
        print(f"{k.replace('_', ' ')} : {v}")
    if args.verifier:
        identique = rapport == json.loads(RAPPORT.read_text(encoding='utf-8'))
        print('Rapport identique à celui publié.' if identique else 'ERREUR : le rapport diffère de celui publié.')
        sys.exit(0 if identique else 1)
    RAPPORT.write_text(json.dumps(rapport, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
