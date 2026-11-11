"""Le serveur MCP de connaissances de Recrutement Élite inc.

Trois outils, un par source : chercher_candidats, consulter_regles, rappeler_preferences (descriptions de la
version 2 par défaut ; CONNAISSANCES_OUTILS=trois pour celles de la version 1).
Avec CONNAISSANCES_OUTILS=unique, le serveur expose plutôt un seul outil, chercher_partout, qui interroge les trois
sources à la fois. L'évaluation compare les deux interfaces.

Lancement (transport stdio, celui de Claude Desktop et de Claude Code) :
    python -m connaissances.serveur
"""
import os
import sys

from mcp.server.mcpserver import MCPServer

from .sources import UTILISATEURS, SourceCV, SourceMemoire, SourceRegles
from .vecteurs import Vecteurs

INSTRUCTIONS = (
    "Connaissances internes de Recrutement Élite inc., un cabinet de recrutement en TI (entreprise fictive). "
    "Les CV des candidats, les règles et les clients du cabinet, et les préférences de chaque recruteur."
)
vecteurs = Vecteurs()
_cv = _regles = _memoire = None


def cv():
    global _cv
    if _cv is None:
        _cv = SourceCV(vecteurs)
    return _cv


def regles():
    global _regles
    if _regles is None:
        _regles = SourceRegles()
    return _regles


def memoire():
    global _memoire
    if _memoire is None:
        _memoire = SourceMemoire(vecteurs)
    return _memoire


def texte_candidats(resultats):
    lignes = []
    for i, c in enumerate(resultats, 1):
        extrait = ' '.join(c['passage'].split())[:400]
        lignes.append(f"{i}. {c['id']} · {c['nom']} · {c['specialite']} · {c['ville']} · {c['annees_experience']} ans\n   Passage : {extrait}")
    return '\n'.join(lignes) or 'Aucun candidat trouvé.'


def texte_regles(pages):
    return '\n\n'.join(f"## [[{p['titre']}]]\n{p['texte']}" for p in pages) or 'Aucune page trouvée.'


def texte_memoire(utilisateur, faits):
    if faits is None:
        return f"Utilisateur inconnu : {utilisateur}. Utilisateurs connus : {', '.join(UTILISATEURS)}."
    lignes = []
    for f in faits:
        avant = ''.join(f" (remplace : « {h} », noté le {d})" for d, h in f['historique'])
        lignes.append(f"- [{f['date']}] {f['texte']}{avant}")
    return '\n'.join(lignes) or 'Aucun souvenir.'


def chercher_candidats(requete: str, nombre: int = 5) -> str:
    """Cherche des candidats dans la banque de 50 CV : compétences, certifications, projets, ville, expérience.
    À utiliser pour trouver ou comparer des candidats. Retourne les candidats classés, avec un passage de leur CV."""
    resultat = texte_candidats(cv().chercher(requete, max(1, min(nombre, 10))))
    vecteurs.enregistrer()
    return resultat


def consulter_regles(requete: str) -> str:
    """Cherche dans le wiki des règles internes du cabinet : politique de recrutement, grille d'évaluation, Loi 25,
    conservation des dossiers, références, entrevues, utilisation de l'IA, et les fiches des clients (exigences,
    garanties). Retourne les 3 pages les plus pertinentes."""
    return texte_regles(regles().chercher(requete))


def rappeler_preferences(utilisateur: str, sujet: str) -> str:
    """Retrouve ce que l'assistant sait d'un recruteur : ses préférences, ses habitudes de travail, ses contacts et
    ses événements récents. utilisateur : julie, karim ou sophie (la personne qui parle). sujet : ce qu'on cherche."""
    resultat = texte_memoire(utilisateur, memoire().chercher(utilisateur.strip().lower(), sujet))
    vecteurs.enregistrer()
    return resultat


def chercher_partout(requete: str, utilisateur: str) -> str:
    """Cherche dans toutes les connaissances du cabinet à la fois : les CV des candidats, les règles et les clients,
    et les préférences du recruteur qui pose la question. utilisateur : julie, karim ou sophie."""
    parties = [
        '# Candidats\n' + texte_candidats(cv().chercher(requete, 3)),
        '# Règles et clients\n' + texte_regles(regles().chercher(requete, 2)),
        '# Préférences du recruteur\n' + texte_memoire(utilisateur, memoire().chercher(utilisateur.strip().lower(), requete, 3)),
    ]
    vecteurs.enregistrer()
    return '\n\n'.join(parties)


# Version 2 des descriptions, écrite après la première mesure : avec la version 1 (les docstrings ci-dessus), l'agent
# envoyait presque toutes les questions sur les préférences d'un recruteur vers consulter_regles.
DESCRIPTIONS_V2 = {
    'chercher_candidats': (
        "Cherche des CANDIDATS dans la banque de 50 CV. À utiliser seulement pour trouver, comparer ou décrire des "
        "candidats : compétences, certifications, projets, ville, expérience."),
    'consulter_regles': (
        "Cherche dans les RÈGLES OFFICIELLES du cabinet et les FICHES DES CLIENTS : politique de recrutement, grille "
        "d'évaluation, Loi 25, conservation des dossiers, références, entrevues, utilisation de l'IA, exigences et garanties "
        "des clients. Ne contient rien sur les préférences personnelles d'un recruteur."),
    'rappeler_preferences': (
        "Retrouve la MÉMOIRE PERSONNELLE du recruteur qui te parle : ce qu'il ou elle t'a dit dans vos conversations "
        "passées (ses préférences, sa façon de travailler, ses short-lists, ses rapports, ses contacts, ses objectifs, ses "
        "placements, ses vacances, ses événements). À utiliser dès que la question parle de « je », « mes », « mon » ou « ma ». "
        "utilisateur : l'identifiant de la personne qui parle (julie, karim ou sophie). sujet : ce qu'on cherche."),
}


def creer_serveur(interface=None):
    """interface : « trois » (descriptions de la version 1), « trois-v2 » ou « unique »."""
    interface = interface or os.environ.get('CONNAISSANCES_OUTILS', 'trois-v2')
    serveur = MCPServer(name='connaissances-recrutement', instructions=INSTRUCTIONS)
    if interface == 'unique':
        serveur.add_tool(chercher_partout)
        return serveur
    for f in (chercher_candidats, consulter_regles, rappeler_preferences):
        serveur.add_tool(f, description=DESCRIPTIONS_V2[f.__name__] if interface == 'trois-v2' else None)
    return serveur


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
    creer_serveur().run('stdio')
