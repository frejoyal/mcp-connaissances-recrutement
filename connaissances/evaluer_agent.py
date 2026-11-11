"""Un agent local connecté au serveur MCP, mesuré sur 34 questions, avec deux interfaces.

Usage :
    python -m connaissances.evaluer_agent --mode direct   # gemma4:e4b répond (Ollama) et le cache se remplit
    python -m connaissances.evaluer_agent                 # rejoue à partir du cache, sans modèle
    python -m connaissances.evaluer_agent --verifier      # s'arrête en erreur si les résultats diffèrent (utilisé par la CI)

L'agent est un vrai client MCP : il lance le serveur (transport stdio), lit la liste de ses outils, et chaque appel d'outil
passe par le protocole. Deux interfaces :
- trois outils : chercher_candidats, consulter_regles, rappeler_preferences ; l'agent choisit. Deux versions des
  descriptions : la v1, puis la v2 écrite après avoir vu l'agent confondre la mémoire et les règles ;
- outil unique : chercher_partout, qui interroge les trois sources à la fois.

Si l'agent veut répondre sans avoir appelé d'outil, on lui demande une fois d'utiliser ses outils (la « relance »).
Une réponse est bonne si elle contient, pour chaque groupe de « attendu », une des formes acceptées (minuscules,
sans accents, mot entier).
"""
import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import unicodedata

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .sources import UTILISATEURS
from .vecteurs import RACINE

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
RESULTATS = RACINE / 'resultats'
CACHE = RESULTATS / 'cache' / 'agent.json'
OLLAMA = os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11434')
MODELE = 'gemma4:e4b'
OPTIONS = {'temperature': 0, 'seed': 7, 'num_ctx': 16384, 'num_predict': 800}
ETAPES_MAX = 6
INTERFACES = [('trois', 'Trois outils, descriptions v1'), ('trois-v2', 'Trois outils, descriptions v2'),
              ('unique', 'Un outil unique qui cherche partout')]


def normaliser(texte):
    texte = ''.join(c for c in unicodedata.normalize('NFD', texte.lower()) if unicodedata.category(c) != 'Mn')
    return re.sub(r'\s+', ' ', texte.replace('’', "'").replace(' ', ' ')).strip()


def contient(texte, forme):
    avant = r'(?<![\w-])' if forme[0].isdigit() else r'(?<!\w)'
    return re.search(avant + re.escape(forme) + r'(?!\w)', texte) is not None


def reussie(q, reponse):
    texte = normaliser(reponse)
    return all(any(contient(texte, normaliser(f)) for f in groupe) for groupe in q['attendu'])


class Modele:
    def __init__(self, mode):
        self.mode = mode
        self._r = json.loads(CACHE.read_text(encoding='utf-8')) if CACHE.exists() else {}
        self.modifie = False

    def discuter(self, messages, outils):
        requete = {'model': MODELE, 'messages': messages, 'tools': outils, 'options': OPTIONS}
        k = hashlib.sha1(json.dumps(requete, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
        if k not in self._r:
            if self.mode == 'cache':
                raise RuntimeError('Réponse absente du cache : relancez en mode direct, avec Ollama démarré.')
            import requests
            r = requests.post(f'{OLLAMA}/api/chat', json={**requete, 'stream': False, 'think': False}, timeout=900)
            r.raise_for_status()
            m = r.json()['message']
            appels = [{'function': {'name': a['function']['name'], 'arguments': a['function'].get('arguments') or {}}}
                      for a in m.get('tool_calls') or []]
            self._r[k] = {'content': (m.get('content') or '').strip(), 'tool_calls': appels}
            self.modifie = True
        return self._r[k]

    def enregistrer(self):
        if self.modifie:
            CACHE.write_text(json.dumps(dict(sorted(self._r.items())), ensure_ascii=False, indent=0) + '\n', encoding='utf-8')
            self.modifie = False


def systeme(utilisateur):
    return (f"Tu es l'assistant de {UTILISATEURS[utilisateur]} (identifiant : {utilisateur}) chez Recrutement Élite inc., "
            "un cabinet de recrutement en TI. Nous sommes le 30 juin 2026. Tu ne connais rien du cabinet par toi-même : "
            "utilise tes outils pour chercher l'information, puis réponds en français, en deux ou trois phrases, avec les noms "
            "et les chiffres trouvés. Si les outils ne donnent pas la réponse, dis-le.")


async def repondre(session, outils, modele, q):
    messages = [{'role': 'system', 'content': systeme(q['utilisateur'])}, {'role': 'user', 'content': q['question']}]
    appels, mots, relance = [], 0, False
    for _ in range(ETAPES_MAX):
        mots += sum(len(str(m.get('content', '')).split()) for m in messages)
        r = modele.discuter(messages, outils)
        if not r['tool_calls']:
            if not appels and not relance:
                relance = True
                messages += [{'role': 'assistant', 'content': r['content']},
                             {'role': 'user', 'content': "Utilise tes outils pour chercher l'information avant de répondre."}]
                continue
            return r['content'], appels, mots, relance
        messages.append({'role': 'assistant', 'content': r['content'], 'tool_calls': r['tool_calls']})
        for a in r['tool_calls']:
            nom, args = a['function']['name'], a['function']['arguments']
            appels.append(nom)
            resultat = await session.call_tool(nom, args if isinstance(args, dict) else {})
            texte = '\n'.join(c.text for c in resultat.content if getattr(c, 'text', None)) or '(vide)'
            messages.append({'role': 'tool', 'tool_name': nom, 'content': texte})
    messages.append({'role': 'user', 'content': 'Réponds maintenant à la question avec ce que tu as trouvé.'})
    mots += sum(len(str(m.get('content', '')).split()) for m in messages)
    return modele.discuter(messages, outils)['content'], appels, mots, relance


async def evaluer_interface(interface, questions, modele, mode):
    env = {**os.environ, 'CONNAISSANCES_OUTILS': interface, 'CONNAISSANCES_MODE': mode, 'QDRANT_CHEMIN': ':memory:',
           'PYTHONIOENCODING': 'utf-8'}
    env.pop('QDRANT_URL', None)
    parametres = StdioServerParameters(command=sys.executable, args=['-m', 'connaissances.serveur'], cwd=str(RACINE), env=env)
    async with stdio_client(parametres) as (lecture, ecriture):
        async with ClientSession(lecture, ecriture) as session:
            await session.initialize()
            liste = await session.list_tools()
            outils = [{'type': 'function', 'function': {'name': t.name, 'description': t.description, 'parameters': t.input_schema}}
                      for t in liste.tools]
            details = {}
            for q in questions:
                reponse, appels, mots, relance = await repondre(session, outils, modele, q)
                details[q['id']] = {'reponse': reponse, 'outils_appeles': appels, 'reussie': reussie(q, reponse),
                                    'relance': relance, 'mots_envoyes': mots}
                modele.enregistrer()
            return [t['function']['name'] for t in outils], details


def mesures(nom, outils, questions, details):
    n = len(questions)
    par_source = {}
    for s in dict.fromkeys(q['source'] for q in questions):
        l = [q for q in questions if q['source'] == s]
        par_source[s] = round(sum(details[q['id']]['reussie'] for q in l) / len(l), 3)
    simples = [q for q in questions if q['source'] != 'combinee']
    combinees = [q for q in questions if q['source'] == 'combinee']
    routage = None
    if len(outils) > 1:
        routage = {
            'bon_premier_outil': round(sum(bool(details[q['id']]['outils_appeles']) and details[q['id']]['outils_appeles'][0] in q['outils']
                                           for q in simples) / len(simples), 3),
            'tous_les_outils_attendus_appeles': round(sum(set(q['outils']) <= set(details[q['id']]['outils_appeles'])
                                                          for q in questions) / n, 3),
            'combinees_avec_les_deux_outils': f"{sum(set(q['outils']) <= set(details[q['id']]['outils_appeles']) for q in combinees)}/{len(combinees)}",
        }
    return {
        'interface': nom,
        'outils': outils,
        'bonnes_reponses': round(sum(d['reussie'] for d in details.values()) / n, 3),
        'bonnes_reponses_par_source': par_source,
        'routage': routage,
        'appels_d_outils_par_question': round(sum(len(d['outils_appeles']) for d in details.values()) / n, 2),
        'questions_sans_appel_d_outil': sum(not d['outils_appeles'] for d in details.values()),
        'relances': sum(d['relance'] for d in details.values()),
        'mots_envoyes_par_question': round(sum(d['mots_envoyes'] for d in details.values()) / n),
        'details': details,
    }


async def evaluer(mode):
    questions = json.loads((RACINE / 'evaluation' / 'questions-agent.json').read_text(encoding='utf-8'))
    modele = Modele(mode)
    resultats = []
    for interface, nom in INTERFACES:
        outils, details = await evaluer_interface(interface, questions, modele, mode)
        resultats.append(mesures(nom, outils, questions, details))
        r = resultats[-1]
        print(f"{nom} : bonnes réponses {r['bonnes_reponses']:.0%} {r['bonnes_reponses_par_source']}, routage {r['routage']}, "
              f"{r['appels_d_outils_par_question']} appels d'outils, {r['mots_envoyes_par_question']} mots par question", flush=True)
    return resultats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['cache', 'direct'], default='cache')
    parser.add_argument('--verifier', action='store_true')
    args = parser.parse_args()
    resultats = asyncio.run(evaluer(args.mode))
    publie = RESULTATS / 'resultats-agent.json'
    if args.verifier:
        identique = json.loads(json.dumps(resultats, ensure_ascii=False)) == json.loads(publie.read_text(encoding='utf-8'))
        print('Résultats identiques à ceux publiés.' if identique else 'ERREUR : les résultats diffèrent de ceux publiés.')
        sys.exit(0 if identique else 1)
    publie.write_text(json.dumps(resultats, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'Résultats écrits dans {publie.relative_to(RACINE)}')


if __name__ == '__main__':
    main()
