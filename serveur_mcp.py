"""Point d'entrée du serveur MCP pour Claude Desktop, Claude Code ou tout client MCP : python <chemin>/serveur_mcp.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from connaissances.serveur import creer_serveur  # noqa: E402

if __name__ == '__main__':
    creer_serveur().run('stdio')
