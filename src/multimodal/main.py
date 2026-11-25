
"""
RAG multimodal agentique — point d'entrée du projet.

    python main.py extraire                 # étape 1
    python main.py embeddings               # étape 2 (vérification de l'espace latent)
    python main.py indexer [--forcer]       # étapes 1 à 3
    python main.py rechercher "requête"     # étape 4 (texte)
    python main.py rechercher --image x.png # étape 4 (image)
    python main.py demo-recherche           # étape 4 (exemples A, B, C)
    python main.py agent                    # étapes 5 et 6 (mode interactif)
    python main.py tests                    # étape 7
    python main.py interface                # interface web (Streamlit)
"""

import argparse
import subprocess
import sys
from pathlib import Path

import config  # noqa: F401  (console UTF-8, options Hugging Face)


def main():
    parser = argparse.ArgumentParser(description="RAG multimodal agentique")
    commandes = parser.add_subparsers(dest="commande", required=True)

    commandes.add_parser("extraire", help="extraire textes et images du PDF")
    commandes.add_parser("embeddings", help="vérifier les embeddings CLIP")

    indexer = commandes.add_parser("indexer", help="indexer le PDF dans ChromaDB")
    indexer.add_argument("--forcer", action="store_true", help="réindexer même si inchangé")

    recherche = commandes.add_parser("rechercher", help="recherche par texte ou image")
    recherche.add_argument("texte", nargs="?")
    recherche.add_argument("--image")
    recherche.add_argument("--type", choices=["texte", "image"], default=None)
    recherche.add_argument("-k", type=int, default=5)

    commandes.add_parser("demo-recherche", help="exemples de recherche A, B, C")
    commandes.add_parser("agent", help="dialoguer avec l'agent")
    commandes.add_parser("tests", help="lancer les scénarios de test de l'agent")
    commandes.add_parser("interface", help="ouvrir l'interface web (Streamlit)")

    arguments = parser.parse_args()

    if arguments.commande == "extraire":
        from extraction import afficher_apercu, extraire_pdf
        extraire_pdf()
        afficher_apercu()

    elif arguments.commande == "embeddings":
        from embeddings import verifier_espace_latent
        verifier_espace_latent()

    elif arguments.commande == "indexer":
        from indexation import indexer_pdf
        indexer_pdf(forcer=arguments.forcer)

    elif arguments.commande == "rechercher":
        from recherche import afficher_resultats, rechercher
        afficher_resultats(rechercher(texte=arguments.texte, image=arguments.image,
                                      type_resultat=arguments.type, k=arguments.k))

    elif arguments.commande == "demo-recherche":
        from recherche import demonstration
        demonstration()

    elif arguments.commande == "agent":
        from agent import mode_interactif
        mode_interactif()

    elif arguments.commande == "tests":
        from tests_agent import lancer_tests
        lancer_tests()

    elif arguments.commande == "interface":
        application = Path(__file__).with_name("app.py")
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(application)])
        except KeyboardInterrupt:
            print("\nInterface arrêtée.")


if __name__ == "__main__":
    main()
