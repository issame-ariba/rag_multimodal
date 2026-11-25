
"""
Étape 4 : recherche multimodale par similarité cosinus.

La requête est soit un texte, soit une image, encodé(e) par CLIP dans le
même espace que les données indexées.

Remarque : avec CLIP, les similarités texte→texte (~0,7-0,9) sont bien
plus élevées que les similarités texte→image (~0,2-0,3) : c'est le
« modality gap ». Sans filtre, une requête texte ne renvoie donc que des
textes. La recherche « tous » interroge chaque modalité séparément.
"""

import re
from pathlib import Path

import pymupdf

from config import NOMBRE_RESULTATS, PDF_PATH, REQUETES_DIR
from embeddings import encoder_image, encoder_textes
from extraction import creer_planche
from indexation import obtenir_collection


def rechercher(texte=None, image=None, type_resultat=None, k=NOMBRE_RESULTATS):
    """
    Fournir soit un texte, soit un chemin d'image.
    type_resultat : None, "texte" ou "image".
    """

    if (texte is None) == (image is None):
        raise ValueError("Fournissez soit un texte, soit une image.")
    if type_resultat not in (None, "texte", "image"):
        raise ValueError("type_resultat doit être 'texte', 'image' ou None.")
    if k < 1:
        raise ValueError("k doit être supérieur ou égal à 1.")
    if image is not None and not Path(image).is_file():
        raise FileNotFoundError(f"Image introuvable : {image}")

    collection = obtenir_collection()
    total = collection.count()
    if total == 0:
        raise ValueError("La collection est vide : lancer d'abord l'indexation.")

    vecteur = encoder_textes([texte])[0] if texte is not None else encoder_image(image)

    options = {"where": {"type": type_resultat}} if type_resultat else {}

    return collection.query(
        query_embeddings=[vecteur],
        n_results=min(k, total),
        include=["documents", "metadatas", "distances"],
        **options,
    )


def afficher_resultats(resultats, nom_planche="apercu_resultats.png"):
    """Affiche les k meilleurs résultats ; les images vont dans une planche PNG."""

    images, titres = [], []

    for rang, (document, metadata, distance) in enumerate(
        zip(resultats["documents"][0], resultats["metadatas"][0], resultats["distances"][0]),
        start=1,
    ):
        print(f"\n{rang}. {metadata['type'].upper()} | PDF : {metadata['pdf']} "
              f"| Page : {metadata['page']} | Similarité cosinus : {1 - distance:.3f}")

        if metadata["type"] == "image":
            print(f"   Fichier : {Path(metadata['source']).name}")
            print(f"   Légende : {document[:150]}")
            images.append(metadata["source"])
            titres.append(f"{rang}. p.{metadata['page']}  {1 - distance:.3f}")
        else:
            print(f"   {document}")

    if images:
        print(f"\nAperçu des images : {creer_planche(images, titres, nom_planche)}")


def elargir_contexte(identifiant):
    """
    Texte du fragment entouré de ses voisins (fragment précédent et suivant
    de la même page). Un fragment de 60 tokens, adapté à CLIP, est souvent
    trop court pour que le LLM comprenne un passage ou un tableau.
    """

    document_id, _, page, index = identifiant.rsplit(":", 3)
    voisins = [f"{document_id}:texte:{page}:{int(index) + decalage}" for decalage in (-1, 0, 1)]

    resultat = obtenir_collection().get(ids=voisins, include=["documents"])
    par_id = dict(zip(resultat["ids"], resultat["documents"]))

    return " […] ".join(par_id[i] for i in voisins if i in par_id)


def recuperer_sources(texte=None, image=None, type_resultat="tous", k=NOMBRE_RESULTATS,
                      elargir=False):
    """
    Résultats structurés pour l'agent. type_resultat="tous" renvoie
    k textes + k images (une recherche par modalité).
    elargir=True remplace chaque fragment de texte par son contexte élargi.
    """

    types = ["texte", "image"] if type_resultat == "tous" else [type_resultat]
    sources = []

    for type_courant in types:
        resultats = rechercher(texte=texte, image=image, type_resultat=type_courant, k=k)

        for identifiant, document, metadata, distance in zip(
            resultats["ids"][0],
            resultats["documents"][0],
            resultats["metadatas"][0],
            resultats["distances"][0],
        ):
            sources.append({
                "reference": f"S{len(sources) + 1}",
                "id": identifiant,
                "type": metadata["type"],
                "page": metadata["page"],
                "similarite": round(1 - distance, 3),
                "contenu": (
                    elargir_contexte(identifiant)
                    if elargir and metadata["type"] == "texte" else document
                ),
                "chemin_image": metadata["source"] if metadata["type"] == "image" else None,
            })

            # Numéro de figure lu dans la légende (« Figure 1: ... » -> « Figure 1 »)
            if metadata["type"] == "image":
                numero = re.match(r"Figure \d+", document)
                sources[-1]["figure"] = numero.group(0) if numero else "inconnue"

    return sources


def creer_image_requete():
    """
    Simule une capture d'écran de la figure 1 faite par un utilisateur :
    la zone de la figure est rendue depuis la page 3 à une autre résolution,
    les pixels diffèrent donc de l'image indexée.
    """

    REQUETES_DIR.mkdir(exist_ok=True)
    chemin = REQUETES_DIR / "architecture_transformer.png"

    if not chemin.exists():
        with pymupdf.open(PDF_PATH) as document:
            page = document[2]
            zone = page.get_image_rects(page.get_images()[0][0])[0]
            page.get_pixmap(clip=zone, dpi=60).save(chemin)

    return chemin


def demonstration():
    """Exemples A, B, C de recherche multimodale."""

    print("=" * 70)
    print("Exemple A — requête texte, sans filtre")
    print("=" * 70)
    afficher_resultats(rechercher(
        texte="the Transformer model architecture with encoder and decoder stacks", k=5
    ))

    print("\n" + "=" * 70)
    print("Exemple B — requête texte, uniquement des images")
    print("=" * 70)
    afficher_resultats(
        rechercher(texte="a diagram of the Transformer model architecture",
                   type_resultat="image", k=3),
        nom_planche="exemple_B_texte_vers_image.png",
    )

    print("\n" + "=" * 70)
    print("Exemple C — requête par image (capture de la figure 1)")
    print("=" * 70)
    image_requete = creer_image_requete()
    print(f"Image requête : {image_requete}")
    afficher_resultats(
        rechercher(image=image_requete, type_resultat="image", k=3),
        nom_planche="exemple_C_image_vers_image.png",
    )
