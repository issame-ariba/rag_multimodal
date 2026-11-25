
"""
Étape 1 : extraction du contenu multimodal du PDF avec PyMuPDF.

resultats_pdf/
├── images/            page_003_image_94.png, page_004_image_102.png, ...
├── textes/            page_001.txt, page_002.txt, ...
├── texte_complet.txt
└── legendes.json      nom de l'image -> légende « Figure N: ... »

Les images intégrées au PDF sont extraites telles quelles. Une figure
dessinée en vectoriel est invisible pour get_images() (c'est le cas des
cartes d'attention de l'annexe dans la version arXiv de l'article) :
elle est détectée et rendue en PNG.
"""

import json
import re

import pymupdf
from PIL import Image

from config import (
    FICHIER_LEGENDES,
    IMAGE_DIR,
    OUTPUT_DIR,
    PDF_PATH,
    SEUIL_TRACES_FIGURE,
    SORTIES_DIR,
    TEXT_DIR,
)


def trouver_legende(page):
    """Retourne la ou les légendes « Figure N: ... » présentes sur la page."""

    legendes = []

    for bloc in page.get_text("blocks"):
        texte = " ".join(bloc[4].split())
        if re.match(r"^Figure \d+", texte):
            legendes.append(texte)

    return " | ".join(legendes)


def extraire_images_integrees(document, page, numero_page, legende, legendes):
    """Sauvegarde les images intégrées de la page ; renvoie leur nombre."""

    nombre = 0
    images_vues = set()

    for image_info in page.get_images(full=True):
        xref = image_info[0]
        if xref in images_vues:
            continue
        images_vues.add(xref)

        try:
            image = document.extract_image(xref)
            if not image:
                print(f"Image non extractible : page {numero_page}, xref {xref}")
                continue

            fichier_image = IMAGE_DIR / f"page_{numero_page:03d}_image_{xref}.{image['ext']}"
            fichier_image.write_bytes(image["image"])
            legendes[fichier_image.name] = legende
            nombre += 1

        except Exception as erreur:
            print(f"Erreur d'extraction : page {numero_page}, image {xref} : {erreur}")

    return nombre


def extraire_figure_vectorielle(page, numero_page, legende, legendes):
    """Rend en PNG la figure dessinée en vectoriel d'une page, s'il y en a une."""

    traces = page.get_drawings()
    if len(traces) < SEUIL_TRACES_FIGURE:
        return 0

    # Les tracés sont regroupés en zones ; la plus grande est la figure.
    zones = page.cluster_drawings(drawings=traces)
    if not zones:
        return 0

    zone = max(zones, key=lambda r: r.width * r.height)
    zone = (zone + (-10, -10, 10, 10)) & page.rect

    fichier_image = IMAGE_DIR / f"page_{numero_page:03d}_figure.png"
    page.get_pixmap(clip=zone, dpi=150).save(fichier_image)
    legendes[fichier_image.name] = legende

    return 1


def extraire_pdf(chemin_pdf=PDF_PATH):
    """Extrait textes (par page) et images du PDF dans resultats_pdf/."""

    if not chemin_pdf.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {chemin_pdf.resolve()}")

    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Vider les résultats d'une extraction précédente (autre version du PDF)
    for ancien in [*TEXT_DIR.iterdir(), *IMAGE_DIR.iterdir()]:
        if ancien.is_file():
            ancien.unlink()

    textes = []
    legendes = {}
    nombre_images = 0
    pages_sans_texte = []

    with pymupdf.open(chemin_pdf) as document:
        nombre_pages = len(document)

        for numero_page, page in enumerate(document, start=1):
            # ---- Texte ----
            texte = page.get_text("text", sort=True)
            (TEXT_DIR / f"page_{numero_page:03d}.txt").write_text(texte, encoding="utf-8")
            textes.append(f"===== PAGE {numero_page} =====\n\n{texte}")

            if not texte.strip():
                pages_sans_texte.append(numero_page)

            # ---- Images ----
            legende = trouver_legende(page)
            images_page = extraire_images_integrees(
                document, page, numero_page, legende, legendes
            )

            if images_page == 0:
                images_page = extraire_figure_vectorielle(page, numero_page, legende, legendes)

            nombre_images += images_page

    (OUTPUT_DIR / "texte_complet.txt").write_text("\n\n".join(textes), encoding="utf-8")
    FICHIER_LEGENDES.write_text(
        json.dumps(legendes, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"PDF : {chemin_pdf.name}")
    print(f"Pages traitées : {nombre_pages}")
    print(f"Images enregistrées : {nombre_images}")

    if pages_sans_texte:
        print("Pages sans texte détectable :", pages_sans_texte)
        print("Si elles contiennent du texte scanné, il faut un OCR.")

    return nombre_pages, nombre_images


def creer_planche(chemins, titres, nom_fichier, hauteur=260):
    """Assemble des images côte à côte dans une planche PNG (aperçu)."""

    from PIL import ImageDraw

    vignettes = []
    for chemin in chemins:
        with Image.open(chemin) as image:
            vignette = image.convert("RGB")
        vignette.thumbnail((hauteur * 2, hauteur))
        vignettes.append(vignette)

    largeur = sum(v.width for v in vignettes) + 10 * (len(vignettes) + 1)
    planche = Image.new("RGB", (largeur, hauteur + 40), "white")
    dessin = ImageDraw.Draw(planche)

    x = 10
    for vignette, titre in zip(vignettes, titres):
        planche.paste(vignette, (x, 30))
        dessin.text((x, 8), titre, fill="black")
        x += vignette.width + 10

    SORTIES_DIR.mkdir(parents=True, exist_ok=True)
    chemin_planche = SORTIES_DIR / nom_fichier
    planche.save(chemin_planche)

    return chemin_planche


def afficher_apercu():
    """Affiche un aperçu du texte extrait et crée une planche des images."""

    texte_complet = (OUTPUT_DIR / "texte_complet.txt").read_text(encoding="utf-8")
    legendes = json.loads(FICHIER_LEGENDES.read_text(encoding="utf-8"))

    print("\nAPERÇU DU TEXTE\n")
    print(texte_complet[:1200])

    print("\nAPERÇU DES IMAGES\n")
    fichiers = sorted(IMAGE_DIR.iterdir())
    for fichier in fichiers:
        print(f"- {fichier.name} : {legendes.get(fichier.name, '')[:90]}...")

    planche = creer_planche(fichiers, [f.name for f in fichiers], "apercu_extraction.png")
    print(f"\nPlanche des images : {planche}")
