
"""
Étape 3 : indexation des textes et des images dans ChromaDB.

L'identifiant du document est l'empreinte SHA-256 de son contenu :
- même empreinte que dans la base -> indexation ignorée ;
- PDF modifié (ou --forcer) -> ses anciennes entrées sont supprimées
  puis le document est réindexé.
"""

import hashlib
import json
import re
import threading
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHEVAUCHEMENT,
    CHROMA_DIR,
    EXTENSIONS_IMAGES,
    FICHIER_LEGENDES,
    IMAGE_DIR,
    NOM_COLLECTION,
    PDF_PATH,
    TAILLE_CHUNK,
    TEXT_DIR,
)
from embeddings import charger_clip, encoder_image, encoder_textes
from extraction import extraire_pdf


TAILLE_LOT = 64


_collection = None
_verrou_collection = threading.Lock()


def obtenir_collection():
    """
    Ouvre (ou crée) la collection persistante, en similarité cosinus.

    Elle n'est ouverte qu'une fois : l'agent peut appeler l'outil plusieurs
    fois en parallèle, et ChromaDB n'accepte pas plusieurs clients créés
    simultanément sur le même dossier.
    """

    global _collection

    with _verrou_collection:
        if _collection is None:
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            _collection = client.get_or_create_collection(
                name=NOM_COLLECTION,
                embedding_function=None,  # les embeddings sont calculés par CLIP
                metadata={"hnsw:space": "cosine"},
            )

    return _collection


def calculer_document_id(chemin_pdf):
    """Identifiant lié au contenu du PDF (SHA-256)."""

    return hashlib.sha256(Path(chemin_pdf).read_bytes()).hexdigest()[:24]


def decouper_texte(texte, taille=TAILLE_CHUNK, chevauchement=CHEVAUCHEMENT):
    """
    Découpe le texte en fragments courts adaptés à CLIP, en privilégiant
    les limites naturelles (paragraphes, lignes, mots).

    La longueur est mesurée en tokens BPE de CLIP, mais le texte d'origine
    est conservé : décoder les tokens BPE abîmerait le texte transmis au
    LLM (« 28.4 » deviendrait « 2 8 . 4 », tout passerait en minuscules).
    """

    if not 0 <= chevauchement < taille:
        raise ValueError("Le chevauchement doit être inférieur à la taille.")

    _, _, _, bpe = charger_clip()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=taille,
        chunk_overlap=chevauchement,
        length_function=lambda morceau: len(bpe.encode(morceau)),
        separators=["\n\n", "\n", " ", ""],
        strip_whitespace=True,
    )

    for fragment in splitter.split_text(texte):
        # Recoller les mots coupés en fin de ligne et aplatir les espaces
        fragment = " ".join(re.sub(r"-\n(\w)", r"\1", fragment).split())
        if fragment:
            yield fragment


def numero_page(chemin):
    correspondance = re.search(r"page_(\d+)", Path(chemin).stem)
    return int(correspondance.group(1)) if correspondance else -1


def indexer_textes(collection, document_id, nom_pdf):
    """Découpe chaque page en fragments et les indexe par lots."""

    ids, documents, metadatas = [], [], []

    for fichier in sorted(TEXT_DIR.glob("page_*.txt")):
        page = numero_page(fichier)
        texte = fichier.read_text(encoding="utf-8")

        for index, fragment in enumerate(decouper_texte(texte)):
            ids.append(f"{document_id}:texte:{page}:{index}")
            documents.append(fragment)
            metadatas.append({
                "type": "texte",
                "document_id": document_id,
                "pdf": nom_pdf,
                "page": page,
                "fragment": index,
                "source": str(fichier.resolve()),
            })

    for debut in range(0, len(documents), TAILLE_LOT):
        fin = debut + TAILLE_LOT
        collection.upsert(
            ids=ids[debut:fin],
            documents=documents[debut:fin],
            metadatas=metadatas[debut:fin],
            embeddings=encoder_textes(documents[debut:fin]),
        )

    return len(documents)


def indexer_images(collection, document_id, nom_pdf):
    """Indexe chaque image ; l'embedding est calculé à partir des pixels."""

    legendes = json.loads(FICHIER_LEGENDES.read_text(encoding="utf-8"))
    nombre = 0

    for fichier in sorted(IMAGE_DIR.iterdir()):
        if not fichier.is_file() or fichier.suffix.lower() not in EXTENSIONS_IMAGES:
            continue

        try:
            legende = legendes.get(fichier.name, "")
            collection.upsert(
                ids=[f"{document_id}:image:{fichier.name}"],
                embeddings=[encoder_image(fichier)],
                # La légende décrit l'image au LLM, qui ne voit pas les pixels.
                documents=[legende or f"Image extraite du PDF : {fichier.name}"],
                metadatas=[{
                    "type": "image",
                    "document_id": document_id,
                    "pdf": nom_pdf,
                    "page": numero_page(fichier),
                    "source": str(fichier.resolve()),
                    "legende": legende,
                }],
            )
            nombre += 1

        except Exception as erreur:
            print(f"Image ignorée : {fichier.name} — {erreur}")

    return nombre


def indexer_pdf(chemin_pdf=PDF_PATH, forcer=False):
    """Étapes 1 à 3 : extraction, vectorisation et stockage dans ChromaDB."""

    collection = obtenir_collection()
    document_id = calculer_document_id(chemin_pdf)

    print(f"Identifiant du document (SHA-256) : {document_id}")
    print(f"Éléments déjà présents : {collection.count()}")

    deja_indexe = bool(collection.get(where={"document_id": document_id}, limit=1)["ids"])

    if deja_indexe and not forcer:
        print("PDF déjà indexé avec la même empreinte : indexation ignorée.")
        print("(utiliser --forcer pour réindexer)")
        return collection

    # Supprimer toutes les versions précédentes de ce PDF
    collection.delete(where={"pdf": chemin_pdf.name})

    extraire_pdf(chemin_pdf)

    print(f"{indexer_textes(collection, document_id, chemin_pdf.name)} fragments de texte indexés.")
    print(f"{indexer_images(collection, document_id, chemin_pdf.name)} images indexées.")
    print(f"Total dans ChromaDB : {collection.count()} éléments.")

    return collection
