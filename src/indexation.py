
from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# Définir les chemins du projet
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"


def main():
    # Vérifier que des fichiers PDF sont disponibles
    fichiers_pdf = list(DATA_DIR.glob("*.pdf"))

    if not fichiers_pdf:
        print("Aucun PDF trouvé dans le dossier data.")
        return

    print(f"{len(fichiers_pdf)} fichiers PDF trouvés.")

    # Configurer le modèle d'embeddings
    print("Chargement du modèle d'embeddings...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # Lire les fichiers PDF
    print("Lecture des fichiers PDF...")
    documents = SimpleDirectoryReader(
        input_dir=str(DATA_DIR),
        required_exts=[".pdf"],
    ).load_data()

    if not documents:
        print("Aucun contenu exploitable trouvé dans les PDF.")
        return

    print(f"{len(documents)} pages ou parties chargées.")

    # Construire l'index vectoriel
    print("Création de l'index vectoriel...")
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )

    # Sauvegarder l'index sur le disque
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))

    print("\nIndexation terminée avec succès !")
    print(f"Index sauvegardé dans : {STORAGE_DIR}")


if __name__ == "__main__":
    main()
