
# Importer le lecteur de documents de LlamaIndex
from llama_index.core import SimpleDirectoryReader
from pathlib import Path

# Chemin vers le dossier contenant les PDF
DOSSIER_PDF = Path(__file__).resolve().parent.parent / "data"

def lire_pdf():
    """Lire les fichiers PDF et afficher leur contenu textuel."""

    # Vérifier que le dossier existe
    if not DOSSIER_PDF.exists():
        print("Erreur : le dossier data est introuvable.")
        return

    # Vérifier qu'il contient au moins un PDF
    fichiers_pdf = list(DOSSIER_PDF.glob("*.pdf"))

    if not fichiers_pdf:
        print("Aucun PDF trouvé dans le dossier data.")
        print("Ajoute un fichier PDF puis relance le programme.")
        return

    try:
        # Charger les documents PDF
        documents = SimpleDirectoryReader(
            input_dir=str(DOSSIER_PDF),
            required_exts=[".pdf"]
        ).load_data()

        print("Lecture du PDF réussie !")
        print("Nombre de pages chargées :", len(documents))

        # Afficher le texte extrait
        for i, document in enumerate(documents, start=1):
            print(f"\n--- Page ou partie {i} ---")
            print(document.text[:1500])

    except Exception as erreur:
        print("Une erreur est survenue :", erreur)

if __name__ == "__main__":
    lire_pdf()
