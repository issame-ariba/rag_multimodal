
"""
Configuration commune du RAG multimodal agentique (texte + images).
"""

import os
import sys
from pathlib import Path


# La console Windows (cp1252) ne sait pas afficher certains
# symboles mathématiques de l'article.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Le téléchargement Hugging Face via « xet » reste bloqué sur cette
# machine : on force le téléchargement HTTP classique.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Si les poids de CLIP sont déjà téléchargés, travailler hors ligne :
# pas de vérification réseau au démarrage (ni avertissement HF_TOKEN).
_CACHE_CLIP = (Path.home() / ".cache" / "huggingface" / "hub"
               / "models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K" / "snapshots")
if any(_CACHE_CLIP.glob("*/open_clip_model.safetensors")):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


# ============================================================
# CHEMINS DU PROJET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

PDF_PATH = BASE_DIR / "data" / "Attention is all you need.pdf"

# Résultats de l'extraction (étape 1)
OUTPUT_DIR = BASE_DIR / "resultats_pdf"
TEXT_DIR = OUTPUT_DIR / "textes"
IMAGE_DIR = OUTPUT_DIR / "images"
FICHIER_LEGENDES = OUTPUT_DIR / "legendes.json"

# Base vectorielle (étape 3)
CHROMA_DIR = BASE_DIR / "chroma_pdf"
NOM_COLLECTION = "pdf_clip_vit_b32"

# Images utilisées comme requêtes et sorties (aperçus, rapports)
REQUETES_DIR = BASE_DIR / "requetes"
SORTIES_DIR = BASE_DIR / "sorties"


# ============================================================
# MODÈLE D'EMBEDDING MULTIMODAL (CLIP) — étape 2
# ============================================================

# ViT : Vision Transformer, B : variante Base, 32 : patchs de 32 × 32 pixels
MODEL_NAME = "ViT-B-32"
# Poids entraînés sur ~2 milliards de paires image-texte LAION
PRETRAINED = "laion2b_s34b_b79k"


# ============================================================
# EXTRACTION ET DÉCOUPAGE
# ============================================================

# CLIP ne lit que 77 tokens : fragments de 60 tokens BPE,
# avec 10 tokens de chevauchement pour ne pas couper les idées.
TAILLE_CHUNK = 60
CHEVAUCHEMENT = 10

# Une page sans image intégrée mais avec beaucoup de tracés
# contient une figure vectorielle : on la rend en PNG.
SEUIL_TRACES_FIGURE = 200

EXTENSIONS_IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".jp2"}


# ============================================================
# AGENT — étapes 5 et 6
# ============================================================

NOM_MODELE_LLM = "qwen3:14b"
NOMBRE_RESULTATS = 5
