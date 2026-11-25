
"""
Étape 2 : vectorisation dans un espace latent commun avec CLIP.

Le même modèle encode le texte et les images : une requête texte peut
retrouver une image, et inversement. Tous les vecteurs sont normalisés
(norme L2 = 1) : leur produit scalaire est alors la similarité cosinus.
"""

import threading
from functools import lru_cache

import open_clip
import torch
from open_clip.tokenizer import SimpleTokenizer
from PIL import Image

from config import IMAGE_DIR, MODEL_NAME, PRETRAINED


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


_verrou_clip = threading.Lock()


def charger_clip():
    """Charge CLIP une seule fois, même si l'agent appelle l'outil en parallèle."""

    with _verrou_clip:
        return _charger_clip()


@lru_cache(maxsize=1)
def _charger_clip():
    """Modèle, prétraitement, tokenizer et BPE de CLIP."""

    print(f"Chargement de CLIP {MODEL_NAME} ({PRETRAINED}) sur {DEVICE}...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model = model.to(DEVICE).eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    # Tokeniseur BPE de CLIP, utilisé pour découper le texte en fragments
    bpe = SimpleTokenizer()

    return model, preprocess, tokenizer, bpe


@torch.inference_mode()
def encoder_textes(textes):
    """Textes -> tokenisation -> encodage CLIP -> normalisation L2 -> listes."""

    model, _, tokenizer, _ = charger_clip()

    tokens = tokenizer(textes).to(DEVICE)
    vecteurs = model.encode_text(tokens).float()
    vecteurs = vecteurs / vecteurs.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    return vecteurs.cpu().tolist()


@torch.inference_mode()
def encoder_image(chemin):
    """Image -> prétraitement [3, 224, 224] -> encodage CLIP -> normalisation L2."""

    model, preprocess, _, _ = charger_clip()

    with Image.open(chemin) as image:
        tenseur = preprocess(image.convert("RGB"))

    # Ajouter la dimension du lot : [3, H, W] -> [1, 3, H, W]
    tenseur = tenseur.unsqueeze(0).to(DEVICE)
    vecteur = model.encode_image(tenseur).float()
    vecteur = vecteur / vecteur.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    return vecteur.cpu().tolist()[0]


def verifier_espace_latent():
    """Vérifie la normalisation et compare les 3 figures à 3 descriptions."""

    v_texte = torch.tensor(encoder_textes(["multi-head attention"])[0])
    print(f"Dimension texte : {len(v_texte)} | norme : {v_texte.norm():.4f}")

    # Figure 1 (page 3), puis les deux schémas de la figure 2 (page 4)
    figures = sorted(IMAGE_DIR.glob("page_00[34]_image_*"))
    descriptions = [
        "a diagram of the Transformer encoder-decoder model architecture",
        "a diagram of scaled dot-product attention with MatMul, Scale, Mask and SoftMax blocks",
        "a diagram of multi-head attention with parallel linear layers, Concat and Linear",
    ]

    v_image = torch.tensor(encoder_image(figures[0]))
    print(f"Dimension image : {len(v_image)} | norme : {v_image.norm():.4f}")

    # Vecteurs normalisés : la similarité cosinus est un produit matriciel
    images = torch.tensor([encoder_image(f) for f in figures])
    textes = torch.tensor(encoder_textes(descriptions))
    similarites = images @ textes.T

    print("\nSimilarités cosinus (lignes = images, colonnes = descriptions D0..D2)")
    for numero, description in enumerate(descriptions):
        print(f"  D{numero} : {description}")
    print()
    for figure, ligne in zip(figures, similarites):
        valeurs = "  ".join(f"{s:.3f}" for s in ligne.tolist())
        print(f"  {figure.name:24s} {valeurs}   -> D{ligne.argmax().item()}")
