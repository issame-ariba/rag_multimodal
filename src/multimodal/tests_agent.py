
"""
Étape 7 : tests et validation de l'agent RAG multimodal.

Pour chaque scénario, on analyse le comportement dynamique de l'agent :
- décision d'appeler (ou non) l'outil, nombre d'appels et arguments ;
- contextes récupérés (textes / images, pages) ;
- réponse finale.

Le rapport est écrit dans sorties/rapport_tests.md.
"""

import json

from agent import construire_agent, interroger_agent
from config import SORTIES_DIR
from recherche import creer_image_requete


def scenarios():
    return [
        {
            "nom": "1. Question textuelle précise",
            "attendu": "outil, modalité texte",
            "question": "Pour le modèle Transformer de base, combien de couches ont l'encodeur "
                        "et le décodeur, et quelles sont les valeurs de d_model, d_ff et du "
                        "nombre de têtes d'attention ?",
        },
        {
            "nom": "2. Texte → figure",
            "attendu": "outil, modalité image",
            "question": "Retrouve la figure 2 de l'article, qui présente Scaled Dot-Product "
                        "Attention et Multi-Head Attention, et explique les opérations représentées.",
        },
        {
            "nom": "3. Requête par image",
            "attendu": "outil avec chemin_image",
            "question": "À quelle figure de l'article correspond cette image ? Explique les rôles "
                        "de l'encodeur et du décodeur qu'elle représente.",
            "image": creer_image_requete(),
        },
        {
            "nom": "4. Hybride texte + image",
            "attendu": "plusieurs appels (texte et image)",
            "question": "Quels scores BLEU obtient le Transformer (big) en anglais-allemand et "
                        "anglais-français ? Montre aussi la figure de l'attention multi-têtes "
                        "et explique comment elle permet ces performances.",
        },
        {
            "nom": "5. Question de suivi (mémoire)",
            "attendu": "outil, question comprise grâce à l'historique",
            "question": "Et pour le modèle de base, quels sont ces deux scores ?",
            "suite": True,
        },
        {
            "nom": "6. Information absente",
            "attendu": "outil puis « absent du document »",
            "question": "D'après Attention Is All You Need, quelle précision GPT-4 obtient-il "
                        "sur le benchmark MMLU ?",
        },
        {
            "nom": "7. Hors sujet",
            "attendu": "pas d'appel d'outil",
            "question": "Bonjour ! Peux-tu te présenter en une phrase ?",
        },
    ]


def ecrire_rapport(resultats):
    """Rapport Markdown : tableau de synthèse puis détail de chaque test."""

    lignes = [
        "# Rapport de tests — agent RAG multimodal\n",
        "| Test | Attendu | Appels outil | Textes | Images | Pages | Durée |",
        "|---|---|---|---|---|---|---|",
    ]

    for scenario, r in resultats:
        pages = ", ".join(map(str, sorted({s["page"] for s in r["sources"]}))) or "—"
        lignes.append(
            f"| {scenario['nom']} | {scenario['attendu']} | {len(r['appels'])} "
            f"| {sum(s['type'] == 'texte' for s in r['sources'])} "
            f"| {sum(s['type'] == 'image' for s in r['sources'])} "
            f"| {pages} | {r['duree']:.0f} s |"
        )

    for scenario, r in resultats:
        lignes += [f"\n## {scenario['nom']}\n", f"**Question :** {r['question']}\n"]

        if r["raisonnements"]:
            lignes += ["**Premier raisonnement de l'agent :**\n",
                       "> " + r["raisonnements"][0][:600].replace("\n", " ") + "\n"]

        lignes.append("**Appels de l'outil :**\n")
        lignes += [f"- `{json.dumps(a['args'], ensure_ascii=False)}`" for a in r["appels"]] \
            or ["- aucun"]

        if r["images"]:
            lignes.append("\n**Figures retrouvées :** "
                          + ", ".join(f"`{i.split(chr(92))[-1]}`" for i in r["images"]))

        lignes += ["\n**Réponse finale :**\n", r["reponse"] or "_(vide)_"]

    SORTIES_DIR.mkdir(parents=True, exist_ok=True)
    chemin = SORTIES_DIR / "rapport_tests.md"
    chemin.write_text("\n".join(lignes), encoding="utf-8")

    return chemin


def lancer_tests():
    agent = construire_agent(memoire=True)
    resultats = []
    thread_id = None

    for numero, scenario in enumerate(scenarios(), start=1):
        # Nouvelle conversation, sauf pour une question de suivi
        if not scenario.get("suite"):
            thread_id = f"test-{numero}"

        print("\n" + "#" * 70)
        print(f"# TEST {scenario['nom']}  (attendu : {scenario['attendu']})")
        print(f"# {scenario['question']}")
        print("#" * 70)

        resultat = interroger_agent(
            agent,
            scenario["question"],
            image=scenario.get("image"),
            nom_planche=f"test_{numero}_figures.png",
            thread_id=thread_id,
        )
        resultats.append((scenario, resultat))

    print(f"\nRapport écrit dans : {ecrire_rapport(resultats)}")
