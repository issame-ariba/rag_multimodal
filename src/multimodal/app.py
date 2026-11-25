
"""
Interface utilisateur (User App) du RAG multimodal agentique, avec Streamlit.

    streamlit run app.py        (depuis src/multimodal)
    python main.py interface

Onglets :
- Agent     : conversation avec l'agent ReAct (texte + image jointe), trace
              du cycle Raisonnement -> Action -> Observation, figures retrouvées ;
- Recherche : recherche multimodale directe (texte ou image), k meilleurs résultats ;
- Document  : contenu extrait du PDF (figures, légendes, texte par page) ;
- Tests     : rapport de l'étape 7.
"""

import json
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage

from config import (
    FICHIER_LEGENDES,
    IMAGE_DIR,
    NOM_MODELE_LLM,
    PDF_PATH,
    REQUETES_DIR,
    SORTIES_DIR,
    TEXT_DIR,
)


st.set_page_config(page_title="RAG Multimodal Agentique", page_icon="🤖", layout="wide")


# ============================================================
# RESSOURCES (chargées une seule fois)
# ============================================================

@st.cache_resource(show_spinner="Chargement de CLIP et de ChromaDB...")
def ressources():
    from embeddings import charger_clip
    from indexation import obtenir_collection

    charger_clip()
    return obtenir_collection()


@st.cache_resource(show_spinner="Construction de l'agent LangGraph...")
def obtenir_agent():
    from agent import construire_agent

    return construire_agent(memoire=True)


def enregistrer_image(fichier):
    """Sauvegarde une image envoyée par l'utilisateur et renvoie son chemin."""

    REQUETES_DIR.mkdir(exist_ok=True)
    chemin = REQUETES_DIR / f"envoi_{uuid.uuid4().hex[:8]}{Path(fichier.name).suffix.lower()}"
    chemin.write_bytes(fichier.getvalue())
    return chemin


def legendes():
    if FICHIER_LEGENDES.exists():
        return json.loads(FICHIER_LEGENDES.read_text(encoding="utf-8"))
    return {}


# ============================================================
# AFFICHAGE DU CYCLE ReAct
# ============================================================

def decrire_etape(message):
    """Transforme un message de l'agent en étapes affichables."""

    etapes = []

    if isinstance(message, AIMessage):
        raisonnement = message.additional_kwargs.get("reasoning_content")
        if raisonnement:
            etapes.append({"type": "raisonnement", "texte": raisonnement.strip()})
        for appel in message.tool_calls:
            etapes.append({"type": "action", "texte": json.dumps(appel["args"], ensure_ascii=False)})

    elif isinstance(message, ToolMessage):
        try:
            sources = json.loads(message.content)
            etapes.append({"type": "observation", "sources": sources})
        except json.JSONDecodeError:
            etapes.append({"type": "observation", "sources": [], "texte": message.content})

    return etapes


def afficher_etape(etape):
    if etape["type"] == "raisonnement":
        with st.expander("💭 Raisonnement (MIND)"):
            st.markdown(etape["texte"])

    elif etape["type"] == "action":
        st.markdown("🔧 **Action (BODY)** : appel de `retriever_tool`")
        st.code(etape["texte"], language="json")

    elif etape["type"] == "observation":
        sources = etape["sources"]
        if not sources:
            st.warning(f"👁 Observation : {etape.get('texte', 'aucun résultat')}")
            return
        textes = sum(s["type"] == "texte" for s in sources)
        st.markdown(f"👁 **Observation** : {textes} passage(s), {len(sources) - textes} figure(s)")
        st.dataframe(
            [{"Réf.": s["reference"], "Type": s["type"], "Page": s["page"],
              "Similarité": s["similarite"], "Contenu": s["contenu"][:160]} for s in sources],
            hide_index=True, width="stretch",
        )


def afficher_figures(sources, colonnes=4):
    """Affiche les figures retrouvées, de la plus similaire à la moins similaire."""

    images, vues = [], set()
    for s in sorted(sources, key=lambda s: -s["similarite"]):
        if s["type"] == "image" and s["chemin_image"] not in vues:
            vues.add(s["chemin_image"])
            images.append(s)

    if not images:
        return

    st.markdown("**🖼 Figures retrouvées**")
    for debut in range(0, len(images), colonnes):
        for colonne, s in zip(st.columns(colonnes), images[debut:debut + colonnes]):
            colonne.image(s["chemin_image"], width="stretch",
                          caption=f"p. {s['page']} · similarité {s['similarite']:.3f}")


def afficher_echange(echange):
    """Réaffiche un échange de l'historique."""

    with st.chat_message("user"):
        st.markdown(echange["question"])
        if echange["image"]:
            st.image(echange["image"], width=220, caption="Image jointe")

    with st.chat_message("assistant"):
        with st.expander(f"Cycle ReAct — {echange['resume']}"):
            for etape in echange["etapes"]:
                afficher_etape(etape)
        st.markdown(echange["reponse"])
        afficher_figures(echange["sources"])


# ============================================================
# BARRE LATÉRALE
# ============================================================

collection = ressources()

with st.sidebar:
    st.title("🤖 RAG Multimodal Agentique")
    st.caption("Attention Is All You Need · CLIP · ChromaDB · LangGraph · Ollama")

    nombre_textes = len(collection.get(where={"type": "texte"}, include=[])["ids"])
    nombre_images = len(collection.get(where={"type": "image"}, include=[])["ids"])
    colonne_1, colonne_2 = st.columns(2)
    colonne_1.metric("Fragments", nombre_textes)
    colonne_2.metric("Figures", nombre_images)

    from agent import probleme_ollama
    probleme = probleme_ollama(NOM_MODELE_LLM)
    if probleme:
        st.error(probleme)
    else:
        st.success(f"LLM : {NOM_MODELE_LLM} (Ollama)")

    if st.button("🔄 Réindexer le PDF", width="stretch"):
        from indexation import indexer_pdf
        with st.spinner("Extraction, vectorisation et indexation..."):
            indexer_pdf(forcer=True)
        st.rerun()

    if st.button("🧹 Nouvelle conversation", width="stretch"):
        st.session_state.thread_id = uuid.uuid4().hex
        st.session_state.historique = []
        st.rerun()

    st.divider()
    st.markdown(
        "**Architecture**\n"
        "- **MIND** : LLM qui raisonne et décide\n"
        "- **BODY** : `retriever_tool` (recherche CLIP dans ChromaDB)\n"
        "- **ReAct** : Raisonnement → Action → Observation\n"
        "- **Mémoire** : historique de la conversation"
    )

if nombre_textes == 0:
    st.warning("La base est vide : cliquer sur « Réindexer le PDF » dans la barre latérale.")
    st.stop()

st.session_state.setdefault("thread_id", uuid.uuid4().hex)
st.session_state.setdefault("historique", [])
st.session_state.setdefault("numero_envoi", 0)

onglet_agent, onglet_recherche, onglet_document, onglet_tests = st.tabs(
    ["💬 Agent", "🔎 Recherche", "📄 Document", "🧪 Tests"]
)


# ============================================================
# ONGLET AGENT
# ============================================================

with onglet_agent:
    st.subheader("Interroger l'agent sur l'article")
    st.caption("Question en français ; image facultative (capture d'une figure, schéma...). "
               f"Chaque réponse prend environ 1 à 3 minutes avec {NOM_MODELE_LLM} en local.")

    for echange in st.session_state.historique:
        afficher_echange(echange)

    image_envoyee = st.file_uploader(
        "📎 Joindre une image à la prochaine question (facultatif)",
        type=["png", "jpg", "jpeg", "webp"],
        key=f"image_{st.session_state.numero_envoi}",
    )

    with st.form("formulaire_question", clear_on_submit=True):
        question = st.text_area("Ta question", height=80,
                                placeholder="Ex. : Pourquoi le Transformer utilise-t-il "
                                            "plusieurs têtes d'attention ?")
        envoyer = st.form_submit_button("Envoyer", type="primary", disabled=bool(probleme))

    if envoyer and question.strip():
        from agent import interroger_agent

        chemin_image = enregistrer_image(image_envoyee) if image_envoyee else None

        with st.chat_message("user"):
            st.markdown(question)
            if chemin_image:
                st.image(str(chemin_image), width=220, caption="Image jointe")

        with st.chat_message("assistant"):
            etapes = []
            statut = st.status("L'agent raisonne...", expanded=True)

            def sur_etape(message):
                for etape in decrire_etape(message):
                    etapes.append(etape)
                    with statut:
                        afficher_etape(etape)

            resultat = interroger_agent(
                obtenir_agent(), question.strip(), image=chemin_image, afficher=False,
                thread_id=st.session_state.thread_id, sur_etape=sur_etape,
            )

            resume = f"{len(resultat['appels'])} appel(s) d'outil · {resultat['duree']} s"
            statut.update(label=f"Cycle ReAct terminé — {resume}", state="complete",
                          expanded=False)
            st.markdown(resultat["reponse"])
            afficher_figures(resultat["sources"])

        st.session_state.historique.append({
            "question": question,
            "image": str(chemin_image) if chemin_image else None,
            "etapes": etapes,
            "reponse": resultat["reponse"],
            "sources": resultat["sources"],
            "resume": resume,
        })
        st.session_state.numero_envoi += 1  # vide le champ d'image
        st.rerun()


# ============================================================
# ONGLET RECHERCHE
# ============================================================

with onglet_recherche:
    st.subheader("Recherche multimodale par similarité cosinus")
    st.caption("Requête texte (de préférence en anglais, la langue de l'article) ou image.")

    colonne_requete, colonne_options = st.columns([3, 1])
    with colonne_requete:
        texte_requete = st.text_input("Requête texte", placeholder="multi-head attention")
        image_requete = st.file_uploader("… ou une image requête",
                                         type=["png", "jpg", "jpeg", "webp"], key="image_recherche")
    with colonne_options:
        type_resultat = st.radio("Modalité", ["tous", "texte", "image"], horizontal=True,
                                 help="« tous » = k passages + k figures (une recherche par modalité)")
        k = st.slider("k", 1, 10, 5)

    if st.button("Rechercher", type="primary"):
        from recherche import recuperer_sources

        if image_requete:
            chemin = enregistrer_image(image_requete)
            st.image(str(chemin), width=220, caption="Image requête")
            sources = recuperer_sources(image=chemin, type_resultat=type_resultat, k=k)
        elif texte_requete.strip():
            sources = recuperer_sources(texte=texte_requete.strip(), type_resultat=type_resultat, k=k)
        else:
            st.warning("Saisir une requête texte ou choisir une image.")
            sources = []

        figures = [s for s in sources if s["type"] == "image"]
        passages = [s for s in sources if s["type"] == "texte"]

        if figures:
            st.markdown(f"**Figures ({len(figures)})**")
            for colonne, s in zip(st.columns(max(len(figures), 3)), figures):
                colonne.image(s["chemin_image"], width="stretch",
                              caption=f"p. {s['page']} · similarité {s['similarite']:.3f}")

        if passages:
            st.markdown(f"**Passages ({len(passages)})**")
            for s in passages:
                with st.container(border=True):
                    st.markdown(f"**{s['reference']}** · page {s['page']} · "
                                f"similarité cosinus **{s['similarite']:.3f}**")
                    st.write(s["contenu"])


# ============================================================
# ONGLET DOCUMENT
# ============================================================

with onglet_document:
    from indexation import calculer_document_id

    st.subheader(PDF_PATH.name)
    st.caption(f"Identifiant SHA-256 : `{calculer_document_id(PDF_PATH)}`")

    st.markdown("**Figures extraites**")
    toutes_legendes = legendes()
    fichiers_images = sorted(IMAGE_DIR.glob("*")) if IMAGE_DIR.exists() else []
    for colonne, fichier in zip(st.columns(max(len(fichiers_images), 1)), fichiers_images):
        colonne.image(str(fichier), width="stretch", caption=fichier.name)
        colonne.caption(toutes_legendes.get(fichier.name, ""))

    pages = sorted(TEXT_DIR.glob("page_*.txt")) if TEXT_DIR.exists() else []
    if pages:
        page = st.selectbox("Texte extrait de la page", pages, format_func=lambda p: p.stem)
        st.text(page.read_text(encoding="utf-8"))


# ============================================================
# ONGLET TESTS
# ============================================================

with onglet_tests:
    rapport = SORTIES_DIR / "rapport_tests.md"
    if rapport.exists():
        st.markdown(rapport.read_text(encoding="utf-8"))
    else:
        st.info("Aucun rapport : lancer `python main.py tests`.")
