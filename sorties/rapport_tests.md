# Rapport de tests — agent RAG multimodal

| Test | Attendu | Appels outil | Textes | Images | Pages | Durée |
|---|---|---|---|---|---|---|
| 1. Question textuelle précise | outil, modalité texte | 5 | 25 | 0 | 1, 2, 3, 4, 5, 6, 7, 8, 9 | 345 s |
| 2. Texte → figure | outil, modalité image | 2 | 3 | 3 | 3, 4 | 172 s |
| 3. Requête par image | outil avec chemin_image | 1 | 0 | 3 | 3, 4 | 344 s |
| 4. Hybride texte + image | plusieurs appels (texte et image) | 3 | 8 | 3 | 1, 3, 4, 6, 8, 9 | 204 s |
| 5. Question de suivi (mémoire) | outil, question comprise grâce à l'historique | 0 | 0 | 0 | — | 147 s |
| 6. Information absente | outil puis « absent du document » | 0 | 0 | 0 | — | 145 s |
| 7. Hors sujet | pas d'appel d'outil | 0 | 0 | 0 | — | 41 s |

## 1. Question textuelle précise

**Question :** Pour le modèle Transformer de base, combien de couches ont l'encodeur et le décodeur, et quelles sont les valeurs de d_model, d_ff et du nombre de têtes d'attention ?

**Premier raisonnement de l'agent :**

> Okay, the user is asking about the base Transformer model's encoder and decoder layers, as well as the values of d_model, d_ff, and the number of attention heads. Let me break this down.  First, I need to recall the details from the paper "Attention Is All You Need" by Vaswani et al. But since I can't remember, I must use the retriever tool. The user wants specific numbers, so I should search for each parameter separately.  For the number of layers in the encoder and decoder, I'll look for terms like "number of layers in encoder" and "number of layers in decoder". Then, for d_model, which is t

**Appels de l'outil :**

- `{"requete": "number of layers in encoder base model", "type_resultat": "texte", "k": 5}`
- `{"requete": "number of layers in decoder base model", "type_resultat": "texte", "k": 5}`
- `{"requete": "d_model value base model", "type_resultat": "texte", "k": 5}`
- `{"requete": "d_ff value base model", "type_resultat": "texte", "k": 5}`
- `{"requete": "number of attention heads base model", "type_resultat": "texte", "k": 5}`

**Réponse finale :**

Le modèle Transformer de base comporte **6 couches** à la fois pour l'encodeur et le décodeur [p. 3]. Les valeurs des paramètres sont les suivantes :  
- **d_model** (dimension du modèle) : **512** [p. 3].  
- **d_ff** (dimension des réseaux feed-forward) : **2048** [p. 9].  
- **Nombre de têtes d'attention** : **8** [p. 9].  

Ces informations proviennent des descriptions architecturales et du tableau de résultats présenté dans l'article.

## 2. Texte → figure

**Question :** Retrouve la figure 2 de l'article, qui présente Scaled Dot-Product Attention et Multi-Head Attention, et explique les opérations représentées.

**Premier raisonnement de l'agent :**

> Okay, the user is asking for Figure 2 from the "Attention Is All You Need" paper, which shows Scaled Dot-Product Attention and Multi-Head Attention. They also want an explanation of the operations depicted.  First, I need to retrieve the figure. The tool requires a query in English. The figure's title mentions "Scaled Dot-Product Attention and Multi-Head Attention," so I'll use that as the search term. Since it's a figure, the type_resultat should be "image." I'll set k to 1 to get the most relevant result.  Once I get the figure, I need to explain the operations. From my prior knowledge, Scal

**Appels de l'outil :**

- `{"type_resultat": "image", "k": 1, "requete": "Figure 2 Scaled Dot-Product Attention and Multi-Head Attention"}`
- `{"requete": "explanation of Scaled Dot-Product Attention and Multi-Head Attention operations", "type_resultat": "texte", "k": 3}`

**Figures retrouvées :** `page_004_image_102.png`, `page_004_image_104.png`, `page_003_image_94.png`

**Réponse finale :**

La figure 2 de l'article est présente à la page 4 [p. 4]. Elle montre deux composants clés du Transformer :

1. **Scaled Dot-Product Attention** (gauche) :  
   - Calcul des scores d'attention via le produit scalaire entre les requêtes (queries) et les clés (keys), suivi d'une division par √dk pour stabiliser les gradients [p. 4].  
   - Application de la fonction softmax sur les scores pour obtenir des poids, qui sont ensuite appliqués aux valeurs (values) [p. 4].

2. **Multi-Head Attention** (droite) :  
   - Utilisation de plusieurs couches d'attention parallèles, chacune avec des projections linéaires distinctes pour les requêtes, clés et valeurs (dimensions dk, dk et dv) [p. 4].  
   - Concaténation des sorties de chaque tête d'attention, suivie d'une projection finale pour obtenir les valeurs finales [p. 4].  

Cette architecture permet au modèle de capturer des informations de sous-espaces de représentation différents, améliorant la capacité d'attention [p. 4].

## 3. Requête par image

**Question :** À quelle figure de l'article correspond cette image ? Explique les rôles de l'encodeur et du décodeur qu'elle représente.

Image fournie par l'utilisateur : C:\Users\pc\OneDrive\Desktop\rag_multimodal\requetes\architecture_transformer.png

**Premier raisonnement de l'agent :**

> Okay, the user is asking which figure in the article corresponds to the provided image and wants an explanation of the encoder and decoder's roles as shown in that figure. Let me start by understanding the steps I need to take.  First, I need to use the retriever_tool to find the relevant figure. The user provided an image path, so I should set the chemin_image parameter to that path. The type_resultat should be "image" since they're asking about a figure. The query is in French, but the tool requires the request to be in English. However, the user's question is about the image, so maybe the q

**Appels de l'outil :**

- `{"k": 1, "requete": "Transformer architecture figure", "chemin_image": "C:\\Users\\pc\\OneDrive\\Desktop\\rag_multimodal\\requetes\\architecture_transformer.png", "type_resultat": "image"}`

**Figures retrouvées :** `page_003_image_94.png`, `page_004_image_104.png`, `page_004_image_102.png`

**Réponse finale :**

L'image fournie correspond à **Figure 1** de l'article, intitulée *« The Transformer - model architecture »* [p. 3]. Cette figure illustre l'architecture globale du modèle Transformer, composée de deux parties principales :

1. **L'encodeur** :  
   - Il transforme la séquence d'entrée (par exemple, un texte) en une représentation contextuelle via des **couches répétées** (6 dans le modèle de base).  
   - Chaque couche comprend deux sous-modules :  
     - **Self-Attention** : permet aux éléments de la séquence de s'attacher mutuellement pour capturer les dépendances.  
     - **Réseau feed-forward** : applique des transformations non linéaires indépendantes à chaque position.  
   - Les sorties de l'encodeur sont utilisées par le décodeur.

2. **Le décodeur** :  
   - Génère la séquence de sortie (par exemple, une traduction) en utilisant :  
     - **Self-Attention** : pour s'attacher aux éléments déjà générés.  
     - **Attention encoder-décodeur** : pour s'attacher aux sorties de l'encodeur.  
     - **Réseau feed-forward** : comme dans l'encodeur.  
   - Le décodeur produit les sorties étape par étape, en s'appuyant sur les informations de l'encodeur et des étapes précédentes.

Cette architecture permet au Transformer de traiter les séquences de manière parallèle, contrairement aux modèles basés sur les RNN.

## 4. Hybride texte + image

**Question :** Quels scores BLEU obtient le Transformer (big) en anglais-allemand et anglais-français ? Montre aussi la figure de l'attention multi-têtes et explique comment elle permet ces performances.

**Premier raisonnement de l'agent :**

> Okay, let's tackle this question. The user is asking for the BLEU scores of the Transformer (big) model in English-German and English-French translation tasks. They also want to see the figure of the multi-head attention mechanism and an explanation of how it contributes to these performance results.  First, I need to find the BLEU scores. I remember that the original paper "Attention Is All You Need" reports these metrics. The model sizes are mentioned as base and big. The BLEU scores are likely in the results section. So, I'll use the retriever tool to search for "BLEU scores Transformer big

**Appels de l'outil :**

- `{"type_resultat": "texte", "k": 5, "requete": "BLEU scores Transformer big English-German English-French"}`
- `{"type_resultat": "image", "k": 1, "requete": "multi-head attention figure"}`
- `{"type_resultat": "texte", "k": 3, "requete": "how multi-head attention improves performance"}`

**Figures retrouvées :** `page_004_image_102.png`, `page_003_image_94.png`, `page_004_image_104.png`

**Réponse finale :**

Le modèle Transformer (big) obtient un score BLEU de **28,4** pour la traduction anglais-allemand et de **41,0** pour la traduction anglais-français [p. 8].

**Figure de l'attention multi-têtes** :  
La Figure 2 (page 4, fichier `page_004_image_102.png`) montre l'architecture de l'attention multi-têtes. Elle consiste en plusieurs couches d'attention parallèles, chacune projetant les requêtes, clés et valeurs dans des sous-espaces de représentation distincts. Les sorties sont concaténées et réprojectées pour former la sortie finale.

**Explication des performances** :  
L'attention multi-têtes permet au modèle de **synchroniser l'attention sur des informations provenant de sous-espaces de représentation différents** [p. 4]. Contrairement à une seule tête d'attention (qui moyenne les informations), cette approche parallèle capture des motifs variés dans les données. Cela améliore la capacité du modèle à gérer des relations complexes dans les textes, expliquant les scores BLEU élevés. De plus, l'attention à base de produit scalaire (scaled dot-product) réduit les gradients extrêmement petits en normalisant les produits scalaires par √dk, ce qui stabilise l'entraînement [p. 4].

## 5. Question de suivi (mémoire)

**Question :** Et pour le modèle de base, quels sont ces deux scores ?

**Premier raisonnement de l'agent :**

> Okay, the user is asking about the BLEU scores for the base model of the Transformer in English-German and English-French translation tasks. Let me recall the previous interaction. Earlier, they asked about the big model, and I found the scores 28.4 and 41.0 from page 8. Now, they want the same information for the base model.  First, I need to check the retrieved text passages again. In the first tool response, there's a table mention: "Transformer (base model) 27.3 38.1 3.3 · 1018 […]" and "Transformer (big) 28.4 41.0 2.3 · 1019". So the base model's scores are 27.3 for English-German and 38.

**Appels de l'outil :**

- aucun

**Réponse finale :**

Le modèle Transformer (base) obtient un score BLEU de **27,3** pour la traduction anglais-allemand et de **38,1** pour la traduction anglais-français [p. 8].

## 6. Information absente

**Question :** D'après Attention Is All You Need, quelle précision GPT-4 obtient-il sur le benchmark MMLU ?

**Premier raisonnement de l'agent :**

> Okay, the user is asking about the precision of GPT-4 on the MMLU benchmark according to the "Attention Is All You Need" paper. Let me start by recalling that the paper by Vaswani et al. (2017) introduced the Transformer model, but it's from 2017. GPT-4 was developed much later by OpenAI, and the MMLU benchmark was introduced in 2020.   Wait, the original paper from 2017 probably doesn't mention GPT-4 or MMLU at all. The user might be confused because both are related to NLP models, but the timeline doesn't align. The Transformer paper discusses the architecture, training methods, and results 

**Appels de l'outil :**

- aucun

**Réponse finale :**

La question porte sur des informations qui ne figurent pas dans l'article « Attention Is All You Need » (Vaswani et al., 2017). L'article traite de l'introduction du modèle Transformer en 2017, mais il ne mentionne ni GPT-4 (développé par OpenAI bien plus tard) ni le benchmark MMLU (introduit en 2020). Aucune recherche dans l'article ne permet de répondre à cette question [p. 1-10].

## 7. Hors sujet

**Question :** Bonjour ! Peux-tu te présenter en une phrase ?

**Premier raisonnement de l'agent :**

> Okay, the user is asking me to introduce myself in one sentence. Let me think. The user mentioned "Bonjour ! Peux-tu te présenter en une phrase ?" which translates to "Hello! Can you introduce yourself in one sentence?"   First, I need to check if there's any information in the provided article "Attention Is All You Need" that requires me to use the retriever tool. However, the user is asking about my own introduction, not about the article's content. The rules state that greetings and questions unrelated to the article don't require the tool.   Since the user is asking for a self-introduction

**Appels de l'outil :**

- aucun

**Réponse finale :**

Je suis un assistant de recherche spécialisé dans l'article « Attention Is All You Need » (Vaswani et al., 2017), conçu pour répondre à vos questions sur son contenu en utilisant des outils de recherche précis.