# Assistant d'entretien de Guillaume Legall

Assistant RAG qui répond aux recruteurs sur le profil de Guillaume : son CV est envoyé au
modèle à chaque question, et des ouvrages d'IA engineering indexés dans ChromaDB apportent
les notions techniques. Il garde le fil d'une conversation : chaque question est comprise
avec les tours qui la précèdent. Il s'utilise en ligne de commande ou dans une interface
web Streamlit. Le projet doit être hébergé sur Fly.io. Le README décrit
l'usage en détail ; ce fichier dit comment travailler dans le code.

## Commandes

Tout se lance depuis la racine du projet, avec `uv` (Python 3.11).

```powershell
uv run python -m unittest discover -s tests              # tests (unittest, pas pytest)
uv run cv-assistant ingest     # indexe data/pdfs et data/epubs
uv run cv-assistant ask "..."  # pose une question
uv run cv-assistant chat       # conversation : l'assistant se souvient des tours précédents
uv run cv-assistant graph      # redessine docs/graph.png
uv run streamlit run src/cv_assistant/app.py   # interface web
```

- Les tests n'appellent ni OpenAI ni ChromaDB : ils passent sans clé et sans réseau.
- `ingest`, `ask`, `chat` et `graph` appellent l'API d'OpenAI avec la clé de Guillaume. `ingest`
  y envoie le texte des ouvrages, `ask` et `chat` le CV. Ces appels sont facturés : un `ask` pour
  vérifier un changement est normal, mais ne relance pas `ingest` sans raison.
- L'interface web fait les mêmes appels : une question y vaut un `ask`, et le bouton
  « Enregistrer » avec un ouvrage y vaut un `ingest` complet.
- `graph` passe aussi par le service en ligne mermaid.ink.
- `app.py` n'a pas de test dans `tests/`. Pour vérifier qu'il s'affiche sans rien
  facturer, `streamlit.testing.v1.AppTest.from_file(...).run()` exécute le script sans
  poser de question.

## Architecture

```
src/cv_assistant/
├── cli.py           # la ligne de commande (argparse) : ingest, ask, chat, graph
├── app.py           # l'interface web (Streamlit) : dépôt des documents et conversation
├── embeddings.py    # modèle d'embedding, partagé par l'ingest et la recherche
├── vector_store.py  # ouverture de la collection ChromaDB, pour l'ingest et la recherche
├── ingestion/       # lecture des PDF et EPUB, découpage en chunks, écriture dans ChromaDB
└── rag/
    ├── graph.py     # le graphe LangGraph : nœuds + arêtes + mémoire. Point d'entrée du RAG
    ├── state.py     # RagState (ce qui circule entre les nœuds), Route et Exchange
    ├── nodes/       # un fichier par nœud, plus persona.py pour le ton commun
    ├── llm.py       # le seul fichier qui parle au modèle de langage d'OpenAI
    ├── retriever.py # le seul fichier du RAG qui interroge la collection ChromaDB
    └── cv.py        # lecture du CV
```

Le graphe :

```
START → classify ─┬─ profile ────────→ profile ───────────────────┐
                  ├─ ai_engineering ─→ retrieve → ai_engineering ─┼→ END
                  └─ off_topic ──────→ off_topic ─────────────────┘
```

Règles à respecter dans `rag/` :

- **Trois couches, dépendances dans un seul sens** : le graphe connaît les nœuds, les
  nœuds connaissent les services (`llm.py`, `retriever.py`). Un nœud n'importe jamais
  `openai` ni `chromadb`, et un service n'importe jamais un nœud.
- **Le client ChromaDB n'est créé que dans `vector_store.py`** (`open_collection`) :
  l'ingest et la recherche partagent ainsi la même collection et le même modèle
  d'embedding. Ne pas appeler `chromadb.PersistentClient` ailleurs.
- **Tous les nœuds ont la même forme** : une classe dont le constructeur reçoit ses
  services, et dont `__call__(state)` renvoie uniquement les clés qu'il ajoute à l'état.
- **Chaque nœud porte ses propres consignes** (constante `INSTRUCTIONS` dans son fichier).
  Pas de constructeur de prompt partagé ; seul le ton commun est dans `nodes/persona.py`.
- **Un nouveau nœud** = un fichier dans `nodes/`, son export dans `nodes/__init__.py`, son
  branchement dans `graph.py`, puis `graph` pour redessiner l'image et la mise à jour du
  tableau des routes dans le README.
- **Un nom de nœud ne doit pas être aussi une clé de `RagState`** : LangGraph le refuse.
- **La mémoire d'une conversation est tenue par le checkpointer** de `graph.py`
  (`InMemorySaver`, un `thread_id` par conversation) : le graphe s'exécute toujours avec
  `{"configurable": {"thread_id": ...}}`. Le nœud qui répond ajoute son tour à `history` ;
  un nœud qui appelle le modèle lui passe `state.get("history", [])`.
- **`Llm` transforme l'historique en messages** `user` / `assistant` et n'envoie que les
  `HISTORY_LIMIT` derniers tours. Les nœuds ne recopient pas l'historique dans leur message.
- **`retrieve` cherche avec `standalone_question`**, la question reformulée par `classify`,
  et non avec `question` : une question de suivi ne désigne aucun sujet à elle seule.
- **Un type du projet rangé dans `RagState`** (comme `Route`) doit être
  déclaré dans le `JsonPlusSerializer` de `graph.py`, sinon LangGraph refuse de le relire.

## Comportement attendu de l'assistant

Ces règles viennent de Guillaume et sont portées par les consignes des nœuds :

- Il s'adresse à un recruteur : vouvoiement, ton courtois, Guillaume à la troisième personne.
- Il répond dans la langue de la question, détectée par `classify`.
- Il comprend une question de suivi (« Et en production ? ») grâce aux tours précédents
  de la conversation.
- Le parcours de Guillaume vient uniquement du CV : aucune expérience inventée.
- Les ouvrages sont des livres que Guillaume a lus. Une notion qui en vient est présentée
  comme une connaissance de Guillaume, sans lui prêter une mise en pratique que le CV ne
  montre pas, et sans citer les ouvrages ni afficher de sources.
- Une question qui ne concerne ni son profil ni l'IA engineering reçoit un refus poli.
- Les coordonnées du CV (e-mail, téléphone, adresse) peuvent être données : l'accès à
  l'application sera protégé par mot de passe et réservé à de futurs employeurs.

## Pièges

- **Changer `EMBEDDING_MODEL`** impose de supprimer `data/chroma/` puis de relancer
  l'ingest : une collection ne peut pas mélanger des vecteurs de modèles différents.
- **Renommer un fichier dans `data/pdfs` ou `data/epubs`** laisse ses anciens chunks dans
  la collection (ils sont indexés par chemin) : supprimer `data/chroma/` avant de réingérer.
- **Une requête d'embedding est limitée** à 2 048 textes et 300 000 tokens, d'où l'envoi
  des chunks par lots dans `ingestion/ingest.py`.
- **`gpt-5.4-nano` est instable sur les messages d'un seul mot** (« Hola », « LangGraph ? ») :
  la route peut varier d'un appel à l'autre. Après une modification des consignes de
  `classify`, rejouer plusieurs fois quelques questions réelles, pas une seule, et
  enchaîner quelques questions de suivi dans `chat`.
- **Les conversations vivent dans la mémoire du processus** : elles sont perdues à son
  arrêt et ne sont pas partagées entre deux machines Fly.io. Les conserver demandera un
  checkpointer persistant (SQLite sur un volume, par exemple).
- **Chaque conversation doit avoir son propre `thread_id`** : deux recruteurs qui
  partageraient le même verraient leurs questions mêlées. La ligne de commande n'en
  utilise qu'un (`CONVERSATION` dans `cli.py`) ; l'interface web en crée un par session
  du navigateur (`_start_conversation` dans `app.py`).
- **Dans `app.py`, le graphe est gardé par `st.cache_resource`** : Streamlit relit le
  script à chaque interaction, et un graphe réassemblé perdrait la mémoire des
  conversations. Enregistrer un document vide ce cache, donc toutes les conversations.
- **Le volet de dépôt de `app.py` est visible par tout visiteur** : n'importe qui peut
  remplacer le CV ou déclencher un ingest facturé. À réserver à Guillaume avant la mise
  en ligne ; le mot de passe d'accès à l'application reste lui aussi à faire.
- **Ajouter un ouvrage dans l'interface réindexe tous les ouvrages** : `ingest_documents`
  traite les dossiers entiers, pas le seul fichier déposé.
- **Le fichier `.env` est lu dans le dossier courant**, comme les chemins `data/` par
  défaut : les commandes doivent être lancées depuis la racine.
- **`.env`, `data/chroma/`, les ouvrages et le CV ne sont pas dans Git.** Ne jamais les y
  ajouter : le CV contient des coordonnées personnelles. Sur Fly.io, la clé passe par
  `fly secrets` et le CV doit être fourni autrement que par le dépôt.

## Conventions de code

- Identifiants en anglais ; docstrings, commentaires, messages affichés et README en français.
- `from __future__ import annotations` en tête de chaque module, annotations de type partout.
- Les erreurs prévisibles (clé absente, collection absente, CV illisible) ont leur propre
  exception, attrapée dans `main` (`cli.py`) pour afficher un message clair et sortir avec le code 1.
  `app.py` attrape les mêmes et affiche leur message dans la page.
- Tests avec `unittest` et des faux objets écrits à la main (`FakeLlm`, `FakeRetriever`…),
  pas de bibliothèque de mock pour les services. `FakeLlm` enregistre l'historique reçu à
  chaque appel : c'est ainsi qu'on vérifie la mémoire sur deux `invoke` de même `thread_id`.
- Guillaume veut un code simple et lisible : pas d'abstraction sans besoin, pas de
  constante ou de classe intermédiaire qui n'apporte rien.
- Pour un changement d'architecture, proposer d'abord le plan et attendre sa validation
  avant de modifier.
- Garder le README à jour quand une commande, une route ou la structure change.
