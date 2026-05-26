# BUC Factory — Spécification complète

![Use Case Factory](logo_lockup.svg)

## Qu'est-ce que BUC Factory ?

BUC Factory est un système automatisé qui génère des **cas pratiques de recrutement en data**. À partir d'un simple fichier de configuration décrivant un poste, un secteur et un outil, il produit l'intégralité du matériel nécessaire à un entretien technique :

- Un **brief candidat** (l'énoncé du cas pratique)
- Des **données synthétiques** réalistes et cohérentes
- Un **projet starter** prêt à l'emploi (Power BI ou Python)
- Une **solution recruteur** détaillée avec formules, code et grille d'évaluation

Chaque génération est unique : les thèmes, volumes, contextes et pièges sont tirés aléatoirement pour éviter que deux candidats ne reçoivent le même sujet.

---

## Table des matières

1. [Vue d'ensemble du fonctionnement](#1-vue-densemble-du-fonctionnement)
2. [Configuration d'un cas](#2-configuration-dun-cas)
3. [Le pipeline de génération — les 8 étapes](#3-le-pipeline-de-génération--les-8-étapes)
4. [Les prompts et le système de templates](#4-les-prompts-et-le-système-de-templates)
5. [L'agent et ses outils](#5-lagent-et-ses-outils)
6. [Le Judge — évaluateur automatique de solutions](#6-le-judge--évaluateur-automatique-de-solutions)
7. [Le Simulateur de candidat](#7-le-simulateur-de-candidat)
8. [Le Scoring](#8-le-scoring)
9. [La soumission via l'API](#9-la-soumission-via-lapi)
10. [L'interface Streamlit](#10-linterface-streamlit)
11. [Le suivi des expériences avec MLflow](#11-le-suivi-des-expériences-avec-mlflow)
12. [La recherche sémantique](#12-la-recherche-sémantique)
13. [Les modèles de langage utilisés](#13-les-modèles-de-langage-utilisés)
14. [La structure des fichiers produits](#14-la-structure-des-fichiers-produits)
15. [Les variables d'environnement](#15-les-variables-denvironnement)
16. [Déploiement et infrastructure](#16-déploiement-et-infrastructure)
    - [16.1 Local et Docker Compose](#161-local-et-docker-compose)
    - [16.2 Cloud managé — Render.com](#162-cloud-managé--rendercom)
    - [16.3 Cloud managé — Railway.com](#163-cloud-managé--railwaycom)
    - [16.4 OVH Cloud Public](#164-ovh-cloud-public)
    - [16.5 OVH AI Deploy](#165-ovh-ai-deploy)

---

## 1. Vue d'ensemble du fonctionnement

Voici comment BUC Factory fonctionne, de bout en bout :

```
Fichier YAML (configuration du poste)
         │
         ▼
┌─────────────────────────────┐
│  Pipeline de génération     │  ← 8 étapes automatiques, pilotées par un LLM
│  (agent LangGraph)          │
└─────────────┬───────────────┘
              │
     ┌────────┴────────┐
     │                 │
     ▼                 ▼
Brief candidat    Solution recruteur
Données CSV       Projet Power BI / Notebook Python
     │
     ▼
┌─────────────────────────────┐
│  Simulateur de candidat     │  ← Simule un vrai candidat qui complète l'exercice
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Scoring automatique        │  ← Note le travail simulé par rapport à la solution
└─────────────────────────────┘
```

Tout peut être déclenché depuis la ligne de commande, l'API REST, ou l'interface web Streamlit.

---

## 2. Configuration d'un cas

Tout commence par un fichier YAML qui décrit le contexte du poste et de l'entreprise.

### Exemple de fichier de configuration

```yaml
industry: "P&C insurance"
company_context: >
  Compagnie d'assurance IARD de taille moyenne basée à Paris,
  spécialisée en assurance automobile et habitation.
  Elle gère 1,2 million de polices actives sur la France entière.
location: "Paris, France"
language: "French"
role: "Data Analyst"
seniority: "Mid-Senior"
tool: "Power BI Desktop"
duration_minutes: 75
deliverable_format: "PBIP"
```

### Description de chaque champ

| Champ | Description | Exemple |
|---|---|---|
| `industry` | Secteur d'activité | `"P&C insurance"`, `"Retail"` |
| `company_context` | Description de l'entreprise (2–4 phrases) | Voir ci-dessus |
| `location` | Ville et pays | `"Paris, France"` |
| `language` | Langue du brief et de la solution | `"French"`, `"English"` |
| `role` | Poste évalué | `"Data Analyst"`, `"Data Scientist"` |
| `seniority` | Niveau de séniorité | `"Junior"`, `"Mid-Senior"`, `"Senior"`, `"Staff"` |
| `tool` | Outil utilisé | `"Power BI Desktop"`, `"Python (Notebook)"` |
| `duration_minutes` | Durée prévue de l'exercice | `75` |
| `deliverable_format` | Format du livrable | `"PBIP"` (Power BI), `"IPYNB"` (Jupyter) |

### Dimensions et entités optionnelles

Il est possible de préciser des **dimensions** (les axes du scénario) et des **entités** (les tables de données) directement dans le YAML :

```yaml
dimensions:
  branche: ["MRH", "Auto", "Santé"]
  angle: ["Pilotage sinistralité", "Analyse portefeuille"]

entities:
  - "polices"
  - "sinistres"
  - "clients"
```

Si ces champs sont absents, le système les **infère lui-même** à l'aide du LLM lors de la première étape. Si les deux sont fournis, l'étape d'inférence est court-circuitée et le système fonctionne de façon entièrement déterministe.

---

## 3. Le pipeline de génération — les 8 étapes

La génération complète d'un cas s'articule en **8 étapes séquentielles**, certaines s'exécutant en parallèle pour gagner du temps. Chaque étape est pilotée par un LLM spécifique et validée automatiquement avant de passer à la suivante.

```
Étape 1 : Bootstrap du domaine       (Claude Sonnet)
Étape 2 : Tirage du scénario          (Claude Haiku)
           ┌──────────────┐
Étape 3 :  │  Brief       │  ← en parallèle
Étape 4 :  │  Schéma data │  ← en parallèle
           └──────────────┘
Étape 5 : Génération des données      (OpenAI o4-mini)
           ┌──────────────────────┐
Étape 6 :  │  Projet starter      │  ← en parallèle
Étape 7 :  │  Solution recruteur  │  ← en parallèle
           └──────────────────────┘
Étape 8 : Assemblage final            (Claude Haiku)
```

### Étape 1 — Bootstrap du domaine

**Rôle :** Déduire les dimensions du scénario et les entités métier adaptées au secteur.

**Ce qu'il produit :** `bootstrap.json`

```json
{
  "dimensions": {
    "branche": ["MRH", "Auto", "Santé"],
    "angle": ["Pilotage sinistralité", "Analyse portefeuille"],
    "historique_mois": ["12", "24", "36"],
    "volumetrie": ["50k_polices", "120k_polices"],
    "twist": ["Inflation pièces auto", "Fraude en hausse"],
    "restitution": ["Dashboard direction", "Rapport mensuel"]
  },
  "entities": ["polices", "sinistres", "clients", "agents", "garanties"],
  "rationale": "Ces entités correspondent aux objets centraux d'une assurance IARD..."
}
```

**Particularité :** Si `dimensions` et `entities` sont déjà fournis dans le YAML, cette étape ne fait appel à aucun LLM — elle écrit directement le fichier.

---

### Étape 2 — Tirage du scénario

**Rôle :** Choisir aléatoirement une valeur pour chaque dimension, construisant ainsi un contexte unique.

**Ce qu'il produit :** `scenario.json`

```json
{
  "branche": "Auto particuliers",
  "angle": "Pilotage de la sinistralité",
  "historique_mois": "36",
  "volumetrie": "120k_polices",
  "twist": "Inflation pièces auto",
  "restitution": "Dashboard direction technique"
}
```

Le tirage est effectué **en Python** avant d'appeler le LLM, pour que les nouvelles tentatives (en cas d'erreur) restent déterministes — le scénario ne change pas si on relance la même étape.

---

### Étape 3 — Rédaction du brief candidat

**Rôle :** Écrire l'énoncé que recevra le candidat pendant l'entretien.

**Ce qu'il produit :** `brief/candidate_brief.md`

Le brief contient :
- Le contexte de l'entreprise et du poste
- La mission confiée au candidat
- Les KPIs à calculer / visualisations à produire
- La description des données à disposition
- Les critères d'évaluation

**Contrainte de validation :** Le brief doit faire au minimum 1 500 caractères.

---

### Étape 4 — Conception du schéma de données

**Rôle :** Définir précisément la structure de toutes les tables CSV que le candidat utilisera.

**Ce qu'il produit :** `brief/data_schema.json`

Ce schéma décrit pour chaque table : les colonnes, les types, le sens métier, et surtout les **pièges** (traps) — des anomalies intentionnelles cachées dans les données pour tester la rigueur du candidat.

**Exemple de piège :**
```json
{
  "column": "montant_sinistre",
  "trap": "Les montants utilisent '.' comme séparateur décimal mais le fichier
           est encodé en locale fr-FR, ce qui peut provoquer des erreurs de type
           si le candidat n'adapte pas la requête Power Query."
}
```

**Contrainte de validation :** Au moins un piège déclaré et au moins 80 % des entités couvertes.

---

### Étape 5 — Génération du script de données

**Rôle :** Écrire et exécuter un script Python qui génère les fichiers CSV synthétiques.

**Ce qu'il produit :** `starter/generate_data.py` + tous les fichiers `starter/data/*.csv`

Le script :
- Utilise `seed=42` pour être reproductible
- Génère 300 à 800 lignes par table
- Respecte les relations entre tables (clés étrangères)
- Intègre les pièges définis à l'étape 4

**Validation automatique :** Le script est exécuté dans le pipeline. L'agent vérifie que tous les CSV ont été créés et qu'il n'y a pas d'orphelins dans les clés étrangères.

---

### Étape 6 — Génération du projet starter

**Rôle :** Créer le projet que le candidat recevra et devra compléter.

**Ce qu'il produit :** Un dossier `starter/` au format Power BI (PBIP) ou Jupyter (IPYNB)

**Pour Power BI (PBIP) :**
- Fichier `.pbip` (point d'entrée Power BI)
- Modèle sémantique TMDL v4 avec colonnes, relations et mesures partielles
- Structure de rapport avec pages et visuels

**Pour Python (IPYNB) :**
- Notebook Jupyter avec au moins 5 cellules
- Imports, chargement des données, sections à compléter

---

### Étape 7 — Rédaction de la solution recruteur

**Rôle :** Écrire la correction complète que le recruteur utilisera pour évaluer le candidat.

**Ce qu'il produit :** `solution/recruiter_solution.md`

La solution contient :
- Toutes les formules DAX (Power BI) ou tout le code Python attendu
- Les explications sur chaque piège et comment le détecter
- Une grille d'évaluation avec des niveaux de maîtrise
- Des questions d'entretien de suivi suggérées

**Contrainte de validation :** Au moins 3 000 caractères et présence de formules/calculs concrets.

---

### Étape 8 — Assemblage final

**Rôle :** Vérifier que tous les fichiers attendus sont bien présents avant de clore la génération.

**Fichiers requis :**
- `scenario.json`
- `bootstrap.json`
- `brief/candidate_brief.md`
- `brief/data_schema.json`
- `solution/recruiter_solution.md`
- `starter/generate_data.py`

---

### Parallélisme — ce qu'il apporte réellement

Deux paires d'étapes s'exécutent en parallèle via un `ThreadPoolExecutor` Python à 2 workers :

- **Groupe 1** : rédaction du brief ∥ conception du schéma de données (étapes 3 et 4)
- **Groupe 2** : génération du projet starter ∥ rédaction de la solution recruteur (étapes 6 et 7)

En théorie, exécuter deux tâches sur deux cœurs devrait réduire le temps de moitié. En pratique, **le gain est faible, voire imperceptible**. Voici pourquoi.

#### Le vrai goulot d'étranglement : l'API du LLM, pas le CPU

Chaque étape passe 95 % de son temps à **attendre la réponse du LLM** — une requête réseau vers l'API Anthropic ou OpenAI. Le processeur local ne fait presque rien pendant ce temps. Ajouter des cœurs CPU ne change donc rien à cette attente.

Quand les deux tâches parallèles appellent la même API simultanément, deux effets se produisent :

1. **Compétition sur le quota de tokens** : les APIs LLM limitent le débit en *tokens par minute* (TPM). Deux tâches simultanées consomment ce quota deux fois plus vite. Si le quota est atteint, l'une des deux tâches est mise en attente — ce qui annule le bénéfice du parallélisme.

2. **Latence côté serveur** : un modèle comme Claude Opus génère les tokens séquentiellement. Traiter deux requêtes en parallèle côté Anthropic prend presque aussi longtemps que deux requêtes séquentielles, car la capacité GPU est partagée entre les deux.

#### Pourquoi Python threads et non multiprocessing ?

Python a un mécanisme appelé le **GIL (Global Interpreter Lock)** qui empêche deux threads d'exécuter du code Python pur en même temps sur le même processus. Pour du calcul CPU intensif, les threads Python ne font donc rien en parallèle.

Ici ce n'est pas un problème, car les tâches sont *I/O-bound* (elles attendent le réseau, pas le CPU) : Python libère le GIL automatiquement pendant les opérations réseau, ce qui permet aux threads de coexister réellement. Mais cela ne change pas la contrainte côté API.

Le `multiprocessing` (vrais processus parallèles) ne résoudrait pas le problème non plus — le goulot est l'API distante, pas Python.

#### Ce que le parallélisme apporte quand même

Malgré ces limites, le parallélisme a un effet réel lorsque :
- Les deux tâches appellent des APIs **différentes** (ex. : étapes 3+4 où l'une va vers Claude Sonnet et l'autre vers GPT) — elles ne partagent pas le même quota
- Le quota TPM est largement disponible (compte avec une limite élevée) — les deux requêtes partent vraiment en même temps
- Une des deux tâches est nettement plus courte que l'autre — la plus courte finit pendant que la longue tourne encore

Dans les conditions typiques, le gain réel observé est de **15 à 30 %** sur la durée totale du pipeline, loin du 50 % théorique.

---

### Mécanisme de retry

Si une étape échoue (fichier manquant, contenu insuffisant, JSON invalide, etc.), le pipeline **relance automatiquement la même étape** en injectant dans le prompt le message d'erreur exact. Cela permet au LLM de corriger son erreur sans intervention humaine.

**Maximum de 3 tentatives** par étape. Au-delà, le pipeline s'arrête et la génération est marquée comme échouée.

---

## 4. Les prompts et le système de templates

### Comment fonctionne un prompt ?

Chaque étape reçoit un **prompt système** (qui décrit le rôle et le contexte) et un **prompt de tâche** (qui décrit ce qui est attendu à cette étape précise).

Le prompt système est construit à partir du fichier de configuration :

```
Tu es un consultant data senior spécialisé dans le secteur {industry},
basé à {location}, travaillant avec {tool}.
Tu rédiges en {language}.
Tu génères un cas pratique pour un poste de {role} niveau {seniority}.
```

### Les templates de prompts

Tous les templates se trouvent dans `src/buc_factory/conf/prompt_templates.yml`. Ils contiennent des variables comme `{industry}`, `{language}`, `{scenario_json}`, etc., qui sont remplacées dynamiquement à l'exécution.

Il existe également un fichier `prompt_templates_openai_patch.yml` qui contient des **surcharges spécifiques aux modèles OpenAI** — certaines formulations fonctionnent mieux avec GPT qu'avec Claude, et vice versa.

### Contexte pré-chargé

Pour éviter que l'agent lise les fichiers lui-même (ce qui serait lent), les contenus des étapes précédentes sont **injectés directement dans le prompt** :

- Étape 3 (brief) : reçoit `scenario.json` et `bootstrap.json`
- Étape 5 (données) : reçoit `data_schema.json` et le brief
- Étape 6 (starter) : reçoit `data_schema.json`
- Étape 7 (solution) : reçoit le brief, `scenario.json` et `data_schema.json`

### Prompt de retry

Quand une étape échoue et doit être relancée, un suffixe est ajouté au prompt :

```
FEEDBACK DE TENTATIVE PRÉCÉDENTE :
L'étape a échoué avec l'erreur suivante :
[message d'erreur exact]
Corrige ce problème dans ta prochaine tentative.
```

### Versioning des prompts dans MLflow

Le template du prompt système est enregistré dans le **Registre de Prompts MLflow** sous le nom `buc-factory-system`. Chaque modification du template crée une nouvelle version, ce qui permet de tracer quels prompts ont produit quels résultats.

---

## 5. L'agent et ses outils

### Qu'est-ce qu'un agent ?

Un **agent** est un LLM qui dispose d'outils (des fonctions qu'il peut appeler) pour accomplir une tâche. Plutôt que de simplement générer du texte, l'agent peut lire des fichiers, en écrire, exécuter du code, etc. Il choisit lui-même quels outils utiliser et dans quel ordre.

### Les outils disponibles

| Outil | Description |
|---|---|
| `write_file(chemin, contenu)` | Crée ou écrase un fichier texte |
| `read_file(chemin)` | Lit le contenu d'un fichier |
| `list_files(répertoire)` | Liste tous les fichiers d'un répertoire |
| `run_python(script)` | Exécute un script Python et retourne la sortie |
| `validate_csv_integrity(...)` | Vérifie les clés étrangères entre tables CSV |
| `validate_json(chemin)` | Vérifie qu'un fichier JSON est valide |
| `mark_subtask_complete(résumé)` | Signal de fin de tâche |

Tous les outils ne sont pas disponibles pour toutes les étapes. Par exemple, `run_python` n'est accessible qu'à l'étape de génération de données.

### La boucle agentique

L'agent fonctionne en boucle :

```
1. Appel LLM → le modèle répond avec des appels d'outils
2. Les outils sont exécutés
3. Les résultats sont renvoyés au LLM
4. → Répéter jusqu'à ce que l'agent appelle mark_subtask_complete
                                  ou dépasse 40 itérations
```

### LangGraph — l'orchestrateur

![Architecture du pipeline](factory_architecture.svg)

Le pipeline est implémenté avec **LangGraph**, un framework qui modélise le workflow comme un graphe d'états :

```
prepare_task → run_task → validate_task ──► prepare_task (retry)
                                        ├──► run_parallel_group
                                        ├──► fail_task
                                        └──► END
```

Deux groupes d'étapes s'exécutent en parallèle (via un pool de threads) :
- Étapes 3+4 : brief et schéma de données
- Étapes 6+7 : projet starter et solution recruteur

---

## 6. Le Judge — évaluateur automatique de solutions

### Qu'est-ce que le Judge ?

Après chaque génération, un **Judge** automatique évalue la qualité de la solution recruteur produite. Il vérifie que la solution répond concrètement à tous les attendus du brief.

### Comment fonctionne-t-il ?

1. Il lit le brief candidat (les attendus de l'exercice)
2. Il lit la solution recruteur générée
3. Il extrait **tous les éléments requis** (KPIs, visualisations, modèle de données, pièges, grille d'évaluation)
4. Pour chaque élément, il évalue si la solution est **concrète** (avec des formules, du code, des explications précises) ou **superficielle**
5. Il attribue un score de 1 à 10 et rédige un commentaire justificatif

### Grille de notation du Judge

| Score | Signification |
|---|---|
| 9–10 | Tous les éléments sont couverts avec des formules/code précis |
| 7–8 | Les éléments principaux sont couverts, 1–2 points restent vagues |
| 5–6 | Les éléments analytiques principaux sont présents, mais grille ou interview manquantes |
| 3–4 | Seulement une partie des éléments ; modèle ou préparation de données manquants |
| 1–2 | La majorité des éléments est absente |

### Modèle utilisé

Le Judge utilise **un modèle OpenAI GPT** (plutôt qu'un modèle Anthropic) pour éviter le biais d'auto-complaisance — un modèle Claude ne note pas le travail d'un autre modèle Claude.

Le prompt du Judge est également versionné dans le Registre de Prompts MLflow sous `buc-factory-solution-relevancy-judge`.

---

## 7. Le Simulateur de candidat

### À quoi sert le simulateur ?

Le simulateur permet de tester un cas pratique généré **en simulant un vrai candidat** qui le complète. C'est utile pour :
- Vérifier que le cas est réalisable dans le temps imparti
- Tester le scoring automatique
- Produire des exemples de rendus à différents niveaux

### Les niveaux de compétence

Le simulateur reçoit un paramètre `proficiency` entre 0.0 et 1.0 qui détermine le niveau du candidat simulé :

| Niveau | Seuil | Comportement |
|---|---|---|
| Expert | ≥ 0.90 | Complète TOUT avec précision, code propre, aucune erreur |
| Senior | ≥ 0.70 | Rate ~(1−score)% des exigences subtiles, 1–2 bugs mineurs, 1 piège raté |
| Mid-level | ≥ 0.50 | Complète ~score% des exigences, 1–6 TODO, 1–2 bugs |
| Junior-Mid | ≥ 0.35 | Nombreux TODO, plusieurs bugs, tombe dans la plupart des pièges |
| Junior | < 0.35 | Tente seulement 1–2 tâches basiques, beaucoup de placeholders |

### Deux modes de simulation

**Mode `perfect`** : le simulateur joue toujours un candidat expert (proficiency = 1.0). Utile pour valider rapidement qu'un cas est complet.

**Mode `random`** : le simulateur tire aléatoirement (ou reçoit) une valeur de proficiency. Un paramètre `seed` peut être fourni pour reproduire exactement la même simulation.

### Comment le simulateur travaille-t-il ?

Comme l'agent de génération, le simulateur est un LLM avec accès aux outils `write_file`, `read_file` et `list_files`. Il reçoit :

1. Le projet starter (le fichier Power BI ou notebook à compléter)
2. Le brief candidat
3. Des instructions comportementales adaptées à son niveau (persona + règles)

**Exemple d'instruction pour un candidat Mid-level (60%) :**
```
Tu es un data analyst de niveau intermédiaire (60% de maîtrise).
- Complète 60% des exigences du brief
- Laisse 3 sections avec un commentaire TODO
- Fais 1–2 bugs mineurs dans le code
- Tombe dans 1–2 pièges de données sans les corriger
```

Le simulateur modifie le projet starter (en écrivant directement dans les fichiers TMDL ou le notebook) pour produire un rendu réaliste.

### Validation de la simulation

Le simulateur n'est accepté que si le projet starter a été **réellement modifié de façon substantielle**. Une tentative qui ne change presque rien est rejetée et relancée (jusqu'à 3 fois).

---

## 8. Le Scoring

### À quoi sert le scoring ?

Le scoring évalue automatiquement le travail d'un candidat par rapport à la solution recruteur. Il fonctionne dans **deux cas d'usage distincts**, mais repose sur la même fonction `score_submission()` :

1. **Après une simulation** — le scoring est déclenché automatiquement une fois que le simulateur a complété le projet starter
2. **Sur une soumission réelle** — un recruteur téléverse le ZIP rendu par un vrai candidat, et le système le note immédiatement

### La fonction de scoring

La fonction `score_submission(output_dir, recruiter_solution, deliverable_format)` :

1. Lit `brief/candidate_brief.md` pour extraire les exigences
2. Collecte le travail du candidat selon le format :
   - **PBIP** : lit tous les fichiers `.tmdl` du modèle sémantique + tous les `.json` des pages du rapport
   - **IPYNB** : lit directement `notebook.ipynb`
3. Lit la solution recruteur (fournie en texte)
4. Tronque chaque section à **12 000 caractères** maximum pour tenir dans la fenêtre de contexte
5. Appelle Claude Opus avec un prompt qui compare le travail aux exigences, en s'appuyant sur la solution comme référence
6. Retourne un rapport markdown structuré

### Ce que le rapport contient

- Un résumé de la note globale
- Pour chaque exigence du brief : couverte, partiellement couverte, ou absente
- Les pièges détectés ou ratés par le candidat
- Des suggestions pour l'entretien de débriefing

### Modèle utilisé

**Claude Opus 4.7** (le modèle le plus puissant), avec bascule automatique sur **GPT-5.5** (OpenAI) si l'API Anthropic est indisponible.

### Soumission d'une solution réelle par un candidat

Au-delà des simulations automatiques, un recruteur peut soumettre la solution rendue par un vrai candidat pour la faire noter :

```http
POST /runs/{run_id}/score
Content-Type: multipart/form-data

solution: <fichier .zip>
```

Le ZIP doit contenir le projet complété par le candidat — soit la structure Power BI (PBIP), soit le notebook Python (IPYNB). Un ZIP enveloppé dans un dossier racine unique est automatiquement dézippé.

**Traitement :**
1. Validation que le fichier est bien un ZIP valide
2. Récupération du format attendu depuis MLflow (paramètre `deliverable_format` du run)
3. Extraction du ZIP dans un répertoire temporaire `starter/`
4. Téléchargement du brief et de la solution recruteur depuis MLflow
5. Appel de `score_submission()` avec ces fichiers
6. Retour du rapport de scoring en markdown (`200 OK`, `text/plain`)

Cette fonctionnalité permet d'évaluer des candidats réels avec le même niveau de rigueur qu'une simulation automatique.

---

## 9. La soumission via l'API

### Vue d'ensemble

BUC Factory expose une **API REST** (FastAPI) sur le port 8000 qui permet de soumettre des demandes de génération, de les suivre, et de récupérer les résultats.

**Documentation interactive :** `http://localhost:8000/docs`

### Soumettre une génération

```http
POST /runs
Content-Type: application/json

{
  "industry": "P&C insurance",
  "company_context": "Compagnie d'assurance IARD...",
  "location": "Paris, France",
  "language": "French",
  "role": "Data Analyst",
  "seniority": "Mid-Senior",
  "tool": "Power BI Desktop",
  "duration_minutes": 75,
  "deliverable_format": "PBIP"
}
```

**Réponse immédiate (202 Accepted) :**
```json
{
  "run_id": "run_042",
  "status": "queued"
}
```

La génération tourne **en arrière-plan** — l'API répond immédiatement sans attendre la fin du pipeline.

### Suivre l'avancement

```http
GET /runs/run_042
```

```json
{
  "run_id": "run_042",
  "status": "running",
  "scenario": {
    "branche": "Auto particuliers",
    "angle": "Pilotage de la sinistralité"
  }
}
```

Les statuts possibles sont : `queued`, `running`, `done`, `failed`.

### Récupérer les livrables

| Endpoint | Contenu |
|---|---|
| `GET /runs/{id}/brief` | Brief candidat (markdown) |
| `GET /runs/{id}/solution` | Solution recruteur (markdown) |
| `GET /runs/{id}/candidate.zip` | Brief + projet starter (sans la solution) |
| `GET /runs/{id}/recruiter.zip` | Brief + solution complète |
| `POST /runs/{id}/score` | Soumettre le ZIP d'un candidat réel pour notation |

### Soumettre une simulation

```http
POST /simulations
Content-Type: application/json

{
  "run_id": "run_042",
  "mode": "random",
  "proficiency": 0.65,
  "seed": 42
}
```

**Réponse :**
```json
{
  "simulation_id": "sim_007",
  "status": "queued"
}
```

### Récupérer le résultat d'une simulation

| Endpoint | Contenu |
|---|---|
| `GET /simulations/{id}` | Statut, mode, proficiency |
| `GET /simulations/{id}/scoring` | Rapport de scoring (markdown) |
| `GET /simulations/{id}/solution.zip` | Projet starter complété |

### Rechercher dans les générations passées

```http
GET /runs/search?q=sinistralité+auto+Power+BI&limit=5
```

La recherche utilise les embeddings sémantiques — elle comprend le sens de la requête, pas juste les mots-clés exacts. Le paramètre `use_hyde=true` (défaut) améliore la précision pour les requêtes courtes ou ambiguës.

---

## 10. L'interface Streamlit

L'interface web est accessible sur le port 8501 (`http://localhost:8501`).

### Ce qu'elle permet

**Soumettre un cas** : formulaire avec tous les champs de configuration, valeurs par défaut pré-remplies, bouton de randomisation pour explorer rapidement des profils variés (15+ secteurs et pays préconfigurés).

**Naviguer dans les cas existants** : liste de toutes les générations passées avec statut en temps réel (couleur par statut), possibilité de télécharger les archives ZIP.

**Voir le détail d'un cas** : paramètres, scénario tiré, statut de chaque étape.

**Lancer une simulation** : choisir un cas existant, un mode et un niveau, et suivre l'avancement.

**Chercher** : barre de recherche sémantique sur tous les cas générés.

---

## 11. Le suivi des expériences avec MLflow

MLflow est utilisé pour tracer **chaque génération et simulation** dans un journal d'expériences.

### Ce qui est enregistré pour chaque run

**Paramètres** (ce qu'on a demandé) :
- industry, role, tool, language, seniority, location, duration_minutes, deliverable_format

**Métriques** (ce qui s'est passé) :
- Durée de chaque étape (en secondes)
- Nombre de tokens consommés par étape (entrée + sortie)
- Coût estimé par étape (en USD)
- Nombre de tentatives par étape
- Score de pertinence de la solution (1–10)

**Artefacts** (les fichiers produits) :
- Tous les fichiers du cas généré
- Résumé des tâches (`task_summary.md`)
- Prompts utilisés pour chaque étape (`prompts/`)

**Tags** :
- Identifiant API du run (`api_run_id`)
- Justification du score (`solution_relevancy_reasoning`)

### Les deux expériences MLflow

| Expérience | Contenu |
|---|---|
| `buc-factory` | Toutes les générations de cas |
| `buc-factory-simulations` | Toutes les simulations de candidats |

### Calcul du coût

Le coût est estimé automatiquement à partir du nombre de tokens et des tarifs officiels des modèles. Exemple :

```
Claude Opus 4.7 :  $5.00 / million tokens en entrée, $25.00 en sortie
Claude Sonnet 4.6 : $3.00 / million tokens en entrée, $15.00 en sortie
Claude Haiku 4.5 : $1.00 / million tokens en entrée, $5.00 en sortie
```

---

## 12. La recherche sémantique

### Pourquoi une recherche sémantique ?

La recherche par mots-clés classique (LIKE SQL) ne retrouve que des correspondances exactes. La recherche sémantique comprend le **sens** de la requête : "sinistres auto France" retrouvera un cas intitulé "gestion de la sinistralité automobile IARD en France", même si aucun mot ne correspond exactement.

### Architecture

```
Requête utilisateur
      │
      ▼ (si use_hyde=True, activé par défaut)
GPT-4o-mini génère une description fictive d'un run correspondant
      │
      ▼
text-embedding-3-small → vecteur de 1536 dimensions
      │
      ▼
Cosine similarity contre tous les vecteurs stockés dans SQLite
      │
      ▼
Top-k résultats classés par score
```

### Étape 1 — Construction du texte d'index

Lors de chaque génération réussie, un **texte d'index** est construit et stocké avec son vecteur :

```
"A Mid-Senior Data Analyst working in the P&C insurance industry,
using Power BI Desktop, based in Paris, speaking French.
Scenario: branche: Auto particuliers, angle: Pilotage de la sinistralité,
historique_mois: 36, volumetrie: 120k_polices, twist: Inflation pièces auto"
```

Ce texte combine les paramètres du run (rôle, secteur, outil, lieu, langue) et le scénario tiré au sort. C'est ce texte qui est converti en vecteur et stocké.

### Étape 2 — Embedding du texte d'index

Le texte est envoyé au modèle **OpenAI `text-embedding-3-small`** (1 536 dimensions). Le vecteur résultant est sérialisé en JSON et stocké dans SQLite :

```sql
CREATE TABLE run_embeddings (
    api_run_id    TEXT PRIMARY KEY,
    mlflow_run_id TEXT,
    embed_text    TEXT,       -- texte d'index lisible
    embedding     TEXT        -- vecteur JSON [0.023, -0.41, ...]
)
```

### Étape 3 — HyDE (Hypothetical Document Embeddings)

La difficulté de la recherche vectorielle est que la requête utilisateur ("data analyst assurance auto") et le texte indexé ("A Mid-Senior Data Analyst working in the P&C insurance industry...") ne sont **pas dans le même espace sémantique** — l'un est une requête courte, l'autre est une description de run.

HyDE résout ce problème en deux temps :

**1. Génération d'un document hypothétique**

La requête brute est envoyée à GPT-4o-mini avec ce prompt :
```
"Write one sentence describing a BUC coaching run for: {query}.
Mention role, seniority, industry, tool, location, and language where relevant."
```

Exemple — requête `"data analyst assurance auto power bi"` → description hypothétique :
```
"A Mid-Senior Data Analyst in the automotive insurance industry
using Power BI Desktop, based in France, speaking French."
```

**2. Embedding du document hypothétique**

C'est cette description (plus riche et dans le même format que les textes indexés) qui est convertie en vecteur, pas la requête brute. Cela aligne les espaces vectoriels et améliore drastiquement la pertinence pour les requêtes courtes ou ambiguës.

Pour désactiver HyDE et interroger directement sur la requête brute :
```
GET /runs/search?q=data+analyst+assurance&use_hyde=false
```

### Étape 4 — Similarité cosine

La recherche charge tous les vecteurs stockés, puis calcule la **similarité cosine** entre le vecteur de la requête et chaque vecteur indexé :

```
score = (vecteur_requête · vecteur_document) / (|vecteur_requête| × |vecteur_document|)
```

En pratique, les deux vecteurs sont normalisés (division par leur norme L2), ce qui réduit le calcul à un simple produit scalaire. NumPy est utilisé pour vectoriser ce calcul sur l'ensemble de la base en une seule opération matricielle.

Un score de 1.0 signifie une correspondance parfaite, 0.0 aucune corrélation. En pratique, les scores pertinents se situent entre 0.7 et 0.95.

### Endpoint de recherche

```http
GET /runs/search?q=data+scientist+retail+python&limit=5&use_hyde=true
```

**Paramètres :**

| Paramètre | Défaut | Description |
|---|---|---|
| `q` | obligatoire | Requête en texte libre |
| `limit` | 10 | Nombre de résultats (max 100) |
| `use_hyde` | `true` | Activer l'expansion HyDE |

**Réponse :** liste de `SearchResult` triés par score décroissant, chacun contenant : `score` (float), `run_id`, `status`, `parameters`, `scenario`.

### Backfill — indexer les runs existants

Les runs générés avant l'ajout de la recherche ne sont pas indexés. La commande suivante les indexe rétrospectivement :

```bash
uv run buc-factory-backfill
```

**Processus :**
1. Récupère tous les runs `FINISHED` dans MLflow ayant un tag `api_run_id`
2. Filtre ceux qui ne sont pas encore dans la table `run_embeddings`
3. Pour chaque run manquant : télécharge `scenario.json` depuis MLflow, construit le texte d'index, calcule l'embedding, stocke dans SQLite
4. Affiche un résumé : X indexés, Y déjà présents

Nécessite `MLFLOW_TRACKING_URI` et `OPENAI_API_KEY`.

---

## 13. Les modèles de langage utilisés

BUC Factory utilise **plusieurs modèles** selon les étapes, en choisissant le meilleur rapport qualité/coût pour chaque tâche.

| Étape | Modèle principal | Raison |
|---|---|---|
| Bootstrap du domaine | Claude Sonnet | Bonne compréhension métier, bon rapport qualité/coût |
| Tirage du scénario | Claude Haiku | Tâche simple, rapide et peu coûteuse |
| Brief candidat | Claude Sonnet | Rédaction fluide en plusieurs langues |
| Schéma de données | GPT (OpenAI) | Conception structurée de schémas |
| Script de données | o4-mini (OpenAI) | Génération de code Python robuste |
| Projet starter | Claude Opus | Tâche complexe, fichiers longs, haute qualité requise |
| Solution recruteur | Claude Opus | Contenu long et dense, formules précises requises |
| Assemblage final | Claude Haiku | Vérification simple, peu coûteuse |
| Scoring | Claude Opus | Évaluation nuancée et détaillée |
| Judge (évaluation) | GPT (OpenAI) | Biais croisé : pas de modèle Anthropic qui s'auto-évalue |

### Bascule automatique Anthropic → OpenAI

Si l'API Anthropic devient indisponible (quota dépassé, problème de facturation, clé manquante), **tous les appels Claude basculent automatiquement** sur l'équivalent OpenAI sans interruption du pipeline :

| Claude | OpenAI de substitution |
|---|---|
| Claude Haiku | GPT mini |
| Claude Sonnet | GPT standard |
| Claude Opus | GPT premium |

---

## 14. La structure des fichiers produits

Voici l'arborescence complète d'une génération réussie :

```
output-dir/
├── bootstrap.json          ← dimensions et entités du domaine
├── scenario.json           ← scénario tiré au sort
├── state.json              ← point de reprise (checkpoint)
│
├── brief/
│   ├── candidate_brief.md  ← énoncé à remettre au candidat
│   └── data_schema.json    ← structure des tables + pièges
│
├── starter/
│   ├── generate_data.py    ← script de génération des données (seed=42)
│   ├── data/
│   │   ├── polices.csv
│   │   ├── sinistres.csv
│   │   └── clients.csv
│   │
│   │   ── Si Power BI (PBIP) ──
│   ├── Assessment.pbip
│   ├── Assessment.SemanticModel/
│   │   └── definition/
│   │       ├── model.tmdl
│   │       └── tables/
│   │           ├── polices.tmdl
│   │           └── sinistres.tmdl
│   └── Assessment.Report/
│       └── pages/...
│
│   │   ── Si Python (IPYNB) ──
│   ├── notebook.ipynb
│   └── requirements.txt
│
└── solution/
    └── recruiter_solution.md   ← correction complète + grille d'évaluation
```

### Le checkpoint (`state.json`)

À chaque étape validée, un fichier `state.json` est mis à jour. Si la génération est interrompue (crash, timeout), on peut la **reprendre là où elle s'est arrêtée** en relançant avec le même répertoire de sortie :

```bash
python -m buc_factory --config conf/... --output-dir data/run_001
# Si state.json existe déjà dans data/run_001, la génération reprend à l'étape suivante
```

---

## 15. Les variables d'environnement

À configurer dans un fichier `.env` à la racine du projet.

### Obligatoires

```bash
ANTHROPIC_API_KEY=sk-ant-...   # Clé API Anthropic (pour Claude)
OPENAI_API_KEY=sk-...          # Clé API OpenAI (pour GPT et embeddings)
```

### Optionnelles

```bash
# Serveur MLflow (par défaut : SQLite locale)
MLFLOW_TRACKING_URI=http://mlflow:5000

# Chemin de la base de recherche sémantique
EMBEDDINGS_DB_PATH=data/embeddings.db

# Stockage cloud pour les artefacts MLflow
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
MLFLOW_S3_ENDPOINT_URL=...
ARTIFACT_BUCKET=...
```

---

## 16. Déploiement et infrastructure

### Deux façons de déployer

BUC Factory peut être déployé de deux manières différentes selon le contexte :

**A — Depuis le dépôt GitHub (build à la volée)**
- La plateforme clone le repo et construit l'image Docker elle-même
- Avantage : simple à configurer, pas besoin de registry externe
- Inconvénient : le build prend 3–5 minutes à chaque déploiement, et le repo doit être accessible depuis la plateforme
- Prérequis : accès au dépôt GitHub (public ou autorisation accordée à la plateforme)

**B — Depuis l'image GHCR pré-construite**
- GitHub Actions construit et publie des images multi-arch (`amd64` + `arm64`) dans le GitHub Container Registry à chaque push sur `main` ou tag de version
- Image : `ghcr.io/allantamdem/buc_factory:latest` (ou un tag précis comme `ghcr.io/allantamdem/buc_factory:v1.2.3`)
- Avantage : déploiement instantané, image versionnée et testée par la CI, aucun accès au code source requis
- Prérequis : Docker ≥ 24, les fichiers `compose.yaml` et `.env.example` (téléchargeables sans cloner le repo)

---

### 16.1 Local et Docker Compose

**Prérequis :**
- Python ≥ 3.12 + [uv](https://docs.astral.sh/uv/) (développement)
- Docker Desktop (Docker Compose)
- `ANTHROPIC_API_KEY` et `OPENAI_API_KEY`

**Développement :**
```bash
uv sync
cp .env.example .env   # renseigner les clés API
uv run python -m buc_factory --config conf/industry_spec/p_and_c_france.yml --output-dir data/run_001
uv run buc-factory-api   # port 8000
uv run buc-factory-ui    # port 8501
```

**Docker Compose (méthode recommandée en local) :**
```bash
make build
make up
```

| Service | Port | Rôle |
|---|---|---|
| `mlflow` | 5001 | Suivi d'expériences + artefacts |
| `api` | 8000 | API REST FastAPI |
| `ui` | 8501 | Interface Streamlit |

**Depuis l'image GHCR (sans cloner le repo) :**
```bash
curl -LO https://raw.githubusercontent.com/AllanTamdem/buc_factory/main/compose.yaml
curl -LO https://raw.githubusercontent.com/AllanTamdem/buc_factory/main/.env.example
cp .env.example .env   # renseigner les clés API
mkdir -p log data/mlflow/artifacts
docker compose up mlflow api ui -d
```

L'image `ghcr.io/allantamdem/buc_factory:latest` est téléchargée automatiquement au premier démarrage.

---

### 16.2 Cloud managé — Render.com

Render simplifie le déploiement sans gérer de serveur. L'architecture minimale nécessite **deux services** : l'API et MLflow.

#### Prérequis
- Compte Render (plan Starter à $7/mois minimum pour les disques persistants)
- `ANTHROPIC_API_KEY` et `OPENAI_API_KEY`
- Accès au dépôt GitHub **ou** l'image GHCR (`ghcr.io/allantamdem/buc_factory:latest`)

#### Service 1 — MLflow

MLflow stocke les expériences et les artefacts. Il doit être déployé en premier car l'API en dépend.

**Depuis l'image pré-construite :**
1. New → Web Service → **Deploy an existing image**
2. Image : `ghcr.io/mlflow/mlflow:v3.12.0`
3. **Docker Command** :
   ```
   mlflow server --host 0.0.0.0 --port $PORT --backend-store-uri sqlite:////mlflow/mlflow.db --default-artifact-root /mlflow/artifacts --serve-artifacts --artifacts-destination /mlflow/artifacts
   ```
4. Ajouter un **Disk** monté sur `/mlflow` (512 MB minimum) — sans disque persistant, toutes les données sont perdues à chaque redéploiement
5. Utiliser `$PORT` et non un port fixe — Render impose sa propre valeur via la variable d'environnement `PORT`

Une fois déployé, noter l'URL publique du service (ex. `https://mlflow-xxxx.onrender.com`).

**Limitations Render pour MLflow :**
- Le plan Starter (512 MB RAM) est insuffisant pour MLflow v3 — utiliser le plan **Standard (2 GB RAM, ~$25/mois)**
- Render ne supporte pas les disques persistants sur le plan gratuit

#### Service 2 — API buc-factory

**Depuis le repo GitHub :**
1. New → Web Service → **Connect a repository** → sélectionner le repo `buc_factory`
2. Runtime : **Docker** (Render détecte le `Dockerfile` automatiquement)
3. Variables d'environnement à configurer dans le dashboard :

| Variable | Valeur |
|---|---|
| `ANTHROPIC_API_KEY` | `sk-ant-...` |
| `OPENAI_API_KEY` | `sk-...` |
| `MLFLOW_TRACKING_URI` | `https://mlflow-xxxx.onrender.com` |
| `EMBEDDINGS_DB_PATH` | `/app/data/embeddings.db` |

4. Ajouter un **Disk** monté sur `/app/data` pour persister la base d'embeddings et les artefacts locaux

**Depuis l'image GHCR :**
1. New → Web Service → **Deploy an existing image**
2. Image : `ghcr.io/allantamdem/buc_factory:latest`
3. Configurer les mêmes variables d'environnement

#### Différences clés entre les deux modes sur Render

| | Depuis le repo | Depuis l'image GHCR |
|---|---|---|
| Temps de déploiement | 3–5 min (build inclus) | <1 min |
| Accès au code requis | Oui | Non |
| Mise à jour | Automatique à chaque push | Manuelle (redéploiement avec nouveau tag) |
| Contrôle de version | Branch/commit | Tag d'image |

---

### 16.3 Cloud managé — Railway.com

Railway offre une expérience similaire à Render, avec une tarification à l'usage plutôt que par plan fixe. Particulièrement adapté pour MLflow grâce à une RAM plus généreuse par défaut.

#### Prérequis
- Compte Railway (plan Hobby à $5/mois, inclut $5 de crédits d'usage)
- `ANTHROPIC_API_KEY` et `OPENAI_API_KEY`
- Accès au dépôt GitHub **ou** l'image GHCR (`ghcr.io/allantamdem/buc_factory:latest`)

#### Service 1 — MLflow sur Railway

1. New Project → **Deploy from Docker image**
2. Image : `ghcr.io/mlflow/mlflow:v3.12.0`
3. **Start command** :
   ```
   mlflow server --host 0.0.0.0 --port $PORT --backend-store-uri sqlite:////mlflow/mlflow.db --default-artifact-root /mlflow/artifacts --serve-artifacts --artifacts-destination /mlflow/artifacts
   ```
4. Ajouter un **Volume** monté sur `/mlflow` pour la persistance
5. Railway alloue de la RAM dynamiquement — MLflow démarre sans problème sur le plan Hobby

#### Service 2 — API buc-factory sur Railway

**Depuis le repo GitHub :**
1. New Project → **Deploy from GitHub repo** → sélectionner `buc_factory`
2. Railway détecte automatiquement le `Dockerfile`
3. Configurer les variables d'environnement dans Settings → Variables

**Depuis l'image GHCR :**
1. New Project → **Deploy from Docker image**
2. Image : `ghcr.io/allantamdem/buc_factory:latest`

#### Avantages de Railway vs Render

- RAM plus généreuse par défaut (512 MB à 8 GB selon l'usage)
- Scaling automatique à zéro (aucun coût si pas de trafic)
- Provisioning PostgreSQL natif (alternative à SQLite pour MLflow en production)
- Réseau privé entre services dans le même projet (pas besoin d'exposer MLflow publiquement)

#### Utiliser PostgreSQL pour MLflow sur Railway

Railway peut provisionner une base PostgreSQL en un clic. Pour l'utiliser avec MLflow :

1. New → **PostgreSQL** dans le même projet Railway
2. Récupérer la variable `DATABASE_URL` générée automatiquement
3. Modifier la commande MLflow :
   ```
   mlflow server --host 0.0.0.0 --port $PORT \
     --backend-store-uri $DATABASE_URL \
     --default-artifact-root /mlflow/artifacts \
     --serve-artifacts
   ```

Cela évite la limitation de SQLite (un seul writer à la fois) pour les environnements avec plusieurs runs concurrents.

---

### 16.4 OVH Cloud Public

OVH Cloud Public est l'offre IaaS d'OVH : des instances de calcul (CPU/RAM), du stockage et du réseau, sans abstraction managed. C'est le mode de déploiement **le plus flexible mais aussi le plus manuel**.

#### Prérequis
- Compte OVH Cloud Public
- Instance **B2-7** minimum (2 vCPU, 7 GB RAM) pour faire tourner MLflow + API ensemble
- Instance **B2-15** recommandée (4 vCPU, 15 GB RAM) pour la production
- Adresse IP publique associée à l'instance
- Docker et Docker Compose installés sur l'instance
- Nom de domaine (optionnel, mais recommandé pour HTTPS)

#### Déploiement depuis l'image GHCR (recommandé)

C'est la méthode la plus propre : elle n'expose pas le code source sur le serveur et déploie une version construite et testée par la CI.

```bash
# Sur le serveur OVH
mkdir -p /opt/buc-factory && cd /opt/buc-factory

# Télécharger les fichiers du stack (sans cloner le repo)
curl -LO https://raw.githubusercontent.com/AllanTamdem/buc_factory/main/compose.yaml
curl -LO https://raw.githubusercontent.com/AllanTamdem/buc_factory/main/.env.example

# Configurer les variables d'environnement
cp .env.example .env
nano .env   # renseigner ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.

# Créer les répertoires de volumes et démarrer
mkdir -p log data/mlflow/artifacts
docker compose up mlflow api ui -d
```

L'image `ghcr.io/allantamdem/buc_factory:latest` est téléchargée automatiquement au premier démarrage. MLflow est tiré depuis `ghcr.io/mlflow/mlflow`.

#### Déploiement depuis le repo GitHub

```bash
# Sur le serveur OVH
git clone https://github.com/<org>/buc_factory.git /opt/buc-factory
cd /opt/buc-factory
cp .env.example .env
nano .env   # renseigner les clés API

docker compose build
docker compose up -d
```

#### Stockage persistant sur OVH

Pour éviter de perdre les données MLflow et les embeddings entre redémarrages, monter un **volume OVH Block Storage** :

1. Créer un volume dans la console OVH → Block Storage → Attacher à l'instance
2. Formater et monter le volume :
   ```bash
   mkfs.ext4 /dev/sdb
   mount /dev/sdb /mnt/buc-data
   echo "/dev/sdb /mnt/buc-data ext4 defaults 0 2" >> /etc/fstab
   ```
3. Adapter les volumes dans `compose.yml` pour pointer vers `/mnt/buc-data`

#### HTTPS avec Nginx et Let's Encrypt

```bash
apt install nginx certbot python3-certbot-nginx
certbot --nginx -d buc-factory.mondomaine.com
```

Configurer Nginx comme reverse proxy vers `localhost:8000` (API) et `localhost:8501` (UI).

---

### 16.5 OVH AI Deploy

**OVH AI Deploy** est la plateforme serverless d'OVH pour déployer des applications conteneurisées avec accélération GPU optionnelle. Elle est bien adaptée à BUC Factory car elle gère automatiquement le scaling et le HTTPS.

#### Différences avec OVH Cloud Public

| | OVH Cloud Public | OVH AI Deploy |
|---|---|---|
| Gestion du serveur | Manuelle (SSH, Docker) | Aucune |
| Scaling | Manuel | Automatique |
| HTTPS | À configurer (Nginx) | Inclus |
| GPU disponible | Non (instances CPU) | Oui (NVIDIA T4, A100) |
| Tarification | Instance fixe ($/heure) | À l'usage (par heure de run) |
| Persistance des données | Volume Block Storage | Object Storage (Swift/S3) |

#### Prérequis
- Compte OVH Cloud avec accès à AI Deploy
- Image Docker publiée dans un registry (OVH Managed Registry, GitHub Container Registry, Docker Hub)
- `ANTHROPIC_API_KEY` et `OPENAI_API_KEY`
- Bucket OVH Object Storage pour les artefacts MLflow (remplace le volume local)

#### Publier l'image dans OVH Managed Registry

```bash
# Construire et taguer l'image
docker build -t <region>.registry.ovh.net/<namespace>/buc-factory:latest .

# S'authentifier et pousser
docker login <region>.registry.ovh.net
docker push <region>.registry.ovh.net/<namespace>/buc-factory:latest
```

Alternativement, l'image pré-construite par la CI peut être poussée directement depuis GitHub Actions vers OVH Managed Registry.

#### Déployer l'API sur AI Deploy

Via la console OVH ou la CLI `ovhai` :

```bash
ovhai app run \
  --name buc-factory-api \
  --image <region>.registry.ovh.net/<namespace>/buc-factory:latest \
  --cpu 4 \
  --memory 8Gi \
  --env ANTHROPIC_API_KEY=sk-ant-... \
  --env OPENAI_API_KEY=sk-... \
  --env MLFLOW_TRACKING_URI=https://mlflow.mondomaine.com \
  --env MLFLOW_S3_ENDPOINT_URL=https://s3.<region>.io.cloud.ovh.net \
  --env AWS_ACCESS_KEY_ID=<ovh-s3-key> \
  --env AWS_SECRET_ACCESS_KEY=<ovh-s3-secret> \
  --env ARTIFACT_BUCKET=buc-factory-artifacts \
  --port 8000
```

#### MLflow avec Object Storage (S3 OVH)

Sur AI Deploy, il n'y a pas de disque persistant attachable directement. MLflow doit utiliser :
- **Backend store** : PostgreSQL managé OVH (Cloud Databases → PostgreSQL)
- **Artifact store** : OVH Object Storage (compatible S3)

Configuration MLflow pour ce mode :
```bash
mlflow server \
  --host 0.0.0.0 \
  --port $PORT \
  --backend-store-uri postgresql://user:pass@host:5432/mlflow \
  --default-artifact-root s3://buc-factory-artifacts/ \
  --serve-artifacts
```

#### Tableau récapitulatif des options de déploiement

| Plateforme | Effort setup | Coût estimé/mois | RAM disponible | Persistance | HTTPS |
|---|---|---|---|---|---|
| Local / Docker Compose | Minimal | $0 | Illimitée | Volume local | Non |
| Render.com | Faible | $25–50 | 2 GB (Standard) | Disk ($1/GB) | Oui |
| Railway.com | Faible | $10–30 | Dynamique | Volume inclus | Oui |
| OVH Cloud Public | Moyen | $15–40 | 7–15 GB | Block Storage | Manuel |
| OVH AI Deploy | Moyen | Variable (usage) | 4–32 GB | Object Storage | Oui |

### CI/CD

Le workflow GitHub Actions `.github/workflows/package.yml` s'exécute en deux jobs séquentiels :

**Job `ci`** (déclenché à chaque push et PR) :
1. Lint et vérification de format avec `ruff`
2. Vérification des types avec `mypy`
3. Exécution de la suite de tests avec `pytest`

**Job `package`** (déclenché après la réussite du job `ci`) :
1. Construction d'une image Docker multi-arch (`linux/amd64` + `linux/arm64`) via QEMU + Buildx
2. Publication dans le GitHub Container Registry (`ghcr.io/allantamdem/buc_factory`) à chaque merge sur `main` ou tag de version
3. Tags générés : `latest` (branche main), semver (`v1.2.3`, `1.2`), et SHA court (chaque build)
4. Création d'une release GitHub avec notes auto-générées sur les tags `v*`
