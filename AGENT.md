# AGENT — Consignes permanentes pour Copilot (Projet StockGPT)

## 0) Rôle et mission
Tu es l’agent d’ingénierie principal du projet **StockGPT**.  
Objectif : implémenter, tester et documenter un **modèle transformeur autoregressif** entraîné **directement sur les rendements journaliers d’actions**, reproduisant fidèlement le protocole du papier de référence **“StockGPT: A GenAI Model for Stock Prediction and Trading”** (PDF `docs/ssrn-4787199.pdf`) et ses résultats de backtest (déciles long/short). Les décisions de conception **doivent** se conformer aux spécifications ci‑dessous et **aux sections citées** du PDF.

> Référence structurante (source of truth) : `docs/ssrn-4787199.pdf` — Dat Mai (2025).  
> – Discrétisation en **50 bps**, **402 tokens**, fermeture **à droite**, cap ±100 % ; exemple d’encodage (−2.4 %, 0 %, 0 %, 5 %, 4.8 %) → (196, 200, 200, 210, 210).   
> – Architecture **GPT légère** : séquence **256**, embeddings **128**, **4** blocs, **4** têtes, **dropout 0.2**, ≈ **0.93 M** de paramètres.   
> – Données/partitions : CRSP 1926–2023 ; **train 1926–2000**, validation 1991–2000 (hyperparamètres), **test 2001–2023** ; échantillonnage proportionnel à la longueur d’historique par action.   
> – Inférence : distribution sur 402 bins → **E[r]** via moyenne pondérée des midpoints ; multi‑horizon par échantillonnage récursif (optionnel).   
> – Portefeuille : au *close* de t, **long top 10 % / short bottom 10 %** (EW), filtre **cap p10** (et prix min optionnel \$1/\$3/\$5), rééquilibrage quotidien. Référence d’ordre de grandeur : **119 % annuels, Sharpe ≈ 6.5** en EW.   
> – Évaluation Fama–MacBeth : rang‑transformer les signaux dans **[−0.5, 0.5]**, pente moyenne **élevée** (ex. ~158 % annualisé pour t+1) et **t‑stat significatif** ; Newey‑West (20). 

---

## 1) Invariants du papier (non‑négociables sauf Déviation **assumée**) 

1. **Pas de fuite d’information** (*no look‑ahead*) : toutes les features dynamiques doivent être **connues à t** pour prédire t+1. Les partitions temporelles train/val/test doivent respecter strictement l’axe du temps (point‑in‑time).   
2. **Tokenisation des rendements** (module `tokens/`) :  
   - Conversion en **basis points**, troncature, **cap ±10 000 bps**.  
   - **Bins de 50 bps**, **fermés à droite**, **402 tokens** (0–401).  
   - Représentation par **midpoints** (sauf bins extrêmes à ±10 000 bps).  
   - Tests unitaires incluant l’exemple (−2.4 %, 0 %, 0 %, 5 %, 4.8 %) → (196, 200, 200, 210, 210). 
3. **Architecture du modèle** (module `model/`) :  
   - Décodeur **transformer** causal, **séq=256**, **d_model=128**, **n_layers=4**, **n_heads=4**, **dropout=0.2**, **softmax** sur 402 classes.  
   - Masque **triangulaire causal** ; LayerNorm (pre‑norm) ; FFN ×4 (ReLU/GeLU).  
   - Nombre de paramètres cible **≈ 0.93 M** (contrôle d’ordre de grandeur). 
4. **Entraînement** (module `train/`) :  
   - Perte **cross‑entropy** (classification 402).  
   - **Batch=64**, **10 000 steps** (par défaut) ; AMP (bf16) + AdamW + scheduler cosinus (warmup).  
   - Échantillonnage de séquences **aléatoires** (longueur 256), **proba ∝** nb d’observations par titre.  
   - Suivi de la **perte validation** ; repère : CE qui **baisse fortement** et se **stabilise** (ex. ≈ 2.5 vers 5 000 steps dans l’étude). 
5. **Inférence** (module `infer/`) :  
   - Pour chaque action et jour t (test), prédire la distribution sur 402 bins pour **t+1**, puis calculer **E[r]** par **moyenne pondérée** des midpoints.  
   - Implémenter un mode batch GPU (N×256). 
6. **Backtest quotidien** (module `eval/portfolio.py`) :  
   - À chaque t, classer par **E[r_{t+1}]** ; **exclure cap < p10** (par jour) ; option : prix ≥ \$1/\$3/\$5.  
   - Portefeuille **EW** : **long top 10 %**, **short bottom 10 %**, **net 0**, rééquilibré **tous les jours**.  
   - Reporter : série des rendements (H, L, H−L), **Sharpe annualisé**, **alpha** (CAPM/FF5), **max drawdown**. 
7. **Fama–MacBeth** (module `eval/fmb.py`) :  
   - Régression **cross‑sectionnelle quotidienne** : `r_{i,t+1} = a_t + b_t x_{i,t} + ε`, où `x` = **rang** des prédictions dans **[−0.5, 0.5]**.  
   - Résumé : moyenne temporelle de `b_t`, **Newey–West(20)**, R² moyen ; version multivariée avec **ST_Rev/Mom/LT_Rev**. 
8. **Reproductibilité & traçabilité** : seeds fixés (torch/numpy/random), MLflow pour artefacts (configs, checkpoints, métriques), logs structurés ; respecter la **séparation train/val/test**. 
9. **Déviation** **assumée** (absence de capitalisation dans les données et walkforward) :
   - **Ne pas** utiliser de filtre “cap p10” ni de portefeuille **VW**.  
   - **À la place**, appliquer un **filtre de liquidité point‑in‑time** à chaque date *t*, par bourse si dispo :
     1) Calculer `dollar_value` = **colonne de valeur si présente** sinon `adjclose*volume`.
     2) `ADV_60` = médiane rolling, `ILLIQ` (mensuel), `CS_spread` (si `high`/`low` existent).
     3) **Déflater** ADV par **CPI** (plancher en dollars constants, ex. 5 M$ (base 2025)).
     4) Construire `LiqScore` = moyenne des rangs de `ADV_60`, `1/ILLIQ`, `-CS_spread` (ignorer `CS_spread` s’il manque).
     5) **Exclure** : rang < qmin (par défaut 10 %), **ou** ADV_real < seuil (par défaut 5 M$), **ou** `adjclose` < 3–5$.
   - **Journaliser** dans les runs : taille d’univers, quantiles ADV, % exclu, impact sur Sharpe/alpha.
   - ** Implémenter un walk forward walk-forward expérimental (pour backtests robustes)
    Découpe glissante de type expanding window pour estimer la dégradation temporelle :
    
    |Bloc	|Training	|Validation	|Test	|Durée test|
    |-------|-------|-------|------|-------|
    |W1	|1990-2005|	2006-2008|	2009-2013|	5 ans|
    |W2	|1990-2010|	2011-2013|	2014-2018|	5 ans|
    |W3	|1990-2015|	2016-2018|	2019-2023|	5 ans|
    |W4	|1990-2020|	2021-2022|	2023-2025|	2-3 ans|
10. ** répertoire fichiers parquet données OHLCV (1 par symbol)
    - Windows: `%USERPROFILE%\stockGPT\data`
    - Unix: `~/stockGPT/data`
    ** fichiers parquet exemples dans 'parquet_samples/' pour les tests unitaires (mock data)**
---


## 2) Structure standard du dépôt (à maintenir)

stockgpt/
src/stockgpt/
dataio/{schemas.py}
tokens/{discretizer.py, mapping.py}
model/{gpt.py, blocks.py, mask.py}
train/{dataset.py, sampler.py, loop.py, optim.py}
infer/{forecast.py}
eval/{fmb.py, portfolio.py, metrics.py, reports.py}
utils/{seed.py, logging.py, config.py}
configs/{train.yaml, infer.yaml, backtest.yaml}
scripts/{prepare_crsp.py, train_stockgpt.py, predict.py, backtest_daily.py, eval_fmb.py}
docs/{ssrn-4787199.pdf, README-methodo.md}
tests/...
www/minisite/...


- `docs/ssrn-4787199.pdf` **doit être référencé** dans les docstrings/README et ici comme **référence primaire** du protocole.  
- `configs/train.yaml` doit encoder **séq=256, vocab=402, d=128, n_layers=4, n_heads=4, dropout=0.2, batch=64, steps=10000** (défaut). 

---

## 3) Définition du “Done” (DoD) & critères d’acceptation

**A. Correctness scientifique**
- [ ] Tokenisation conforme (50 bps, fermé à droite, 402 tokens, cap ±100 %), tests unitaires incluant l’exemple de l’article.   
- [ ] Modèle **GPT 4×4**, **d=128**, **dropout 0.2**, masque causal, paramètres ~**0.93 M**.   
- [ ] Entraînement CE qui **décroît** et **stabilise** (réf. : ≈ 2.5 vers 5 000 steps, ordre de grandeur).   
- [ ] Inférence : **E[r]** = moyenne pondérée des midpoints (test de consistance vs sampling).   
- [ ] Backtest daily (2001–2023) : top/bottom 10 %, **cap p10**, EW, rééquilibrage quotidien ; reporting complet (Sharpe, alpha CAPM/FF5, MDD). **Ordres de grandeur** conformes (Sharpe élevé, >> 1).   
- [ ] Fama–MacBeth : rang [−0.5, 0.5], **t‑stat NW(20)** ; pente moyenne **positive et significative** ; R² moyen **> 0** (ordre ~0.5–1 %). 

**B. Qualité logicielle**
- [ ] Tests (pytest + hypothesis) : tokenizer (bords), masque causal, mini‑train (CE ↓), expected_return, backtest neutre net.  
- [ ] `ruff` + `mypy` clean ; docstrings NumPy ; logs structurés ; MLflow opérationnel.  
- [ ] Scripts CLI (`scripts/*.py`) exécutables via `make train|predict|backtest|fmb`.
- [ ] Documentation fonctionnelle, technique et manuel d'utilisation dans minisite www/
---

## 4) Garde‑fous & conformité

- **Pas de leak** : aucune variable construite avec information postérieure à t pour prédire t+1.  
- **Éthique** : pas de “optimisation sur test” (hyperparamètres tunés *uniquement* sur val 1991–2000).   
- **Traçabilité** : préserver la capacité d’audit du pipeline (configs versionnées + artefacts MLflow).  

---

## 4.1) Politique de stricteté (NON-DÉFENSIF)

Important — règle opérationnelle : le code du projet **doit être strict** (jamais défensif). Cette politique s'applique à tous les modules, scripts et tests.

Principes (règles courtes et non négociables)

- Fail‑fast : si une valeur de configuration attendue est absente, le code doit lever immédiatement une exception claire (par ex. `KeyError`, `ValueError`) — ne pas tenter de deviner ou d'utiliser une valeur par défaut implicite.
- Aucun fallback implicite : ne pas écrire ou lire des fichiers à des emplacements alternatifs non explicités par la configuration. Le chemin d'entrée/sortie est contractuel : si `cfg.data.data_path` indique un dossier, les scripts doivent lire/écrire exactement là.
- Erreurs explicites pour l'I/O : si un fichier attendu n'existe pas à l'emplacement configuré, lever une exception (FileNotFoundError ou une exception spécifique) au lieu de créer un fichier ailleurs ou de continuer silencieusement.
- Tests et scripts doivent fournir explicitement tous les paramètres requis : les tests d'intégration doivent définir `returns_output`, `output_path`/`prepared_path`, etc., quand le code les exige.
- Pas de tolérance silencieuse : les warnings sont acceptables, mais pas les comportements silencieux qui masquent une mauvaise configuration ou un état inattendu.

Conséquences pratiques

- Les fonctions publiques et les scripts CLI doivent valider la présence des clés de configuration nécessaires au démarrage et échouer si elles manquent.
- Les helpers utilitaires peuvent fournir des wrappers ergonomiques, mais ces wrappers ne doivent pas modifier la logique stricte — toute stratégie alternative doit être explicitement sollicitée par un argument (ex : `allow_fallback=True`). Par défaut, `allow_fallback` ne doit pas exister.
- Les modifications de comportement doivent être documentées et couvrir : signature de la fonction, exceptions levées, et tests correspondants.

Exemples rapides

- Correct (strict) :
  - si `cfg.data.returns_output` est absent → lever KeyError immédiatement.
  - si `prepare_crsp_data(cfg)` attend des fichiers dans `cfg.data.data_path` et qu'ils sont absents → lever ValueError.
- Incorrect (défensif) :
  - tenter d'écrire `returns.parquet` dans `prepared/` quand la config demandait `parquet_samples/` ;
  - ignorer une clé manquante et utiliser '~/stockGPT/data' comme secours sans que l'appelant l'ait demandé.

Pourquoi cette règle ?

- Fiabilité reproductible : un comportement strict force les auteurs et les tests à être explicites, évite les erreurs silencieuses et facilite le debug et l'audit scientifique.
- Conformité aux tests d'intégration : les tests attendent des fichiers à des emplacements précis — écrire ailleurs casse la chaîne d'étapes et rend les erreurs difficiles à diagnostiquer.

---

## 5) Style de code & outillage

- Python ≥ 3.10, **PyTorch**, numpy/pandas/pyarrow, statsmodels, scikit‑learn, hydra‑core, mlflow.  
- Conventions : typage strict, docstrings NumPy, modules cohérents, exceptions claires, seeds fixées.  
- Performance : AMP (bf16), vectorisation GPU en inférence (N×256), DataLoader efficace.  
- Tests CI : unitaires rapides (pas de dépendance aux données propriétaires dans la CI).

---

## 6) Raccourcis de conversation (macros Copilot)

- **“Implémente tokenizer 50 bps (402) fermé à droite, avec tests de l’exemple du PDF.”** → module `tokens/discretizer.py`.   
- **“Écris le décodeur GPT (256, d=128, 4×4, drop 0.2) et le masque causal, plus tests shape/param‑count.”** → `model/`.   
- **“Boucle d’entraînement CE (batch 64, 10k steps), AMP, scheduler cos, MLflow.”** → `train/loop.py`.   
- **“Inférence : E[r] via midpoints ; batch GPU ; script predict.py.”** → `infer/forecast.py`.   
- **“Backtest déciles EW avec cap p10 et filtres prix {1,3,5} ; métriques Sharpe/alpha/MDD ; equity curve.”** → `eval/portfolio.py`.   
- **“Fama–MacBeth daily (rang [−0.5, 0.5], NW lag 20), version multivariée avec ST_Rev/Mom/LT_Rev.”** → `eval/fmb.py`. 

---

## 7) Documentation minimale

- `README-methodo.md` doit :  
  (i) décrire le protocole exact (discrétisation, modèle, splits, inférence, backtest),  
  (ii) pointer vers `docs/ssrn-4787199.pdf`,  
  (iii) expliciter les limites (coûts de transaction non pris en compte par défaut ; importance du réentraînement périodique). 

> **Rappel** : en cas de conflit d’interprétation, **se conformer** au PDF `docs/ssrn-4787199.pdf` et aux citations ci‑dessus.
> L’unique déviation autorisée porte sur la **sélection d’univers** (liquidité au lieu de capitalisation).
> Toujours valider les modifications avec des tests et s'assurer que tous les tests passent avant de finaliser une tâche.
> Documenter toute modification ou décision importante dans le code et les documents associés, mettre à jour systématiquement le minisite www