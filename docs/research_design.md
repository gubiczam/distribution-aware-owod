# Research Design — Distribution-Aware OWOD

**This document is the architecture contract.** It is derived strictly from
`docs/proposal.docx`. Every file in this repository must map onto a requirement stated
here. If it cannot, it is deleted.

Quotations are from the proposal, cited by section heading. The proposal is in Hungarian;
load-bearing wording is quoted verbatim and translated.

---

## 1. Central problem

§*Bevezető* — the aspect of OWOD this research targets:

> "az ismeretlen objektumok egyenlőtlen (long-tail) eloszlásának hatását, valamint ezen
> hatások csökkentését eloszlás-tudatos open-world objektumdetektálás bevezetésével"

§*Feladat- és ütemterv* — the unifying thesis. The field's three known symptoms are

> "nem feltétlen függetlenek, az ismeretlenek egyenlőtlen eloszlásának három különböző
> megnyilvánulásai"

namely (i) rare classes suppressed by frequent ones → low U-Recall *measured on rare
classes*; (ii) labelling cost badly distributed → redundant samples; (iii) rare new
classes lost during incremental update.

And the methodological claim that justifies every metric in §5:

> "Az aggregált, osztályok közötti átlagra épülő kiértékelés ezt a szerkezetet elrejti"

Aggregate cross-class averaging hides the structure. Grouped evaluation is therefore not
a nicety; it is the instrument.

## 2. Contribution A — distribution-aware active selection

§*A) Eloszlás-tudatos aktív kiválasztás a címkézési költség elosztására*.

Classical OWOD assumes the oracle labels every detected unknown ("A klasszikus OWOD
feltételezi, hogy az orákulum minden detektált ismeretlent felcímkéz (OW-CLIP [6],
CROWD [5])"). In practice it cannot, so a labelling subset must be chosen — deliberately
over-weighting tail classes ("a tail-osztályok szándékos felülsúlyozása").

The score, proposal equation (1):

```
s(x) = U(x) + λ·D(x) + γ·w(ĉ(x))·coh(x)          with   w(c) ∝ 1/n_c
```

`U` uncertainty, `D` diversity/representativeness, `w` a distribution-driven weight
favouring estimated-rare classes, `coh` a local density / cluster-coherence measure —
"a k-adik legközelebbi szomszéd távolságának inverze vagy egy ε-sugáron belüli minták
száma a jelöltek jellemzőterében".

**H-A1 — the gate is indispensable.**

> "Az utolsó tag elengedhetetlen, mert a többi három komponens önmagában az izolált
> outliereket felülértékeli."

A lone candidate is simultaneously uncertain (high `U`), maximally different (high `D`)
and necessarily estimated-rare (high `w`), so pure diversity/rarity selection "éppen a
haszontalan outliereket vonzza" — attracts precisely the useless outliers.

**H-A2 — the AND relation.**

> "a tail-prioritás, valamint a coh tényező között ÉS kapcsolat van, így a ritkasági
> pontszám csak akkor aktiválódik, ha a jelölt ritka ÉS van néhány hasonló társa"

For a lone point, "a közel nulla koherencia semlegesíti azt". The stated novelty is
adapting density-based clustering ("DBSCAN core-point vs. noise megkülönböztetése") to
OWOD active learning, going beyond existing OWOD active selection "mivel azok az
izolált-outlier problémát nem kezelik expliciten".

**H-A3 — estimation is the hard part.** Before the oracle the true class is unknown, so
"ĉ(x) és n̂_c csak becslés (pszeudo-címke vagy klaszter-hozzárendelés alapján), amelyet
minden annotációs kör után iteratívan frissítünk" — re-estimated after every round.

## 3. Contribution B — distribution-aware exemplar allocation

§*B) Eloszlás-tudatos exemplar-allokáció az inkrementális frissítésben*.

Replay-based incremental methods ("Replay, iCaRL, BiC, WA, DER [7]") keep an exemplar
memory. The standard setting stores a fixed count per class or splits a fixed total
evenly, which "osztályonként egyenlő exemplart eredményez". Under a long tail this hurts
rare classes: they already learned from little data ("gyengébb reprezentáció"), are more
sensitive to interference from new classes, "így gyorsabban felejtődnek", and the BiC/WA
bias correction "éppen a tail ellen hat".

**The allocation rule.**

> "Legyen c osztály mérete n_c, ekkor a hozzá tartozó m_c exemplar-allokáció arányos
> n_c^α-val, ahol Σ m_c = M a teljes memória."

**The interpolation.** "α = 0 az egyenletes elosztás (jelenlegi standard), α = 1 a
mérettel arányos (head-favorizáló), α < 0 a tail-favorizáló elosztás."

**H-B1 — the research question**, verbatim: "mi az optimális α, és ez miként függ a tail
súlyosságától."

**H-B2 — the two granularities.** "egy eltárolt kép több objektumot (több osztályt)
tartalmazhat, így az »exemplar per osztály« fogalom objektum-szintű és kép-szintű
allokációként is értelmezhető."

## 4. The closed loop

§*Feladat- és ütemterv* and Fig. 2: A and B operate in "egységes, visszacsatolt
keretrendszerben" — a unified, feedback-coupled framework. The shared signal is the class
distribution estimated from detections, "amelyet az inkrementális frissítés eredményének
visszacsatolásával és az új detekciókból tudunk becsülni a következő körre".

Cycle: detection → pseudo-class → distribution-aware selection (A) → oracle (bounded
cost) → distribution-aware exemplar allocation (B) → incremental update → detection.

## 5. Base model, datasets, metrics

§*Alapmodell, használt adathalmazok és kiértékelés*.

**Base model.** PROB (Probabilistic Objectness) [8], Deformable-DETR backbone, chosen
because it is "jól reprodukálható" and its Gaussian approximation yields probability
values "alkalmas … az aktív kiválasztási stratégiához".

**Datasets.**

| Dataset | Role | Quotation |
|---|---|---|
| M-OWODB, S-OWODB [1] | primary task-sequence protocols | "a legelterjedtebb M-OWODB és S-OWODB OWOD feladat szekvencia protokollokon [1] keresztül végzzük" |
| Controlled long tail | the tail must be *constructed* | "a kontrollált tailt szándékos alulmintavételezéssel állítjuk elő (előre definiált kiegyenlítettlenségi arány mellett)" |
| LVIS [10] | confirmation on a naturally long-tailed dataset | "megerősítésként pedig a valóban long-tail LVIS adathalmazon [10] is lemérjük" |

**Baselines.** For A: uncertainty-only and diversity-only selection; OW-CLIP [6] and
CROWD [5] as the label-everything reference. For B: α = 0 (the "jelenlegi standard") and
α = 1. Decoupled PROB [9] appears in the bibliography only and is **not** a baseline.

**Metrics.** The standard set — "az ismert osztályokra számolt mAP, ismeretlenekre
számolt U-Recall, az ismert/ismeretlen összemosódásra Wilderness Impact (WI) és Absolute
Open-Set Error (A-OSE) [1]" — plus the distinguishing element:

> "A kutatás megkülönböztető eleme a head/medium/tail bontásban végzett kiértékelés,
> amely során csoportonkénti mAP és U-Recall, csoportonkénti felejtés, valamint a
> tail-U-Recall mérőszámokat értékelek ki, mint az orákulum-költség függvénye, ezzel egy
> annotációs hatékonyság-görbe írható fel."

Pre-registered expected direction: "A várt tendencia, hogy az eloszlás-tudatos
kiválasztás azonos tail-szintet lényegesen kevesebb annotációból ér el."

## 6. Required outputs

Annotation-efficiency curve (tail-U-Recall against oracle cost, per arm); the α response
curve including per-group forgetting; the Fig. 3 outlier-versus-rare-class discrimination
as a measurement; grouped mAP / U-Recall / forgetting tables; the standard four metrics
for comparability with the literature.

## 7. What the proposal does NOT require

Nothing in the proposal asks for any of the following, and each was present in the
repository before this refactor. This list is the licence to delete.

1. **Synthetic or simulated candidate pools.** §5 names PROB, M-OWODB, S-OWODB, LVIS.
2. **Two coexisting scoring semantics.** Equation (1) is one formula.
3. **Reproduction of pre-audit numbers.** No versioning or compatibility requirement
   appears anywhere in the proposal.
4. **Stage-gating process artifacts** — GO checklists, audit addenda, supersession notes.
5. **A separate image-level acquisition budget.** In §2, `x` is a "jelölt-régió"
   (candidate region) and §5 puts *oracle cost* on the x-axis.
6. **Runtime auto-downscaling of the evaluation pool.** Nothing licenses mutating the
   protocol mid-run to fit a session budget.
7. **Decoupled PROB [9]** as a baseline.
8. **Notebook generators, one-shot data-prep scripts, per-strategy config files.**

## 8. Implementation status against this contract

| Requirement | Status |
|---|---|
| A: equation (1), all four components, iterative re-estimation | **Implemented and executed** on real S-OWODB Task-1 proposals |
| H-A1 / H-A2 | **Tested, and falsified in the spaces measured** — see `docs/results.md`. This is a result about a named proposal hypothesis and is retained |
| H-A3 | **Tested** via the label-anchored (revealed) estimator |
| Controlled long-tail construction | **Implemented** (three severities) |
| Grouped metrics, annotation-efficiency curve | **Implemented** |
| B: `m_c ∝ n_c^α`, Σ`m_c` = M, object- and image-level | **Allocation core only.** Mathematically complete and unit-tested; *not* experimentally evaluated |
| H-B1 (optimal α vs tail severity) | **Not measured.** Requires incremental model updates |
| §4 closed loop | **Not implemented** |
| LVIS confirmation | **Not implemented** |
| Per-group forgetting | **Not measured.** Requires real incremental updates; deliberately not approximated offline |

**Explicit non-claim.** The allocation core does not make Contribution B experimentally
complete. Measuring optimal α requires actual incremental model updates with PROB
retraining, which is a separate, documented future research step. No offline proxy for
catastrophic forgetting is provided, because a proxy would not answer H-B1.
