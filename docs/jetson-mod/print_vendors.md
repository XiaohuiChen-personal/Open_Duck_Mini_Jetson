# Print vendors — who can supply this part set, and what to ask them

**Researched 2026-08-12** (Task M1 addendum). Seven bureaus examined against
their own published pages; a wider availability sweep covered 19. This document
exists so that quotes can be requested whenever the owner is ready, without
redoing the research.

**The part set:** 52 pieces, 1571.94 cm³ solid volume, largest part
`body_front` at 143 × 110 × 10 mm, 33 of the 52 pieces under 30 × 30 × 15 mm,
several 2–3 mm thick. Chosen process **`fdm-asa`** — see
[`print_process_decision.md`](print_process_decision.md).

---

## 1. Can the material be ordered? Yes.

**ASA in FDM is stocked at 16 of 19 online services surveyed** — it is not a
niche material. It is absent only where the FDM line is unusual: Sculpteo
(rPET + PETG-ESD only), Materialise OnSite (ABS + ULTEM), Quickparts (legacy
Stratasys set). ASA runs on the same enclosed, heated-chamber machines as ABS,
so any bureau already running ABS gets ASA nearly free.

**All seven bureaus examined in depth also offer MJF/SLS PA12**, so the fallback
process needs no different supplier.

## 2. The shortlist

Ranked for this job. "Determinism" means: can the part mass be known **before**
placing the order?

| # | vendor | ASA | shell count | infill | mass determinism | verdict |
|---|---|---|---|---|---|---|
| **1** | **Protolabs Network** (hubs.com) | ✅ | **published, fixed: 3 shells / 1.2 mm wall** | selectable `[20,30,40,60,80]` | **exact — 1163.14 g** | **primary** |
| 2 | Craftcloud (All3DP) | ✅ | not exposed | selectable `[20,40,60,80,95,100]`, default 20; datasheet CC-MDS-ASA-FDM | ±~120 g (shells unknown) | price cross-check |
| 3 | PCBWay | ✅ | not exposed, notes only | selectable `[20,40,60,80,95,100]`, default 20 | ±38–212 g | price cross-check |
| 4 | Xometry | ✅ | **auto, never disclosed** — "2–3 contour layers based on part size" | tiers only: Ultralight/Light/Hexagram/Solid ≈ 7/14/20/100 % | **unknowable** | reject for mass |
| 5 | Treatstock | ✅ | negotiable with the maker | negotiable | contractual if stipulated | niche option |
| 6 | JLC3DP | ✅ | unknown | unknown | unknown | **disqualified — see §3** |
| 7 | Sculpteo | ❌ | — | — | — | no ASA |

### Why Protolabs Network is primary

It is the only bureau found that **publishes an unconditional shell count**:

> *"All parts are printed with 3 outline / perimeter shells or a wall thickness
> of 1.2 mm."*

paired with a closed, machine-readable infill enumeration. Both axes of the mass
model are therefore pinned **by publication, ahead of the order** — no sales
email, no waiting, no trust required. At 3 shells / 20 % infill this part set
measures **1163.14 g** with zero residual profile uncertainty.

That costs 158.85 g against the (unorderable) 2-perimeter assumption and about
10 points of servo continuous-torque margin. It buys back the entire failure
class that produced PLANT-10.

### Why Xometry is rejected despite looking best on mass

Its ASA "Light" tier is ~14 % — closer to the original assumption than anyone
else — and small parts might get 2 contours rather than 3. But the contour count
is chosen by part size and **never disclosed**, the realised infill is
explicitly geometry-dependent (*"the percent density of parts varies
considerably"*), and Xometry runs a mixed fleet (Fortus 450 / Fortus 900 /
Prusa MK4 / Bambu X1C) without telling you which machine printed your part.
Best-case-lighter, worst-case-unknowable is the wrong trade for a plant that
feeds an expensive retrain.

## 3. JLC3DP is disqualified by geometry, not by policy

JLC publishes a **minimum build size of 30 × 30 × 15 mm** for ASA FDM. Measured
against the actual STLs in `print/`: **20 of 37 distinct parts — 33 of the 52
pieces — fall under that floor**, including `body_front` (10 mm thick),
`thermal_partition` (2 mm), `head_bot_sheet` (3 mm), both eyes, both antenna
holders, all eight knee/ankle sheets, all four leg spacers and both feet. Add a
10-files-per-upload limit and a 50-orders-per-batch cap and it is the wrong
vendor for a 52-piece set.

> **A number to distrust.** A "JLC3DP default 50 % infill" figure circulates and
> was briefly repeated in this project's own working notes. **It is
> unsubstantiated** — the source is a customer Q&A page with zero answers. Do
> not plan against it.

## 4. The mass sensitivity these vendors are being judged against

Measured on this machine, PrusaSlicer 2.7.2 arm64, whole set in ASA:

| | 14 % | 15 % | 20 % | 30 % |
|---|---|---|---|---|
| **2 perim** | 998.85 | 1004.29 | 1041.95 | 1119.74 |
| **3 perim** | 1129.00 | 1134.68 | **1163.14** | 1228.83 |
| **4 perim** | — | — | 1254.09 | — |

- one extra perimeter (at 15 % infill): **+130.39 g**
- five more infill points (at 2 perimeters): **+37.66 g**

**82 % of the exposure is the perimeter axis — the one no vendor lets you
choose.** Pinning infill removes only ~18 % of the risk. This is why a
*published* shell count outranks an adjustable infill slider, and why the
shortlist is ordered the way it is.

Six parts carry 62 % of the perimeter sensitivity: `head` +25.26 g,
`body_back` +14.71, `body_front` +11.16, `body_middle_top` +10.57,
`right_cache` +10.06, `left_cache` +10.01 — the large thin-walled shells, where
perimeters are most of the part.

## 5. Costs and minimums found

| vendor | pricing | minimum / fees |
|---|---|---|
| Protolabs Network | instant per-part quote; no rate card | **minimum order value NOT PUBLISHED** — verify in cart |
| Xometry | instant per-part quote | no minimum order quantity (KB 642) |
| PCBWay | quote; bounding-box volume driven | **$25 min order value**, $5–20 startup per material, ~$1/part manual post-processing → ~$52 of per-part labour on this job |
| Craftcloud | binding partner quotes, no markup | none found |
| Treatstock | per-maker, priced per gram | per-maker minimum $7.99–$105 |
| Sculpteo | instant quote | €50 / $50 minimum order |

## 6. What to ask when requesting a quote

Send to **Protolabs Network** (`networksales@protolabs.com`), and to PCBWay and
Craftcloud as a price cross-check. One message.

1. **"Please send the slicer's estimated part mass in grams, per part, from the
   exact profile you will run."** The highest-value ask on this list. They
   already compute material volume to price the job, so it is cheap for them,
   and it collapses perimeters, infill, skins, layer height and nozzle diameter
   into one number that goes straight into the mass model. Ask for it as a
   condition of the order.
2. **"What shell/perimeter count and infill percentage will these parts actually
   be printed at?"** Phrase perimeters as **wall thickness in millimetres** as
   well as loop count — "0.8 mm wall = 2 loops at a 0.4 mm nozzle". Treatstock,
   Craftcloud and PCBWay all speak "wall thickness"; none speaks "perimeters".
   If the answer is a published standard, ask them to confirm in the quote email
   that it applies to *these* parts.
3. **"Is this printed in an enclosed, heated-chamber machine? Which machine?"**
   This separates Protolabs Network's Prototyping tier from its Industrial tier,
   and it is unverified which the Prototyping ASA line uses.
4. **"`body_front` is 143 × 110 × 10 mm and `trunk_top` is similar. What flatness
   can you commit to on the large face?"** Peer-reviewed measurement on almost
   this exact geometry (Armillotta et al. 2018, ABS on a closed-chamber
   Stratasys) puts the expected bow at **0.71–0.98 mm across 140 mm** — larger
   than every published tolerance, because warp is contractually carved out of
   the tolerance guarantee. **A 1 mm bow is a legitimate delivery under every
   spec found.** Also request **0.178 mm layers on that part specifically**;
   the same study measured thinner layers beating 0.254 mm at every length.
5. **"We will specify build orientation for [knee/ankle sheets, `foot_top`,
   `foot_side`, roll/pitch brackets]. Will you honour it?"** Xometry states it
   picks orientation itself unless told otherwise. Layer lines must not sit in
   the primary load path.
6. **"Total price, itemised: material, per-part handling/startup, minimum order
   value, shipping. Is plate-batched pricing available for 52 parts?"** Batching
   is the lever that moves an FDM quote, not the material.
7. **"Will any part be refused?"** 33 of the 52 pieces are under
   30 × 30 × 15 mm and several are 2–3 mm thick. Ask up front.

## 7. What is still unverified

Recorded so nobody mistakes absence of evidence for evidence of absence.

- Whether Protolabs Network's **Prototyping** ASA tier uses an enclosed heated
  chamber. The Industrial tier (ASA Stratasys, Fortus-class) is enclosed by
  construction and also offers 20 % infill — but whether the 3-shell house
  standard applies identically to Stratasys contour generation is unverified.
- Protolabs Network's **minimum order value** is genuinely unpublished, not
  zero.
- The 3-shell standard is stated as an **"or"** — "3 outline/perimeter shells
  **or** a wall thickness of 1.2 mm" — and their Prototyping ASA minimum wall is
  0.8 mm. Which branch governs a given part is not specified. The 1163.14 g
  figure assumes 3 shells, which is the better reading of an ambiguous sentence,
  but it is an inference.
- Whether any vendor will honour a customer slicer profile off-platform. **No
  vendor publishes a statement that they will.** PCBWay has a free-text box,
  Craftcloud takes production notes (with a documented "up to 7 % handling fee"
  for post-acceptance variation), Protolabs has a sales address with 48 h
  turnaround, and Treatstock makes a customer-stipulated wall thickness
  contractually binding under Terms Appendix A. None of that is a published
  commitment.
