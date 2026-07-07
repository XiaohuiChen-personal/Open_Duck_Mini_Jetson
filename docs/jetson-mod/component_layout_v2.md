# Trunk Component Layout v2.1 (post-audit, post-review)

**Status: authoritative.** Supersedes the component positions in the first
pass of `mass_inertia_calculations.md` and the position hints in
`task_plan.md` Phase 3. Regression-tested by `tests/test_cad_dimensions.py`
(phase3). Trunk AND head inertials derived from this layout by
`scripts/compute_trunk_inertial.py` (full-tensor, frame-correct — see
"Inertia frame correction" below). v2.1 incorporates the adversarial-review
fixes of the first v2 commit (7166765): inertia frame correction, board
move reverted, battery hump extension, GPIO/mounting/vent corrections.

## Why the layout changed

The July 2026 pre-Phase-3 audit measured the as-modeled layout and found it
physically unbuildable (all numbers mesh-measured, trunk body frame):

1. The DC-DC converter box sat 100% inside the Jetson envelope (12.6 cm³).
2. The thermal partition clipped the Jetson's rear 3 mm, was 110 mm wide vs
   a 104 mm shell interior, and passed through the servo-driver board.
3. The Jetson envelope (old pos z=0.04, span z[22.6, 57.4] mm) intersected
   the trunk_top plate (~43 cm³), the trunk_bottom spine (~30 cm³), **both
   hip-yaw servo cases (~9.7 cm³ each — servos cannot move)**, and the IMU.
4. The dev-kit fan draws air vertically (verified: stock fan pulls air from
   above and blows down through the fins, exhausting sideways),
   contradicting the planned horizontal cross-flow under a solid trunk_top.

A mesh-level occupancy scan showed: trunk_top is solid only at
z[44.8, 85.8]; the only trunk_bottom obstruction below that is a central
spine (y[-9, +11]; its bottom rises with x — solid from the floor for
x ≲ -10 mm, starting at z≈28 by x=0 and z≈42 by x=+14; measured Jetson
intrusion reaches only x=-1.5). The feasible band for a horizontal Jetson is
bounded by the roll-bearing housings (top −13.4) below and the hip-yaw servo
cases (bottom +47.9) / trunk_top plate (+44.8) above.

## Inertia frame correction (affects ALL prior training)

The upstream MJCF body inertials carry a `quat`: their `diaginertia` values
are PRINCIPAL-frame moments. The original Phase-1 computation (and the first
v2 commit) treated the trunk's principal moments as body-frame (Ixx,Iyy,Izz)
— a near-axis-permutation error (+78%/−10%/−27% per axis), and dropped the
head's ~4.5° principal rotation. **Every policy trained to date (v1-v3 PPO,
all AMP runs) saw the frame-permuted trunk inertia.** The generator now
composes full 3×3 tensors from the upstream URDF body-frame matrices
(self-checked against MJCF quat·diag·quatᵀ at import) and the model files
carry `fullinertia`. Run A (v4-inertials) therefore absorbs BOTH the layout
delta and this frame correction — its gate comparison against v3 measures
the combined effect.

## Layout v2.1 (trunk body frame, meters)

| Component | Position (geom pos) | Envelope z-span (mm) | Notes |
|---|---|---|---|
| Jetson Orin Nano dev kit | (-0.03, 0, **0.006**) | [-11.4, +23.4] | Low mount above trunk_bottom floor; top clears trunk_top plate (44.8) by 21 mm → **fan intake plenum** |
| Thermal partition | (**-0.086**, 0, **0.011**) | [-22.35, +44.35] | **3 × 103.5 × 66.7 mm** (floor-to-plate at its own x-plane; local floor top is -22.38); ≥3 mm behind Jetson; bottom corners get Part-2 chamfers where the shell walls fillet inward (declared `PARTITION_CHAMFER_BOXES`) |
| DC-DC converter | (**-0.060, 0.025, 0.0355**) | [28.5, 42.5] | Under-plate mount (foam pad), rear corner of the plenum, off the fan axis; short 19 V run |
| BNO055 IMU | (**-0.100**, 0, 0.0418) | [41.8, 44.8] | Battery-side pocket between partition and lid: cool, stable temperature, away from Jetson EMI |
| Servo-driver board | **unchanged** (-0.06349, 0.0165, 0.0604) | [58.8, 60.4] | First-pass move REVERTED: the shortened partition (top 44.35) no longer reaches the board's z-band, and the board's original trunk_top tray pocket is the only verified-clear mount (the moved position was inside solid trunk_top) |
| Battery pack 6× 18650 (3S2P) + BMS | hump + **declared rear extension** | — | 2 original cells at the modeled positions; **4 extra cells as a 2×2 vertical grid at (-0.145, 0, 0.0325)** inside the Part-2 hump extension (current hump interior is only ~18 mm deep behind the lid and tapers — it cannot take more than the 2 existing cells) |

Whole-robot mass after the Part-2 shell mods: **2.656149 kg** (the CAD cuts
removed ~89 g of PLA; see "Part-2 status" below). Frame-correct inertials
(from `compute_trunk_inertial.py`, including the Part-2 shell mass deltas;
MJCF `fullinertia`, URDF full matrix):

```
trunk: pos (-0.0635908, 0.0000940, 0.0343123)  mass 1.088626
       ixx 0.00220751  iyy 0.00536410  izz 0.00477170
       ixy -2.458e-05  ixz -9.915e-05  iyz -4.978e-06
       principal (0.0053643, 0.0047755, 0.0022035)
head:  pos (0.0072060, -0.0011494, 0.0223904)  mass 0.362083
       ixx 0.00207359  iyy 0.00146894  izz 0.00088770
       ixy 1.051e-05   ixz 9.104e-05   iyz -1.093e-05
```

Whole-robot standing CoM at the trained stance sits **6.3 mm behind the
foot-frame origins** (was ~2-4 mm ahead pre-layout): the honest battery
placement moved mass rearward. This is well inside the ~93 mm foot support
polygon; Run A's gait gate measures the consequence.

## Clearances (asserted in test_cad_dimensions.py)

- Jetson ↔ cavity floor ≥ 2 mm (12.5 mm actual)
- Jetson ↔ side walls ≥ 6 mm/side (6.75 actual)
- Jetson ↔ trunk_top plate ≥ 5 mm (21.4 actual — plenum)
- Jetson ↔ roll bearings ≥ 1.5 mm (2.0 actual)
- Jetson ↔ partition ≥ 2 mm (3.0 actual)
- Partition ↔ battery lid ≥ 2 mm (26.5 actual; IMU lives in this pocket)
- Board / DC-DC / IMU / partition mesh-level clear of all chassis+shell
  meshes (partition: except the two declared chamfer corners)

## Keep-outs and known compromises (review-corrected)

- **Port face = +y side** (verified: ALL dev-kit I/O — USB-A ×4, RJ45, DP,
  USB-C, barrel jack — is on ONE long edge, SP-11324-001 Fig 1-4). Part 2
  cuts a port opening in the +y side wall of `body_middle_bottom`.
- **40-pin GPIO header is on a SHORT (±x) edge, NOT −y** (SP-11324 Fig
  1-4: expansion header J12 + fan J13 + CAN J17 on a side edge; camera
  connectors J20/J21 near the opposite side edge). GPIO access is from
  ABOVE through the plenum (vertical pins, low-profile IDC, ribbon exits
  through existing plate openings). The −y wall may get a CSI-flex slot
  instead (camera edge). No −y GPIO slot.
- **Mounting: NO documented mounting-hole pattern exists for the dev-kit
  carrier (P3768)** — the earlier "58×86 mm M3" figure is not in any NVIDIA
  document (module standoffs are M2.5; carrier spec gives outline only).
  Part 2 designs an **edge-capture tray** (perimeter walls + top clips
  around the 103×90.5 base plate) integrated into trunk_bottom/floor, with
  the dev kit's own base-plate screws as fallback. Exact hole coordinates,
  if used, come from the P3768 reference design files or physical
  measurement — never from the 58×86 assumption.
- **Ventilation goes in body_middle_bottom/top, NOT body_back**: body_back
  (x[-154,-114] world) is entirely the battery hump shell — venting it
  would ventilate the battery zone. Exhaust: ±y louvers in the body_middle
  shells at fin height (z ≈ -11..+23) in the compute-zone x-range, plus the
  +y port opening. Intake: body_front slots at plenum height (z ≈ +25..44)
  feeding the above-heatsink plenum. body_back stays solid.
- **Fan airflow**: verified top-intake → down through fins → lateral
  exhaust. Plenum above the heatsink (fed by body_front inlets), lateral
  louvers evacuate at fin level. DC-DC occupies a 43×21 mm plenum corner
  off the fan axis.
- **Known thermal leak path** (accepted for v1): compute-zone air above the
  trunk_top plate can migrate rearward and descend into the hump. Long,
  against buoyancy; battery thermistor + software throttling are the
  backstop. If the Task 3.6 IR measurement (>15 °C differential) fails,
  add a foam gasket on the plate top at x ≈ -0.09.
- Partition mass budgeted at 37 g as an ASSEMBLY (bare 2 mm PLA wall +
  1 mm mica at the new size is ~26 g; the remainder budgets retention
  rails, gasket, adhesive). The slab inertia model uses the envelope.

## Part-2 status: IMPLEMENTED (scripts/generate_cad_mods.py)

All cuts below were applied by `scripts/generate_cad_mods.py` (trimesh +
manifold3d booleans) to the sim meshes AND mm-scale print/ copies; every
modified part remains a single watertight component. Measured volume deltas
(printed-PLA effective density 1.116 g/cm³, a documented 0.9×solid
assumption): spine cut −60.3 cm³ (−67.3 g), body_middle_bottom net −2.6 cm³,
body_front slots −7.2 cm³, hump extension net −10.1 cm³ (thin new shell minus
bored thick wall). **Trunk is now 1.088626 kg; total robot 2.656149 kg
(−89.4 g vs the pre-Part-2 model).** Standing CoM: −6.3 mm aft of the foot
origins. As-built details: +y port opening x[−80,−54] z[−15,+24] (the only
walled section of the I/O edge — the leg cutout already opens x[−50,+18]);
−y louvers 3× 8 mm slots z[−8,+20] at x −78/−67/−56; 6 inlet slots 20×6 in
body_front at z rows [27,33]/[37,43]; 4× Ø8 floor bosses at (−75,±38),
(+15,±38) topping at z=−11.5 (0.1 mm under the Jetson base plate — flat
seats + pilot-drill bases, since no vendor hole pattern exists);
hump extension outer to x=−170, bore y±20.5 to spare the USB-C charger
mount; print/thermal_partition.stl = 2 mm PLA wall with 45° bottom-corner
chamfers + 8×5 cable notch (mica sheet is a purchased part).

## Part-2 cut list (as designed)

1. **trunk_bottom**: remove the central spine within
   x[-0.044, 0.021], y[-0.016, 0.017], z[-0.0136, 0.051] (the declared
   `SPINE_CUT_BOX`; the regression test allows Jetson↔trunk_bottom
   intersection only inside this box — Part 2 makes the cut and tightens
   the assertion to zero). Note the spine bottom rises with x (solid from
   floor only for x ≲ -10 mm) — cut what intersects, keep the front slope.
   Add twin replacement ribs at y ±[0.0465, 0.0515] (outside the Jetson
   envelope, floor→plate) to restore trunk_top mid-span support, plus the
   Jetson edge-capture tray (see keep-outs).
2. **thermal_partition (printable)**: 2 mm PLA + 1 mm mica, 103.5 × 66.7 mm,
   bottom-corner chamfers per `PARTITION_CHAMFER_BOXES` (shell wall
   fillets), 5 mm cable notch bottom-center, floor/plate retention.
3. **body_middle_bottom/top**: ±y exhaust louvers at fin height in the
   compute-zone x-range; +y port-face opening sized to the dev-kit I/O
   edge; optional −y CSI-flex slot. **body_front**: inlet slots at plenum
   height. **body_back: solid (no vents)**.
4. **body_back + battery_pack_lid**: extend the hump rearward and
   straighten its taper so the interior takes a 2×2 grid of 18650s behind
   the existing pair (declared extension: interior deepened from
   x≈-0.1396 to ≈-0.166, ≥44 mm straight width); 6-cell (3S2P) lid with
   partition mating lip, NTC thermistor clip, <5 mm cable notch.
5. body_middle x-extension: NOT needed (low mount leaves 21 mm headroom).
