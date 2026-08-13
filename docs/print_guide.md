# Print guide

You can find the `.stl` files under the `print/` directory at the root of this repo. 

~~All the parts are printed in standard PLA with 15% infill~~, except for
`foot_bottom_tpu.stl`, which is to be printed in TPU at 40% infill.

> ## ⛔ The material and profile are decided elsewhere now — 2026-08-12
>
> **The PLA line above is superseded and was wrong.** The single source of truth
> is **`scripts/print_process.json`**, with the reasoning in
> [`docs/jetson-mod/print_process_decision.md`](jetson-mod/print_process_decision.md).
> Task M1 chose:
>
> | field | value |
> |---|---|
> | process | **`fdm-asa`** (ASA, not PLA) |
> | perimeters | **3** (Protolabs Network's published house standard) |
> | infill | **20 %** (the lowest any researched bureau offers) |
> | TPU sole | unchanged — `foot_bottom_tpu` ×2 in TPU 95A at 40 % |
> | measured set mass | **1163.14 g** (52 pieces, 1571.94 cm³) |
>
> **Why not PLA.** The trunk parts *are* the cavity enclosing the Jetson Orin
> Nano, whose heatsink reaches **55–80 °C**. PLA's heat-deflection temperature
> is ~60 °C and it creeps under sustained load well below that. ASA is ~91 °C
> **and** 154 g lighter than PLA at the same profile — better on both axes.
>
> **Perimeter count is now specified: 3.** That resolves the TODO this block
> used to carry, and it is set to what a bureau will actually print rather than
> to a hobbyist default. It matters more than the material: one extra perimeter
> is **+130.39 g** and five infill points are **+37.66 g**, so **82 % of the
> mass exposure sits on the perimeter axis** — and **no researched bureau lets a
> customer choose it**. See [`print_vendors.md`](jetson-mod/print_vendors.md).
>
> **If you are buying these parts from a printing service** — which is the plan;
> there is no printer — the profile above comes from **Protolabs Network's
> published standard** (*"All parts are printed with 3 outline / perimeter
> shells or a wall thickness of 1.2 mm"*, infill selectable from
> `[20,30,40,60,80]`), so the mass is pinned **before** ordering. If you order
> elsewhere, re-run
> `python3 scripts/measure_print_mass.py --process fdm-asa --perimeters N --infill M`
> with that vendor's real numbers and re-run M2/M4. If a vendor will state
> neither, the decision reverts to `mjf-pa12`, whose mass is volume × density
> and needs no promise. Vendor shortlist and the quote question list:
> [`print_vendors.md`](jetson-mod/print_vendors.md).
>
> For a powder process, infill stops being a lever — MJF/SLS parts are solid —
> and the same geometry lands at **1,598 g** in MJF PA12. (Not 1,588 g: that
> figure was the all-PA12 total, 1571.94 cm³ × 1.01, and omitted the TPU sole,
> which is 47.20 cm³ at 1.22 g/cm³ rather than 1.01 — worth +9.91 g.)

## Parts to print
- foot_top.stl x2
- foot_side.stl x2
- foot_bottom_pla.stl x2
- foot_bottom_tpu.stl x2 (TPU)
- knee_to_ankle_left_sheet.stl x4
- knee_to_ankle_right_sheet.stl x4
- leg_spacer.stl x4
- left_roll_to_pitch.stl x1
- right_roll_to_pitch.stl x1
- roll_motor_bottom.stl x2
- roll_motor_top.stl x2
- trunk_bottom.stl x1
- trunk_top.stl x1
- neck_left_sheet.stl x1
- neck_right_sheet.stl x1
- head_pitch_to_yaw.stl x1
- head_yaw_to_roll.stl x1
- head_roll_mount.stl x1
- head.stl x1
- head_bot_sheet.stl x1
- left_antenna_holder.stl x1
- right_antenna_holder.stl x1
- left_cache.stl x1
- right_cache.stl x1
- body_front.stl x1
- body_middle_bottom.stl x1
- body_middle_top.stl x1
- body_back.stl x1
- battery_pack_lid.stl x1
- bulb.stl x1
- flash_light_module.stl x1
- flash_reflector_interface.stl x1
- left_eye.stl x1
- right_eye.stl x1
- speaker_interface.stl x1
- speaker_stand.stl x1

### Jetson modification (not part of the upstream build)
- thermal_partition.stl x1 — 2 mm PLA, backed with a mica sheet; separates the
  Jetson from the battery compartment. `print/thermal_partition.stl` exists in
  this repo but was missing from the list above.

Totals: **36 upstream parts / 51 pieces**, plus the partition = **52 pieces**,
**1,571.94 cm³** of solid volume. Note the nine ×2/×4 rows — counting distinct
files instead of pieces undercounts the set by 13%.
