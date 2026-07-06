# Trunk Component Layout v2 (post-audit)

**Status: authoritative.** Supersedes the component positions in the first
pass of `mass_inertia_calculations.md` and the position hints in
`task_plan.md` Phase 3. Regression-tested by `tests/test_cad_dimensions.py`
(phase3). Trunk inertial derived from this layout by
`scripts/compute_trunk_inertial.py`.

## Why the layout changed

The July 2026 pre-Phase-3 audit measured the as-modeled layout and found it
physically unbuildable (all numbers mesh-measured, trunk body frame):

1. The DC-DC converter box sat 100% inside the Jetson envelope (12.6 cm³).
2. The thermal partition clipped the Jetson's rear 3 mm, was 110 mm wide vs
   a 104 mm shell interior, and passed through the servo-driver board.
3. The Jetson envelope (old pos z=0.04, span z[22.6, 57.4] mm) intersected
   the trunk_top plate (~43 cm³), the trunk_bottom spine (~30 cm³), **both
   hip-yaw servo cases (~9.7 cm³ each — servos cannot move)**, and the IMU.
4. The dev-kit fan draws air vertically, contradicting the planned
   horizontal cross-flow under a solid trunk_top.

A mesh-level occupancy scan (`trunk_top`/`trunk_bottom` ray-cast in the
Jetson footprint) showed: trunk_top is solid only at z[44.8, 85.8]; the only
trunk_bottom obstruction below that is a central spine
(x[-38, +15], y[-9, +11], floor→49.8). The feasible band for a horizontal
Jetson is bounded by the roll-bearing housings (top −13.4) below and the
hip-yaw servo cases (bottom +47.9) / trunk_top plate (+44.8) above.

## Layout v2 (trunk body frame, meters)

| Component | Position (geom pos) | Envelope z-span (mm) | Notes |
|---|---|---|---|
| Jetson Orin Nano dev kit | (-0.03, 0, **0.006**) | [-11.4, +23.4] | Low mount on trunk_bottom posts; top clears trunk_top plate (44.8) by 21 mm → that gap is the **fan intake plenum** |
| Thermal partition | (**-0.086**, 0, **0.0104**) | [-23.6, +44.4] | Resized to **3 × 103.5 × 68 mm** (was 3×110×90): fits the 104 mm interior, spans floor → plate underside; ≥3 mm behind Jetson rear face |
| DC-DC converter | (**-0.060, 0.025, 0.0355**) | [28.5, 42.5] | Under-plate mount (foam pad), rear corner of the plenum, off the fan axis; short 19 V run |
| BNO055 IMU | (**-0.100**, 0, 0.0418) | [41.8, 44.8] | Battery-side pocket between partition and lid: cool, stable temperature (BNO055 drift is thermal), away from Jetson EMI |
| Servo-driver board | (**-0.01599**, 0.0165, 0.0604) | [58.8, 60.4] | Shifted +47.5 mm forward, out of the partition plane; clear of the neck-servo case (4.6 mm) |
| Battery pack (6× 18650, BMS) | unchanged | — | Rear hump behind the lid (x ≤ -0.114) |

Whole-robot mass is unchanged (2.745549 kg). New trunk inertial (from
`compute_trunk_inertial.py`, all components at their v2 positions,
replacing the earlier rounded mass-ratio scaling):

```
pos  = (-0.0579714, 0.0001468, 0.0332356)
mass = 1.178026
diag = (0.00403131, 0.00473388, 0.00337805)
```

CoM moves 4.5 mm rearward / 4.8 mm down vs the first-pass model — this is
the delta training Run A ("v4-inertials") is designed to absorb.

## Clearances (asserted in test_cad_dimensions.py)

- Jetson ↔ cavity floor ≥ 2 mm (12.5 mm actual — standoff posts)
- Jetson ↔ side walls ≥ 6 mm/side (6.75 actual)
- Jetson ↔ trunk_top plate ≥ 5 mm (21.4 actual — plenum)
- Jetson ↔ roll bearings ≥ 1.5 mm (2.0 actual)
- Jetson ↔ partition ≥ 2 mm (3.0 actual)
- Partition ↔ battery lid ≥ 2 mm (26.5 actual, IMU lives in this pocket)

## Keep-outs and known compromises

- **Port face = +y side.** The dev-kit I/O (USB×4, RJ45, DP, USB-C, barrel
  jack) sits on one 103 mm edge; 6.75 mm of side margin cannot take plugs.
  Part 2 cuts a port opening in the +y side wall of `body_middle_bottom`
  (connectors pass through / seat into the opening). GPIO 40-pin header
  faces −y (low-profile IDC only).
- **Fan airflow**: intake plenum above the heatsink (fed by front inlet
  vents through the plenum), exhaust spreads to the rear vents. The DC-DC
  occupies a 43×21 mm corner of the plenum — kept off the fan axis.
- **Known thermal leak path** (accepted for v1): compute-zone air above the
  trunk_top plate can migrate rearward over the plate and descend into the
  battery hump. The path is long and against buoyancy; the battery
  thermistor + software throttling (Task 4.6) are the backstop. If the
  Task 3.6 IR measurement (>15 °C differential requirement) fails, add a
  foam gasket on the plate top at x ≈ -0.09.
- The Jetson bbox is the bare dev kit; cable bend radii (CSI to head, USB
  to servo board) route upward through existing plate openings.

## Part-2 mesh cut list (implements this layout in the printed parts)

1. **trunk_bottom**: remove the central spine within
   x[-0.044, 0.021], y[-0.016, 0.017], z[-0.0136, 0.051] (the declared
   `SPINE_CUT_BOX` in test_cad_dimensions.py — the test currently allows
   Jetson↔trunk_bottom intersection only inside this box; Part 2 makes the
   actual cut and tightens the assertion to zero). Add twin replacement
   ribs at y ±[0.0465, 0.0515] (outside the Jetson envelope, floor→plate)
   to restore trunk_top mid-span support, and 4× M3 standoff posts on the
   Jetson 58×86 mm hole pattern (12.5 mm tall).
2. **thermal_partition (printable)**: 2 mm PLA + 1 mm mica, 103.5 × 68 mm,
   with 5 mm cable notch bottom-center and floor/plate retention features.
3. **body_back / body_front**: compute-zone ventilation (rear hex exhaust
   ~40×20, front inlets ~30×15 feeding the plenum); battery side solid.
4. **body_middle_bottom**: +y port-face opening sized to the dev-kit I/O
   edge; optional −y GPIO slot.
5. **battery_pack_lid**: 6-cell (3S2P) capacity, partition mating lip,
   NTC thermistor clip, <5 mm cable notch.
6. body_middle extension: NOT needed (low mount fits with 21 mm headroom).
