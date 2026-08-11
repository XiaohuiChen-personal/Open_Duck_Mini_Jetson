# Print guide

You can find the `.stl` files under the `print/` directory at the root of this repo. 

All the parts are printed in standard PLA with 15% infill, except for `foot_bottom_tpu.stl`, which is to be printed in TPU at 40% infill.

> **Perimeter count is not specified above, and it should be.** Slicing this
> part set at 15% infill gives **1,158 g at 2 perimeters and 1,309 g at 3** —
> a 151 g swing on a 2.8 kg robot, larger than any material choice. Pick one and
> record it here before printing.
>
> **If you are buying these parts from a printing service rather than printing
> them**, infill stops being a lever for powder processes (MJF/SLS parts are
> solid) and the same geometry lands at ~1,588 g in MJF PA12. See
> `docs/jetson-mod/known_issues.md` [PLANT-10](jetson-mod/known_issues.md#plant-10)
> for why the mass model is sensitive to this.

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
