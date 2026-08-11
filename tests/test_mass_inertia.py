import pytest
import numpy as np


@pytest.mark.phase1
class TestMassInertiaCalculations:

    def test_total_mass_is_correct(self, updated_model, expected_values):
        """Total mass should equal sum of all body masses."""
        total_mass = sum(updated_model.body_mass)
        assert abs(total_mass - expected_values["total_mass_kg"]) < 0.001

    def test_trunk_mass_increased(self, updated_model, expected_values):
        """trunk_assembly mass should reflect Jetson + extra batteries + thermal partition + wiring."""
        trunk_id = updated_model.body("trunk_assembly").id
        trunk_mass = updated_model.body_mass[trunk_id]
        expected = expected_values["trunk_assembly_mass_kg"]
        assert abs(trunk_mass - expected) < 0.001

    def test_head_mass_decreased(self, updated_model, expected_values):
        """head_assembly mass should decrease by ~10g (Pi removed)."""
        head_id = updated_model.body("head_assembly").id
        head_mass = updated_model.body_mass[head_id]
        expected = expected_values["head_assembly_mass_kg"]
        assert abs(head_mass - expected) < 0.001

    def test_trunk_diaginertia_matches_fixture(self, updated_model, expected_values):
        """trunk_assembly principal inertia must match the fixture.

        The model carries a frame-correct fullinertia; MuJoCo's body_inertia
        holds its principal moments. Compared as sorted triples to be robust
        to principal-axis ordering.
        """
        trunk_id = updated_model.body("trunk_assembly").id
        model_inertia = np.sort(updated_model.body_inertia[trunk_id])
        expected = np.sort(expected_values["trunk_assembly_diaginertia"])
        assert np.allclose(model_inertia, expected, atol=1e-7), (
            f"trunk diaginertia {model_inertia} != fixture {expected}"
        )

    def test_head_diaginertia_matches_fixture(self, updated_model, expected_values):
        """head_assembly principal inertia must match the fixture (frame-correct)."""
        head_id = updated_model.body("head_assembly").id
        model_inertia = np.sort(updated_model.body_inertia[head_id])
        expected = np.sort(expected_values["head_assembly_diaginertia"])
        assert np.allclose(model_inertia, expected, atol=1e-7), (
            f"head diaginertia {model_inertia} != fixture {expected}"
        )

    def test_inertia_tensor_positive_definite(self, updated_model):
        """All diagonal inertia values must be positive."""
        for body_id in range(updated_model.nbody):
            inertia = updated_model.body_inertia[body_id]
            if np.sum(inertia) > 0:
                assert all(i >= 0 for i in inertia), f"Body {body_id} has negative inertia"

    def test_triangle_inequality_holds(self, updated_model):
        """Inertia tensor must satisfy triangle inequality: Ixx+Iyy >= Izz, etc."""
        for body_id in range(updated_model.nbody):
            I = updated_model.body_inertia[body_id]
            if np.sum(I) > 0:
                assert I[0] + I[1] >= I[2] - 1e-10
                assert I[0] + I[2] >= I[1] - 1e-10
                assert I[1] + I[2] >= I[0] - 1e-10

    def test_com_within_body_bounds(self, updated_model):
        """Center of mass should be within reasonable bounds of the body geometry."""
        trunk_id = updated_model.body("trunk_assembly").id
        com = updated_model.body_ipos[trunk_id]
        assert -0.15 < com[0] < 0.03, f"trunk CoM X out of bounds: {com[0]}"
        assert -0.06 < com[1] < 0.06, f"trunk CoM Y out of bounds: {com[1]}"
        assert -0.03 < com[2] < 0.10, f"trunk CoM Z out of bounds: {com[2]}"


@pytest.mark.phase1
class TestNoUnauthoredInertials:
    """Regression guard for PLANT-1 (the phantom 1.000 kg on the root).

    MuJoCo tolerates a body with no ``<inertial>`` and gives it mass 0. USD has
    no way to express that, so the MJCF->USD converter emits the *unauthored*
    sentinel and PhysX substitutes its own default: 1.000 kg with an isotropic
    0.4*m*r^2 (r=0.1 m) tensor. Nothing on disk looks wrong; the plant is simply
    heavier at load time than the file says.

    That is invisible to every MuJoCo-side check, which is why it survived from
    v1 to v5d. These tests read the MJCF as text, so they fail the moment a body
    is added without an ``<inertial>`` -- before it can reach a USD conversion.

    See docs/jetson-mod/known_issues.md#plant-1.
    """

    def test_every_body_declares_an_inertial(self):
        """No MJCF body may rely on the converter to invent its mass."""
        import os
        import xml.etree.ElementTree as ET

        mjcf = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "mini_bdx", "robots", "open_duck_mini_v2", "robot_motors.xml",
        )
        missing = [
            b.get("name")
            for b in ET.parse(mjcf).getroot().iter("body")
            if b.find("inertial") is None
        ]
        assert not missing, (
            f"bodies with no <inertial>: {missing}. PhysX will substitute a "
            "1.000 kg default for each -- author an <inertial> or merge the "
            "body into its parent (see known_issues.md PLANT-1)."
        )

    def test_articulation_root_carries_real_mass(self, updated_model):
        """The freejoint body must be a real link, not a massless wrapper.

        Body index 1 is the root (0 is ``world``). A near-zero root mass is the
        exact shape of the PLANT-1 defect and also gives PhysX a pathological
        root-to-child mass ratio.
        """
        root_mass = float(updated_model.body_mass[1])
        root_name = updated_model.body(1).name
        assert root_mass > 0.1, (
            f"articulation root {root_name!r} has mass {root_mass} kg; the root "
            "must carry real inertia (see known_issues.md PLANT-1)."
        )
