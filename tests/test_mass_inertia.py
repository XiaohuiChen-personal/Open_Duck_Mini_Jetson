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
