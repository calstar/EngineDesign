"""Integration tests for /api/experiment/press_test_fit and press_test_save_cv_line."""
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

VALID_ROW = {
    "label": "Run 1",
    "copv_p_start_psi": 4500.0, "copv_p_end_psi": 4490.0,
    "copv_t_start_s":   0.0,    "copv_t_end_s":   2.0,
    "tank_p_start_psi": 400.0,  "tank_p_end_psi": 405.0,
    "tank_t_start_s":   0.1,    "tank_t_end_s":   2.0,
}

BASE_PAYLOAD = {
    "rows": [VALID_ROW],
    "tank_volume_m3": 0.00915,
    "fill_fraction": 0.0,
    "copv_volume_L": 4.5,
    "T_copv_K": 300.0,
    "T_ull_K": 293.0,
    "reg_cv": 0.06,
    "reg_droop_coeff": 0.070,
    "reg_setpoint_psi": 450.0,
    "reg_initial_copv_psi": 4500.0,
}


class TestPressFeedFitEndpoint:
    def test_200_with_valid_payload(self):
        resp = client.post("/api/experiment/press_test_fit", json=BASE_PAYLOAD)
        assert resp.status_code == 200, resp.text

    def test_response_shape(self):
        resp = client.post("/api/experiment/press_test_fit", json=BASE_PAYLOAD)
        data = resp.json()
        for field in ("cv_line_fitted","cv_line_std","cv_reg","cv_eff",
                      "recommendation","rows","save_available"):
            assert field in data, f"Missing field: {field}"

    def test_cv_line_positive(self):
        data = client.post("/api/experiment/press_test_fit", json=BASE_PAYLOAD).json()
        assert data["cv_line_fitted"] > 0

    def test_cv_eff_less_than_reg_cv(self):
        data = client.post("/api/experiment/press_test_fit", json=BASE_PAYLOAD).json()
        # Series combination must be less than reg_cv
        assert data["cv_eff"] < data["cv_reg"]

    def test_multiple_rows_returns_all_results(self):
        payload = {**BASE_PAYLOAD, "rows": [VALID_ROW, VALID_ROW, VALID_ROW]}
        data = client.post("/api/experiment/press_test_fit", json=payload).json()
        assert len(data["rows"]) == 3

    def test_missing_tank_volume_returns_422(self):
        bad = {k: v for k, v in BASE_PAYLOAD.items() if k != "tank_volume_m3"}
        resp = client.post("/api/experiment/press_test_fit", json=bad)
        assert resp.status_code == 422

    def test_missing_rows_returns_422(self):
        bad = {**BASE_PAYLOAD, "rows": []}
        resp = client.post("/api/experiment/press_test_fit", json=bad)
        assert resp.status_code == 422

    def test_row_result_fields_present(self):
        data = client.post("/api/experiment/press_test_fit", json=BASE_PAYLOAD).json()
        row = data["rows"][0]
        for f in ("label","cv_line_estimate","mdot_copv_avg","mdot_tank_avg",
                  "cross_check_ratio","copv_dp_psi","tank_dp_psi"):
            assert f in row, f"Row missing field: {f}"

    def test_copv_dp_is_correct(self):
        data = client.post("/api/experiment/press_test_fit", json=BASE_PAYLOAD).json()
        row = data["rows"][0]
        # COPV drops from 4500 to 4490 = 10 psi
        assert row["copv_dp_psi"] == pytest.approx(10.0, abs=1.0)

    def test_tank_dp_is_correct(self):
        data = client.post("/api/experiment/press_test_fit", json=BASE_PAYLOAD).json()
        row = data["rows"][0]
        # Tank rises from 400 to 405 = 5 psi
        assert row["tank_dp_psi"] == pytest.approx(5.0, abs=1.0)


class TestPressFeedSaveEndpoint:
    def test_save_without_config_returns_400(self):
        resp = client.post("/api/experiment/press_test_save_cv_line",
                           json={"cv_line": 0.035})
        # Without a config loaded, should return 400
        assert resp.status_code == 400

    def test_save_negative_cv_returns_422(self):
        resp = client.post("/api/experiment/press_test_save_cv_line",
                           json={"cv_line": -0.01})
        assert resp.status_code == 422
