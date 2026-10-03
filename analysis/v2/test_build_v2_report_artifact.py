"""Tests for the non-training v2 report-artifact builder."""

from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from analysis.v2.build_v2_report_artifact import (
    MAX_DATASET_ROWS,
    PROJECT_ROOT,
    TITLE,
    build_artifact,
    validate_artifact_shape,
    write_artifact,
)


class BuildV2ReportArtifactTests(unittest.TestCase):
    @staticmethod
    def _write_json(root: Path, relative: str, payload: dict) -> Path:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    @staticmethod
    def _write_text(root: Path, relative: str, text: str) -> Path:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _seed_required_report_inputs(self, root: Path) -> None:
        self._write_json(
            root,
            "artifacts/v2/diagnostics/model_selection_summary.json",
            {
                "development": {
                    "v1_raw_cosine": 0.10,
                    "v2_final_q1p2_cosine": 0.12,
                    "joint_minus_v1_raw": 0.02,
                    "sequence_increment_raw": 0.003,
                },
                "sealed_audit_descriptive_only": {
                    "v1_raw_cosine": 0.11,
                    "v2_final_q1p2_cosine": 0.13,
                },
            },
        )
        self._write_text(
            root,
            "artifacts/v2/diagnostics/model_selection_fold_scores.csv",
            "period,month_start,month_end,model,rows,cosine\n"
            "Dev1,23,34,v1 slow raw,10,0.10\n"
            "Dev1,23,34,v2 final q=1.2,10,0.12\n",
        )
        self._write_json(
            root,
            "artifacts/diagnostics/postmortem/postmortem_summary.json",
            {"public_leaderboard": {"score": 0.124, "rank": 1, "participants": 2}},
        )
        self._write_json(
            root,
            "artifacts/diagnostics/postmortem/domain_shift_summary.json",
            {"recent_train_test_domain_classifier": {"auc": 0.6}},
        )
        self._write_json(
            root,
            "artifacts/v2/diagnostics/deployment_stress_summary.json",
            {},
        )
        self._write_text(
            root,
            "artifacts/v2/diagnostics/deployment_stress_curve.csv",
            "horizon_months\n",
        )

    def test_current_workspace_builds_a_bounded_native_report(self) -> None:
        artifact = build_artifact(PROJECT_ROOT, generated_at="2026-08-24T00:00:00Z")
        validate_artifact_shape(artifact)

        self.assertEqual(artifact["surface"], "report")
        self.assertEqual(artifact["manifest"]["title"], TITLE)
        self.assertEqual(
            artifact["manifest"]["blocks"][0]["body"], f"# {TITLE}"
        )
        self.assertTrue(artifact["manifest"]["charts"])
        self.assertTrue(
            any(table["id"] == "fold_model_table" for table in artifact["manifest"]["tables"])
        )
        for rows in artifact["snapshot"]["datasets"].values():
            self.assertLessEqual(len(rows), MAX_DATASET_ROWS)

    def test_missing_inputs_are_explicitly_partial_not_silently_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = build_artifact(
                Path(directory), generated_at="2026-08-24T00:00:00Z"
            )
        self.assertEqual(artifact["snapshot"]["status"], "partial")
        issue_ids = {
            issue["id"] for issue in artifact["snapshot"]["accessIssues"]
        }
        self.assertIn("missing_selection", issue_ids)
        self.assertIn("missing_final_audit", issue_ids)
        validate_artifact_shape(artifact)

    def test_atomic_write_round_trips_exact_payload(self) -> None:
        artifact = build_artifact(PROJECT_ROOT, generated_at="2026-08-24T00:00:00Z")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact.json"
            write_artifact(artifact, output)
            reloaded = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(reloaded, artifact)

    def test_blend_pointer_and_nested_audit_schema_build_ready_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._seed_required_report_inputs(root)
            self._write_json(
                root,
                "artifacts/v2/diagnostics/frozen_blend_before_dev3.json",
                {
                    "blend_contract": {
                        "tabm_weight": 0.6,
                        "capacity_lightgbm_weight": 0.4,
                        "post_blend_transform": "sign(p) * abs(p)^1.1",
                    }
                },
            )
            self._write_json(
                root,
                "artifacts/v2/diagnostics/tabm_capacity_blend_comparison.json",
                {
                    "selected": {
                        "tabm_weight": 0.6,
                        "capacity_lightgbm_weight": 0.4,
                        "power_exponent": 1.1,
                        "screen_cosine": 0.14,
                    },
                    "dev3_audit": {"cosine": 0.15, "promotion_gate_passed": True},
                    "pooled_reporting": {"cosine": 0.145},
                },
            )
            self._write_json(
                root,
                "artifacts/v2/diagnostics/tabm_capacity_pooled_report.json",
                {"pooled_dev1_dev2_dev3_cosine": 0.145},
            )
            submission = self._write_text(
                root,
                "artifacts/v2/submissions/submission_final.csv",
                "sample_id,prediction\n1,0.25\n2,-0.25\n",
            )
            submission_hash = self._sha256(submission)
            generation_relative = "artifacts/v2/generations/generation-1/manifest.json"
            generation_manifest = self._write_json(
                root,
                generation_relative,
                {
                    "status": "complete",
                    "generation_id": "generation-1",
                    "pipeline_contract": {
                        "feature_sets": {
                            "capacity_lightgbm": "base_plus_sequence_all",
                            "tabm_mini": "multiscale_mechanics_scale",
                        },
                        "blend": {
                            "tabm_weight": 0.6,
                            "capacity_weight": 0.4,
                            "signed_power": 1.1,
                        },
                    },
                    "artifacts": {
                        "generation_submission": {"sha256": submission_hash}
                    },
                },
            )
            generation_hash = self._sha256(generation_manifest)
            self._write_json(
                root,
                "artifacts/v2/models/final_blend_pointer.json",
                {
                    "status": "complete",
                    "generation_id": "generation-1",
                    "generation_manifest": generation_relative,
                    "generation_manifest_sha256": generation_hash,
                    "submission_sha256": submission_hash,
                },
            )
            self._write_json(
                root,
                "artifacts/v2/diagnostics/submission_blend_audit.json",
                {
                    "status": "ready",
                    "generation_id": "generation-1",
                    "generation_manifest": generation_relative,
                    "generation_manifest_sha256": generation_hash,
                    "checks": {"replay": True, "mirrors": True},
                    "prediction": {"rows": 2, "rms": 1.0},
                    "submission_sha256": submission_hash,
                },
            )
            # A stale but internally plausible single-model publication must not
            # override the stable blend pointer/audit pair.
            self._write_json(
                root,
                "artifacts/v2/models/final_training_manifest.json",
                {
                    "status": "complete",
                    "test_rows": 999,
                    "outputs": {
                        "canonical_submission": {"sha256": submission_hash}
                    },
                },
            )
            self._write_json(
                root,
                "artifacts/v2/diagnostics/submission_audit.json",
                {
                    "status": "ready",
                    "checks": {"legacy": True},
                    "rows": 999,
                    "canonical_submission_sha256": submission_hash,
                },
            )

            artifact = build_artifact(root, generated_at="2026-08-24T00:00:00Z")

        self.assertEqual(artifact["snapshot"]["status"], "ready")
        self.assertEqual(artifact["snapshot"]["datasets"]["headline"][0]["submission_rows"], 2.0)
        self.assertEqual(
            artifact["snapshot"]["datasets"]["headline"][0]["submission_sha256"],
            submission_hash,
        )
        manifest_source = next(
            source
            for source in artifact["manifest"]["sources"]
            if source["id"] == "v2_final_manifest"
        )
        self.assertEqual(manifest_source["path"], generation_relative)
        readiness = next(
            block for block in artifact["manifest"]["blocks"] if block["id"] == "final_readiness"
        )
        self.assertIn("stable pointer", readiness["body"])
        validate_artifact_shape(artifact)

    def test_single_model_manifest_and_audit_remain_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._seed_required_report_inputs(root)
            submission = self._write_text(
                root,
                "artifacts/v2/submissions/submission_final.csv",
                "sample_id,prediction\n1,1.0\n",
            )
            submission_hash = self._sha256(submission)
            self._write_json(
                root,
                "artifacts/v2/models/final_training_manifest.json",
                {
                    "status": "complete",
                    "test_rows": 1,
                    "outputs": {
                        "canonical_submission": {"sha256": submission_hash}
                    },
                },
            )
            self._write_json(
                root,
                "artifacts/v2/diagnostics/submission_audit.json",
                {
                    "status": "ready",
                    "checks": {"round_trip": True},
                    "rows": 1,
                    "canonical_submission_sha256": submission_hash,
                },
            )

            artifact = build_artifact(root, generated_at="2026-08-24T00:00:00Z")

        self.assertEqual(artifact["snapshot"]["status"], "ready")
        self.assertEqual(artifact["snapshot"]["datasets"]["headline"][0]["submission_rows"], 1.0)
        validate_artifact_shape(artifact)


if __name__ == "__main__":
    unittest.main()
