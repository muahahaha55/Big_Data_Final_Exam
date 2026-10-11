"""Risk pipeline — Basel II analytics on scored portfolio.

1. Load scored predictions
2. Assign risk tiers
3. Compute Expected Loss (EL = PD × LGD × EAD)
4. Run stress test scenarios
5. Save portfolio report to data/08_reporting/

Run via: make risk
"""

from __future__ import annotations

import json
from datetime import datetime

from credit_risk.config import get_config, project_root
from credit_risk.models.registry import MLflowRegistry
from credit_risk.risk import (
    add_expected_loss,
    add_irb_capital,
    assign_risk_tier,
    compute_portfolio_metrics,
    run_stress_test,
    summarize_segmentation,
)
from credit_risk.utils import get_spark, stop_spark
from credit_risk.utils.logging import configure_logging, get_logger


def run_risk_pipeline() -> None:
    """End-to-end risk analytics on the test set predictions."""
    configure_logging()
    log = get_logger("pipeline.risk")
    cfg = get_config()
    log.info("risk_pipeline_started")

    spark = get_spark("risk-pipeline")

    try:
        # ── Load test set ─────────────────────────────────────
        test_path = project_root() / "data" / "05_model_input" / "test.parquet"
        test_df = spark.read.parquet(str(test_path))
        log.info("test_loaded", rows=test_df.count())

        # ── Score with Production model ───────────────────────
        registry = MLflowRegistry()
        model, info = registry.load_production_model()
        log.info("model_loaded", version=info.version)

        scored = model.transform(test_df)

        # Extract PD from probability column (vector → second element)
        from pyspark.ml.functions import vector_to_array
        scored = scored.withColumn("pd_score", vector_to_array("probability")[1])

        # ── Add EL components ─────────────────────────────────
        scored = add_expected_loss(
            scored,
            pd_column="pd_score",
            ead_column=cfg["risk"]["expected_loss"]["ead_column"],
        )

        # ── Assign risk tiers ─────────────────────────────────
        scored = assign_risk_tier(scored, pd_column="pd_score")

        # ── Basel II IRB capital, per obligor ─────────────────
        scored = add_irb_capital(scored, pd_column="pd_score")
        scored.cache()

        # ── Persist the scored portfolio (S4) ─────────────────
        # Downstream risk analytics (threshold curves, dashboard) read this
        # instead of reloading the model and re-scoring in Spark.
        keep = [
            c for c in [
                "SK_ID_CURR", cfg["data"]["target_column"], "NAME_CONTRACT_TYPE",
                "pd_score", "lgd", "ead", "expected_loss", "risk_tier",
                "basel_exposure_class", "basel_correlation", "irb_k", "irb_capital",
            ] if c in scored.columns
        ]
        scored_path = project_root() / "data" / "07_model_output" / "scored_test.parquet"
        scored_path.parent.mkdir(parents=True, exist_ok=True)
        scored.select(*keep).coalesce(1).write.mode("overwrite").parquet(str(scored_path))
        log.info("scored_portfolio_saved", path=str(scored_path), columns=len(keep))

        seg_summary = summarize_segmentation(scored)
        log.info("segmentation_done", **seg_summary.to_dict())

        # ── Portfolio metrics ─────────────────────────────────
        portfolio = compute_portfolio_metrics(
            scored,
            pd_column="pd_score",
            target_column=cfg["data"]["target_column"],
        )

        # ── Stress testing ────────────────────────────────────
        stress = run_stress_test(scored, pd_col="pd_score", lgd_col="lgd", ead_col="ead")

        # ── Save report ───────────────────────────────────────
        report = {
            **portfolio.to_dict(),
            "stress_test": stress.to_dict(),
            "model_version": info.version,
            "model_stage": info.stage,
            "generated_at": datetime.utcnow().isoformat(),
        }

        out_path = project_root() / "data" / "08_reporting" / "portfolio_risk_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(report, f, indent=2, default=str)

        log.info("risk_pipeline_completed", report=str(out_path))

        # ── Print headline numbers ────────────────────────────
        ov = report["portfolio_overview"]
        print("\n" + "=" * 60)
        print(f"  PORTFOLIO RISK REPORT  ({info.version})")
        print("=" * 60)
        print(f"  Customers:       {ov['total_customers']:>15,}")
        print(f"  Total EAD:       ${ov['total_ead_million']:>14,.1f}M")
        print(f"  Expected Loss:   ${ov['total_expected_loss_million']:>14,.2f}M")
        print(f"  EL rate:         {ov['portfolio_el_rate_pct']:>14.3f}%")
        print(f"  Avg PD:          {ov['avg_pd']:>15.4f}")
        print(f"  Assessment:      {ov['interpretation']}")
        cap = report["capital_requirement"]
        if cap.get("method") == "basel_ii_irb_retail_asrf":
            print("-" * 60)
            print(f"  IRB capital:     ${cap['irb_capital_requirement_million']:>14,.2f}M"
                  f"   ({cap['capital_to_ead_pct']:.3f}% of EAD)")
            print(f"  IRB RWA:         ${cap['irb_rwa_million']:>14,.2f}M"
                  f"   (density {cap['rwa_density_pct']:.2f}%)")
            rot = cap["rule_of_thumb_cross_check"]
            print(f"  [cross-check, assumption] 3xEL = ${rot['ul_million']:,.2f}M "
                  f"({rot['ul_over_irb_capital']}x IRB), "
                  f"4.5xEL = ${rot['ec_million']:,.2f}M ({rot['ec_over_irb_capital']}x IRB)")
        print("=" * 60)
        print("  Stress test results:")
        for s in report["stress_test"]["scenarios"]:
            print(f"    {s['scenario']:<18}EL=${s['total_el_million']:>10,.1f}M  "
                  f"cap=${s['irb_capital_requirement_million']:>10,.1f}M  "
                  f"PD×{s['pd_multiplier']:.1f}  LGD×{s['lgd_multiplier']:.1f}")
        print("=" * 60 + "\n")

        scored.unpersist()

    except Exception:
        log.exception("risk_pipeline_failed")
        raise
    finally:
        stop_spark()


if __name__ == "__main__":
    run_risk_pipeline()
