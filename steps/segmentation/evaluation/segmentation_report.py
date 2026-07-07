import os
from typing import List, Optional, Tuple
import base64
import datetime as dt
import html
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class SegmentationReport:
    """Create a compact HTML report from an eval_all.csv table.

    The expected input is a wide table with one row per patient and metric
    columns such as dice_liver, hd95_liver, hd_liver.
    """

    DEFAULT_METRICS = {
        "dice_": "Dice",
        "hd95_": "HD95",
        "hd_": "HD",
        "assd_": "ASSD",
    }


    def __init__(
        self,
        csv_path: str,
        output_path: str,
        patient_id_column: str = "patient_id",
        include_total: bool = True,
        patient_limit: int = 5
    ):
        self.csv_path = Path(csv_path)
        self.output_path = Path(output_path)
        self.patient_id_column = patient_id_column
        self.include_total = include_total
        self.patient_limit = patient_limit

        self.df = self._read_csv()
        self.long_df = self._to_long_table()

    def create(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        sections = [
            self._html_header(),
            self._overview_section(),
            self._plots_section(),
            self._summary_section(),
            self._patient_ranking_section(),
            self._html_footer(),
        ]

        self.output_path.write_text("\n".join(sections), encoding="utf-8")

    def _read_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)


        unnamed_columns = [col for col in df.columns if col.startswith("Unnamed:")]
        if unnamed_columns:
            df = df.drop(columns=unnamed_columns)

        if self.patient_id_column not in df.columns:
            first_column = df.columns[0]
            df = df.rename(columns={first_column: self.patient_id_column})

        patient_ids = pd.to_numeric(
            df[self.patient_id_column],
            errors="coerce",
        )

        invalid_rows = patient_ids.isna()
        if invalid_rows.any():
            print(
                "Removed invalid patient IDs:",
                df.loc[invalid_rows, self.patient_id_column].tolist(),
            )

        df = df.loc[~invalid_rows].copy()

        df[self.patient_id_column] = (
            patient_ids.loc[~invalid_rows]
            .astype("int64")
            .astype(str)
        )

        return df

    def _metric_prefixes(self) -> List[Tuple[str, str]]:
        return sorted(
            self.DEFAULT_METRICS.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        )

    def _parse_metric_column(self, column: str) -> Optional[Tuple[str, str]]:
        for prefix, metric_name in self._metric_prefixes():
            if column.lower().startswith(prefix):
                structure = column[len(prefix):]
                if not structure:
                    return None
                if not self.include_total and structure.lower() == "tot":
                    return None
                return metric_name, structure
        return None

    def _to_long_table(self) -> pd.DataFrame:
        rows = []

        for column in self.df.columns:
            parsed = self._parse_metric_column(column)
            if parsed is None:
                continue

            metric_name, structure = parsed
            values = pd.to_numeric(self.df[column], errors="coerce")

            for patient_id, value in zip(self.df[self.patient_id_column], values):
                if pd.isna(value):
                    continue
                rows.append(
                    {
                        "patient_id": str(patient_id),
                        "metric": metric_name,
                        "structure": structure,
                        "value": float(value),
                    }
                )

        if not rows:
            raise ValueError(
                "No metric columns found. Expected columns like "
                "dice_liver, hd95_liver, hd_liver."
            )

        return pd.DataFrame(rows)

    def _summary_table(self) -> pd.DataFrame:
        grouped = self.long_df.groupby(["metric", "structure"])["value"]
        summary = grouped.agg(
            n="count",
            mean="mean",
            std="std",
            q1=lambda values: values.quantile(0.25),
            median="median",
            q3=lambda values: values.quantile(0.75),
            min="min",
            max="max",
        ).reset_index()
        return summary.sort_values(["metric", "structure"])

    def _plot_metric(self, metric: str) -> str:
        metric_df = self.long_df[
            (self.long_df["metric"] == metric)
            & (self.long_df["structure"].str.lower() != "tot")
            ]

        structures = sorted(metric_df["structure"].unique())
        data = [
            metric_df.loc[metric_df["structure"] == structure, "value"].values
            for structure in structures
        ]

        fig_width = max(10, min(24, 0.45 * len(structures)))
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        ax.boxplot(data, labels=[self._pretty_name(s) for s in structures], showfliers=False)
        ax.set_title(f"{metric} per structure")
        ax.set_ylabel(self._metric_ylabel(metric))
        ax.tick_params(axis="x", rotation=60)
        ax.grid(axis="y", alpha=0.3)

        if metric == "Dice":
            ax.set_ylim(0.0, 1.0)

        fig.tight_layout()
        return self._figure_to_base64(fig)

    def _figure_to_base64(self, fig) -> str:
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("ascii")

    def _ranked_patients(self, best: bool) -> pd.DataFrame:
        metric_df = self.long_df[
            (self.long_df["metric"] == "Dice")
            & (self.long_df["structure"].str.lower() == "tot")
            ]

        if metric_df.empty:
            raise ValueError("dice_tot not found.")

        table = metric_df[["patient_id", "value"]].rename(
            columns={"value": "dice_tot"}
        )

        return table.sort_values(
            "dice_tot",
            ascending=not best,
        ).head(self.patient_limit)

    def _worst_structure_values(self) -> pd.DataFrame:
        metric_df = self.long_df[
            (self.long_df["metric"] == "Dice")
            & (self.long_df["structure"].str.lower() != "tot")
            ]

        return metric_df.sort_values(
            "value",
            ascending=True,
        ).head(self.patient_limit)

    def _html_header(self) -> str:
        title = "Segmentation Report"
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 32px;
      color: #1f2933;
      background: #f8fafc;
    }}
    h1, h2, h3 {{ color: #111827; }}
    .meta, .note {{ color: #52606d; }}
    .section {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 20px;
      margin: 20px 0;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 14px;
      margin-top: 12px;
    }}
    th, td {{
      border-bottom: 1px solid #e5e7eb;
      padding: 7px 9px;
      text-align: right;
    }}
    th:first-child, td:first-child,
    th:nth-child(2), td:nth-child(2) {{
      text-align: left;
    }}
    img {{
      max-width: 100%;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      background: white;
    }}
  </style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">Created: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
"""

    def _html_footer(self) -> str:
        return "</body>\n</html>"

    def _overview_section(self) -> str:
        metrics = sorted(self.long_df["metric"].unique())
        structures = sorted(
            structure
            for structure in self.long_df["structure"].unique()
            if structure.lower() != "tot"
        )

        return f"""
<div class="section">
  <h2>Overview</h2>
  <p><strong>Input:</strong> {html.escape(str(self.csv_path))}</p>
  <p><strong>Patients:</strong> {self.df[self.patient_id_column].nunique()}</p>
  <p><strong>Structures:</strong> {len(structures)}</p>
  <p><strong>Metrics:</strong> {html.escape(", ".join(metrics))}</p>
</div>
"""

    def _plots_section(self) -> str:
        parts = ['<div class="section"><h2>Boxplots</h2>']
        for metric in sorted(self.long_df["metric"].unique()):
            image = self._plot_metric(metric)
            parts.append(f"<h3>{html.escape(metric)}</h3>")
            parts.append(f'<img src="data:image/png;base64,{image}" alt="{html.escape(metric)} boxplot">')
        parts.append("</div>")
        return "\n".join(parts)

    def _summary_section(self) -> str:
        summary = self._summary_table().copy()
        summary["structure"] = summary["structure"].map(self._pretty_name)

        for column in ["mean", "std", "q1", "median", "q3", "min", "max"]:
            summary[column] = summary[column].map(lambda value: f"{value:.3f}")

        return f"""
<div class="section">
  <h2>Summary Statistics</h2>
  {summary.to_html(index=False, escape=True)}
</div>
"""

    def _patient_ranking_section(self) -> str:
        parts = [
            '<div class="section"><h2>Best and Worst Patients</h2>',
            (
                '<p class="note">Patients are ranked by dice_tot, '
                'the mean Dice over all available structures.</p>'
            ),
        ]

        value_column = "dice_tot"

        best_patients = self._ranked_patients(best=True).copy()
        worst_patients = self._ranked_patients(best=False).copy()

        best_patients[value_column] = best_patients[value_column].map(
            lambda value: f"{value:.3f}"
        )
        worst_patients[value_column] = worst_patients[value_column].map(
            lambda value: f"{value:.3f}"
        )

        parts.append(f"<h3>Best {self.patient_limit} cases by Dice</h3>")
        parts.append(best_patients.to_html(index=False, escape=True))

        parts.append(f"<h3>Worst {self.patient_limit} cases by Dice</h3>")
        parts.append(worst_patients.to_html(index=False, escape=True))

        table = self._worst_structure_values().copy()
        table["structure"] = table["structure"].map(self._pretty_name)
        table["value"] = table["value"].map(lambda value: f"{value:.3f}")

        parts.append(
            f"<h3>Worst {self.patient_limit} structure Dice values</h3>"
        )
        parts.append(table.to_html(index=False, escape=True))

        parts.append("</div>")
        return "\n".join(parts)

    def _pretty_name(self, value: str) -> str:
        replacements = {
            "kidney_right": "Kidney right",
            "kidney_left": "Kidney left",
            "lung_right": "Lung right",
            "lung_left": "Lung left",
            "scapula_left": "Scapula left",
            "scapula_right": "Scapula right",
            "clavicula_left": "Clavicula left",
            "clavicula_right": "Clavicula right",
            "femur_left": "Femur left",
            "femur_right": "Femur right",
            "hip_left": "Hip left",
            "hip_right": "Hip right",
            "vertebrae_L5": "Vertebra L5",
            "vertebrae_L4": "Vertebra L4",
        }
        return replacements.get(value, value.replace("_", " ").capitalize())

    def _metric_ylabel(self, metric):
        if metric == "Dice":
            return "Dice"
        if metric in {"HD", "HD95", "ASSD"}:
            return f"{metric} [mm]"
        return metric

