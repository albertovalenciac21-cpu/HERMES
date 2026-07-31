from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import (
    ParagraphStyle,
    getSampleStyleSheet,
)
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


DEFAULT_PREDICTION_REPORT_DIRECTORY = (
    "data/reports/predictions"
)


class AIReportError(RuntimeError):
    """Raised when a prediction report cannot be generated."""


def _safe_text(value: Any) -> str:
    if value is None:
        return "Not available"

    return str(value).replace("&", "&amp;").replace(
        "<",
        "&lt;",
    ).replace(
        ">",
        "&gt;",
    )


def _format_probability(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "Not available"


def _prediction_strength(
    margin: Any,
) -> tuple[str, str]:
    try:
        absolute_margin = abs(float(margin))
    except (TypeError, ValueError):
        return (
            "Not available",
            "The distance from the decision threshold "
            "could not be calculated.",
        )

    if absolute_margin < 0.05:
        return (
            "Borderline",
            "The predicted probability is close to the "
            "optimized decision threshold.",
        )

    if absolute_margin < 0.15:
        return (
            "Moderate margin",
            "The predicted probability is moderately "
            "separated from the optimized threshold.",
        )

    return (
        "Larger margin",
        "The predicted probability is more clearly "
        "separated from the optimized threshold.",
    )


def _section_heading(
    text: str,
    styles: dict[str, ParagraphStyle],
) -> Paragraph:
    return Paragraph(
        _safe_text(text),
        styles["SectionHeading"],
    )


def _key_value_table(
    rows: list[tuple[str, str]],
    styles: dict[str, ParagraphStyle],
) -> Table:
    data = [
        [
            Paragraph(
                f"<b>{_safe_text(label)}</b>",
                styles["Body"],
            ),
            Paragraph(
                _safe_text(value),
                styles["Body"],
            ),
        ]
        for label, value in rows
    ]

    table = Table(
        data,
        colWidths=[2.0 * inch, 4.6 * inch],
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.35,
                    colors.HexColor("#CBD5E1"),
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor("#F1F5F9"),
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    return table


def _shap_table(
    prediction: dict[str, Any],
    styles: dict[str, ParagraphStyle],
) -> Table | Paragraph:
    shap_explanation = prediction.get(
        "shap_explanation",
        {},
    )

    if not shap_explanation.get("available"):
        error = shap_explanation.get(
            "error",
            "SHAP explanation was not available.",
        )

        return Paragraph(
            _safe_text(error),
            styles["Body"],
        )

    contributions = shap_explanation.get(
        "top_contributions",
        [],
    )

    if not contributions:
        return Paragraph(
            "No SHAP contributions were returned.",
            styles["Body"],
        )

    rows = [
        [
            Paragraph("<b>Feature</b>", styles["Small"]),
            Paragraph("<b>Value</b>", styles["Small"]),
            Paragraph("<b>SHAP value</b>", styles["Small"]),
            Paragraph("<b>Direction</b>", styles["Small"]),
        ]
    ]

    for item in contributions:
        direction = str(
            item.get("direction", "neutral")
        ).replace("_", " ")

        rows.append(
            [
                Paragraph(
                    _safe_text(item.get("feature")),
                    styles["Small"],
                ),
                Paragraph(
                    _safe_text(
                        item.get("feature_value")
                    ),
                    styles["Small"],
                ),
                Paragraph(
                    _safe_text(
                        item.get("shap_value")
                    ),
                    styles["Small"],
                ),
                Paragraph(
                    _safe_text(direction),
                    styles["Small"],
                ),
            ]
        )

    table = Table(
        rows,
        colWidths=[
            2.5 * inch,
            1.1 * inch,
            1.2 * inch,
            1.5 * inch,
        ],
        repeatRows=1,
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#1E293B"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.35,
                    colors.HexColor("#CBD5E1"),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor("#F8FAFC"),
                    ],
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )

    return table


def _draw_page_footer(
    canvas: Any,
    document: Any,
) -> None:
    canvas.saveState()

    canvas.setFont(
        "Helvetica",
        8,
    )
    canvas.setFillColor(
        colors.HexColor("#64748B")
    )

    canvas.drawString(
        0.65 * inch,
        0.42 * inch,
        (
            "Project HERMES - "
            "Research Use Only"
        ),
    )

    canvas.drawRightString(
        7.85 * inch,
        0.42 * inch,
        f"Page {document.page}",
    )

    canvas.restoreState()


def generate_prediction_pdf(
    prediction: dict[str, Any],
    output_path: str | None = None,
) -> dict[str, Any]:
    """
    Generate a research prediction report as a PDF.
    """

    patient_id = str(
        prediction.get(
            "patient_id",
            "anonymous_patient",
        )
    ).strip()

    safe_patient_id = "".join(
        character
        if character.isalnum() or character in "-_"
        else "_"
        for character in patient_id
    )

    generated_at = datetime.now(
        timezone.utc
    )

    if output_path:
        resolved_path = Path(output_path)
    else:
        resolved_path = (
            Path(
                DEFAULT_PREDICTION_REPORT_DIRECTORY
            )
            / (
                f"{safe_patient_id}_"
                "trojan_horse_prediction_report.pdf"
            )
        )

    resolved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=23,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=10,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#475569"),
            spaceAfter=18,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#0F172A"),
            spaceBefore=12,
            spaceAfter=7,
        )
    )

    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#1E293B"),
        )
    )

    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#1E293B"),
        )
    )

    styles.add(
        ParagraphStyle(
            name="Disclaimer",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#991B1B"),
            backColor=colors.HexColor("#FEF2F2"),
            borderColor=colors.HexColor("#FCA5A5"),
            borderWidth=0.6,
            borderPadding=8,
            spaceBefore=10,
        )
    )

    document = SimpleDocTemplate(
        str(resolved_path),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.7 * inch,
        title=(
            "Project HERMES "
            "AI Prediction Report"
        ),
        author="Project HERMES",
    )

    probabilities = prediction.get(
        "probabilities",
        {},
    )

    strength_label, strength_note = (
        _prediction_strength(
            prediction.get(
                "threshold_margin"
            )
        )
    )

    model = prediction.get(
        "model",
        {},
    )

    input_summary = prediction.get(
        "input_summary",
        {},
    )

    story: list[Any] = [
        Paragraph(
            "Project HERMES",
            styles["ReportTitle"],
        ),
        Paragraph(
            (
                "AI Prediction Report - "
                "Research Prototype"
            ),
            styles["ReportSubtitle"],
        ),
        _section_heading(
            "Report Information",
            styles,
        ),
        _key_value_table(
            [
                (
                    "Patient ID",
                    patient_id,
                ),
                (
                    "Generated",
                    generated_at.strftime(
                        "%Y-%m-%d %H:%M UTC"
                    ),
                ),
                (
                    "Prediction source",
                    prediction.get(
                        "prediction_source",
                        {},
                    ).get(
                        "type",
                        "Submitted ML-ready features",
                    ),
                ),
            ],
            styles,
        ),
        _section_heading(
            "Prediction Summary",
            styles,
        ),
        _key_value_table(
            [
                (
                    "Predicted class",
                    prediction.get(
                        "predicted_class"
                    ),
                ),
                (
                    "Alive probability",
                    _format_probability(
                        probabilities.get("alive")
                    ),
                ),
                (
                    "Dead probability",
                    _format_probability(
                        probabilities.get("dead")
                    ),
                ),
                (
                    "Decision threshold",
                    prediction.get(
                        "decision_threshold"
                    ),
                ),
                (
                    "Threshold margin",
                    prediction.get(
                        "threshold_margin"
                    ),
                ),
                (
                    "Prediction margin category",
                    strength_label,
                ),
            ],
            styles,
        ),
        Spacer(1, 0.08 * inch),
        Paragraph(
            _safe_text(strength_note),
            styles["Body"],
        ),
        _section_heading(
            "Model Information",
            styles,
        ),
        _key_value_table(
            [
                (
                    "Model name",
                    model.get("model_name"),
                ),
                (
                    "Model type",
                    model.get("model_type"),
                ),
                (
                    "Optimization version",
                    model.get(
                        "optimization_version"
                    ),
                ),
                (
                    "Expected feature count",
                    input_summary.get(
                        "expected_feature_count"
                    ),
                ),
                (
                    "Supplied feature count",
                    input_summary.get(
                        "supplied_feature_count"
                    ),
                ),
                (
                    "Features defaulted to zero",
                    (
                        ", ".join(
                            input_summary.get(
                                "features_defaulted_to_zero",
                                [],
                            )
                        )
                        or "None"
                    ),
                ),
            ],
            styles,
        ),
        _section_heading(
            "SHAP Explanation",
            styles,
        ),
        Paragraph(
            (
                "Positive SHAP values move the model "
                "toward the positive class (Dead). "
                "Negative SHAP values move it toward "
                "the negative class (Alive)."
            ),
            styles["Body"],
        ),
        Spacer(1, 0.08 * inch),
        _shap_table(
            prediction,
            styles,
        ),
    ]

    observed_label = prediction.get(
        "observed_label"
    )

    if observed_label is not None:
        story.extend(
            [
                _section_heading(
                    "Development-Cohort Context",
                    styles,
                ),
                _key_value_table(
                    [
                        (
                            "Observed label in dataset",
                            observed_label,
                        ),
                    ],
                    styles,
                ),
                Spacer(1, 0.08 * inch),
                Paragraph(
                    _safe_text(
                        prediction.get(
                            "evaluation_warning",
                            (
                                "The patient may have "
                                "been used during model "
                                "development."
                            ),
                        )
                    ),
                    styles["Body"],
                ),
            ]
        )

    story.extend(
        [
            _section_heading(
                "Limitations",
                styles,
            ),
            Paragraph(
                (
                    "- The model predicts a binary vital-status "
                    "label and does not model time-to-event or "
                    "censoring.<br/>"
                    "- The current cohort is TCGA-BRCA and is not "
                    "yet restricted to confirmed TNBC cases.<br/>"
                    "- This report does not represent independent "
                    "external validation.<br/>"
                    "- The probability reflects the fitted model "
                    "and should not be interpreted as an individual "
                    "clinical prognosis."
                ),
                styles["Body"],
            ),
            Paragraph(
                (
                    "RESEARCH USE ONLY. This report is generated "
                    "by an experimental software prototype. It "
                    "must not be used for diagnosis, prognosis, "
                    "treatment selection, or other patient-care "
                    "decisions."
                ),
                styles["Disclaimer"],
            ),
        ]
    )

    try:
        document.build(
            story,
            onFirstPage=_draw_page_footer,
            onLaterPages=_draw_page_footer,
        )
    except Exception as exc:
        raise AIReportError(
            f"The PDF report could not be generated: {exc}"
        ) from exc

    return {
        "report_status": "complete",
        "patient_id": patient_id,
        "report_path": str(resolved_path),
        "generated_at_utc": (
            generated_at.isoformat()
        ),
        "research_use_only": True,
    }
