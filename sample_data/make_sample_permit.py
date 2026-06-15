"""Generate the bundled synthetic permit PDF.

This produces a realistic but entirely fictional environmental operating permit
so the repository runs with zero downloads and no copyright concern (Part B10).
The conditions are written in the regular, numbered style of real Title V air
and NPDES water permits, rich enough to exercise every verification check. The
unique permit number PTP-2026-0001 lets the Mock backend recognize the sample
and add its planted demonstration cases.

Run this only to regenerate sample_permit.pdf; the generated PDF is committed.

    python sample_data/make_sample_permit.py
"""

from __future__ import annotations

import os

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

OUTPUT = os.path.join(os.path.dirname(__file__), "sample_permit.pdf")

# Each condition is one paragraph. The leading number lets the ingest stage
# split the permit into one segment per condition.
TITLE = "SYNTHETIC ENVIRONMENTAL OPERATING PERMIT"
SUBTITLE = "FOR SOFTWARE TESTING ONLY - NOT A REAL PERMIT"

INTRO = (
    "This synthetic operating permit (Permit No. PTP-2026-0001) is issued to "
    "Crystal Valley Manufacturing, LLC, located at 1200 Industrial Parkway, for "
    "the sole purpose of testing the Permit-to-Proof software. All facilities, "
    "limits, and citations below are fictional. The permit combines Title V air "
    "emission conditions and NPDES water discharge conditions in the regular "
    "numbered style used by real permits."
)

SECTION_AIR = "SECTION 3 - AIR EMISSION LIMITS (TITLE V)"
AIR_CONDITIONS = [
    "Condition 3.1. Nitrogen oxides (NOx) emissions from Boiler Unit B-1 shall "
    "not exceed 30 ppm, corrected to 15 percent oxygen, on a 30-day rolling "
    "average. NOx shall be monitored continuously by a certified CEMS. "
    "(40 CFR 60.44c)",
    "Condition 3.2. Sulfur dioxide (SO2) emissions from Boiler Unit B-1 shall "
    "not exceed 0.30 lb/MMBtu on a 30-day rolling average, monitored "
    "continuously. (40 CFR 60.43c)",
    "Condition 3.3. Carbon monoxide (CO) emissions shall not exceed 100 ppm at "
    "the stack outlet, monitored continuously by CEMS. (Condition 3.3)",
    "Condition 3.4. Particulate matter (PM10) emissions shall not exceed 0.030 "
    "gr/dscf, demonstrated by an annual stack test conducted in accordance with "
    "Method 5. (40 CFR 60.42c)",
    "Condition 3.5. Volatile organic compounds (VOC) emissions from the coating "
    "line shall not exceed 25 tons/yr on a 12-month rolling total, with records "
    "updated monthly. (Condition 3.5)",
    "Condition 3.6. Visible emissions from any stack shall not exceed 20 % "
    "opacity, monitored continuously, except for one six-minute period per hour "
    "of up to 27 percent opacity. (40 CFR 60.42)",
]

SECTION_WATER = "SECTION 4 - WATER DISCHARGE LIMITS (NPDES)"
WATER_CONDITIONS = [
    "Condition 4.1. The pH of the discharge from Outfall 001 shall be maintained "
    "between 6.0 and 9.0 standard units at all times, monitored daily by grab "
    "sample. (Condition 4.1)",
    "Condition 4.2. Five-day biochemical oxygen demand (BOD5) shall not exceed "
    "30 mg/L as a monthly average, monitored weekly. (40 CFR 133.102)",
    "Condition 4.3. Total suspended solids (TSS) shall not exceed 30 mg/L as a "
    "monthly average, monitored weekly by composite sample. (40 CFR 133.102)",
    "Condition 4.4. The temperature of the discharge shall not exceed 32 deg C, "
    "monitored daily. (Condition 4.4)",
    "Condition 4.5. Fecal coliform bacteria shall not exceed 200 cfu/100 mL as a "
    "monthly geometric mean, monitored monthly. (Condition 4.5)",
    "Condition 4.6. The discharge flow from Outfall 001 shall not exceed 5.0 "
    "MGD, monitored continuously by a calibrated flow meter. (Condition 4.6)",
]

SECTION_REPORT = "SECTION 5 - MONITORING, REPORTING, AND RECORDKEEPING"
REPORT_CONDITIONS = [
    "Condition 5.1. The permittee shall submit Discharge Monitoring Reports "
    "(DMRs) within 28 days after the end of each monitoring period. "
    "(40 CFR 122.41)",
    "Condition 5.2. The permittee shall maintain all monitoring records, "
    "including calibration and maintenance logs, for a period of five years and "
    "make them available to the Administrator upon request. (40 CFR 122.41)",
    "Condition 5.3. The permittee shall report any exceedance of an emission or "
    "effluent limit by telephone within 24 hours and in writing within 5 days of "
    "becoming aware of the exceedance. (Condition 5.3)",
]


def build() -> str:
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=10.5, leading=15,
        alignment=TA_JUSTIFY, spaceAfter=10,
    )
    header = ParagraphStyle(
        "Header", parent=styles["Heading2"], fontSize=12, spaceBefore=14,
        spaceAfter=8,
    )
    title = ParagraphStyle(
        "Title", parent=styles["Title"], alignment=TA_CENTER, fontSize=18,
    )
    subtitle = ParagraphStyle(
        "Subtitle", parent=styles["Normal"], alignment=TA_CENTER, fontSize=10,
        textColor="#888888", spaceAfter=18,
    )

    story = [
        Paragraph(TITLE, title),
        Paragraph(SUBTITLE, subtitle),
        Paragraph(INTRO, body),
        Spacer(1, 0.15 * inch),
        Paragraph(SECTION_AIR, header),
    ]
    for cond in AIR_CONDITIONS:
        story.append(Paragraph(cond, body))

    story.append(PageBreak())
    story.append(Paragraph(SECTION_WATER, header))
    for cond in WATER_CONDITIONS:
        story.append(Paragraph(cond, body))

    story.append(PageBreak())
    story.append(Paragraph(SECTION_REPORT, header))
    for cond in REPORT_CONDITIONS:
        story.append(Paragraph(cond, body))

    doc = SimpleDocTemplate(
        OUTPUT, pagesize=LETTER,
        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
        topMargin=0.9 * inch, bottomMargin=0.9 * inch,
        title="Synthetic Permit PTP-2026-0001",
    )
    doc.build(story)
    return OUTPUT


if __name__ == "__main__":
    path = build()
    print("Wrote", path)
