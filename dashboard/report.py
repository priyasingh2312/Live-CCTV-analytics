"""PDF alert report for the dashboard (uses reportlab: pip install reportlab)."""
import io
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

MAX_ROWS = 500       # keep the PDF a sensible size; the CSV export has everything
MAX_PICS = 12
MAX_DAYS = 62        # rows in the "Alerts by day" table


def _table(data, widths, header=True):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
                  ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                  ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f5f8")])]
    t.setStyle(TableStyle(style))
    return t


def build_pdf(df, title, period, snap_dir, labels, with_snapshots=True):
    """df columns: timestamp, camera, detector, event, count, snapshot. Returns PDF bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm, title=f"{title} - alert report")
    st = getSampleStyleSheet()
    lab = lambda n: labels.get(n, n)
    out = [Paragraph(title, st["Title"]), Paragraph("Alert report", st["Heading2"]),
           Paragraph(f"Period: {period}<br/>Generated: {datetime.now():%d %b %Y %H:%M}<br/>"
                     f"Total alerts: {len(df)}", st["Normal"]), Spacer(1, 6 * mm)]

    if len(df):
        out.append(Paragraph("Summary by use case", st["Heading3"]))
        summ = df.groupby(["detector", "event"]).size().reset_index(name="alerts")
        rows = [["Use case", "Event", "Alerts"]] + [[lab(r.detector), r.event, int(r.alerts)]
                                                      for r in summ.itertuples()]
        out += [_table(rows, [80 * mm, 60 * mm, 30 * mm]), Spacer(1, 6 * mm)]

        days = df["timestamp"].dt.date
        if days.nunique() > 1:                       # a day-by-day table is only useful for more than one day
            out.append(Paragraph("Alerts by day", st["Heading3"]))
            names = sorted(df["detector"].unique(), key=lambda n: lab(n))
            small = ParagraphStyle("hdr", parent=st["BodyText"], fontSize=7, leading=8.5, textColor=colors.white,
                                   alignment=1)
            counts = df.assign(_day=days).groupby(["_day", "detector"]).size().unstack(fill_value=0)
            counts = counts.reindex(columns=names, fill_value=0).sort_index(ascending=False)
            rows = [[Paragraph("Date", small)] + [Paragraph(lab(n), small) for n in names] + [Paragraph("Total", small)]]
            for day, r in counts.head(MAX_DAYS).iterrows():
                rows.append([f"{day:%d %b %Y}"] + [int(r[n]) if int(r[n]) else "-" for n in names] + [int(r.sum())])
            tot = counts.sum()
            rows.append(["Total"] + [int(tot[n]) for n in names] + [int(tot.sum())])
            first, last = 26 * mm, 16 * mm
            mid = (180 * mm - first - last) / max(1, len(names))
            t = _table(rows, [first] + [mid] * len(names) + [last])
            t.setStyle(TableStyle([("ALIGN", (1, 1), (-1, -1), "CENTER"), ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                                   ("LINEABOVE", (0, -1), (-1, -1), 0.8, colors.HexColor("#1f3a5f"))]))
            out.append(t)
            if len(counts) > MAX_DAYS:
                out.append(Paragraph(f"Showing the latest {MAX_DAYS} of {len(counts)} days. The totals cover all days.",
                                     st["Italic"]))
            out.append(Spacer(1, 6 * mm))

        out.append(Paragraph("Alert list", st["Heading3"]))
        d = df.sort_values("timestamp", ascending=False).head(MAX_ROWS)
        rows = [["Time", "Camera", "Use case", "Event", "Count"]] + [
            [f"{r.timestamp:%d %b %Y %H:%M:%S}", r.camera or "-", lab(r.detector), r.event, int(r.count or 0)]
            for r in d.itertuples()]
        out.append(_table(rows, [38 * mm, 28 * mm, 55 * mm, 40 * mm, 15 * mm]))
        if len(df) > MAX_ROWS:
            out.append(Paragraph(f"Showing the latest {MAX_ROWS} of {len(df)} alerts. Use the CSV export for all.",
                                 st["Italic"]))

        if with_snapshots:
            pics = []
            for r in df.sort_values("timestamp", ascending=False).itertuples():
                p = Path(snap_dir) / str(r.snapshot)
                if p.exists():
                    pics.append((p, f"{r.timestamp:%d %b %H:%M} - {lab(r.detector)}: {r.event}"))
                if len(pics) >= MAX_PICS:
                    break
            if pics:
                out += [Spacer(1, 6 * mm), Paragraph("Latest snapshots", st["Heading3"])]
                cells = []
                for p, cap in pics:
                    try:
                        img = Image(str(p), width=58 * mm, height=38 * mm, kind="proportional")
                    except Exception:
                        continue
                    cells.append([img, Paragraph(cap, st["BodyText"])])
                grid = []
                for i in range(0, len(cells), 3):
                    row = cells[i:i + 3]
                    grid.append([c[0] for c in row] + [""] * (3 - len(row)))
                    grid.append([c[1] for c in row] + [""] * (3 - len(row)))
                t = Table(grid, colWidths=[60 * mm] * 3)
                t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
                out.append(t)
    else:
        out.append(Paragraph("No alerts in this period.", st["Normal"]))

    doc.build(out)
    return buf.getvalue()
