def format_patient_report_context(docs):
    """
    Convert retrieved patient report chunks into
    structured LLM-friendly context.

    Chunks belonging to the same report are grouped together.
    """

    if not docs:
        return ""

    reports = {}

    for doc in docs:

        report_id = doc.metadata.get(
            "report_id",
            doc.metadata.get(
                "report_name",
                "unknown_report"
            )
        )

        if report_id not in reports:
            reports[report_id] = {
                "report_name": doc.metadata.get(
                    "report_name",
                    "Unknown Report"
                ),
                "report_type": doc.metadata.get(
                    "report_type",
                    "Medical Report"
                ),
                "upload_date": doc.metadata.get(
                    "upload_date",
                    "Unknown Date"
                ),
                "chunks": []
            }

        reports[report_id]["chunks"].append(doc)

    context_parts = []

    for report in reports.values():

        # Sort chunks according to their original chunk number
        report["chunks"].sort(
            key=lambda doc: doc.metadata.get(
                "chunk",
                0
            )
        )

        content = "\n\n".join(
            doc.page_content
            for doc in report["chunks"]
        )

        context_parts.append(
            f"""
UPLOADED MEDICAL REPORT

Report Name:
{report["report_name"]}

Report Type:
{report["report_type"]}

Upload Date:
{report["upload_date"]}

Report Content:
{content}

================================
"""
        )

    return "\n".join(context_parts).strip()