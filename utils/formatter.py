def format_patient_report_context(docs):
    """
    Convert retrieved patient report chunks into
    LLM-friendly context grouped by report.
    """

    reports = {}

    for doc in docs:

        report_name = doc.metadata.get(
            "report_name",
            "Unknown Report"
        )

        report_type = doc.metadata.get(
            "report_type",
            "Unknown Type"
        )

        upload_date = doc.metadata.get(
            "upload_date",
            "Unknown Date"
        )

        if report_name not in reports:

            reports[report_name] = {
                "report_type": report_type,
                "upload_date": upload_date,
                "chunks": []
            }

        reports[report_name]["chunks"].append(
            doc.page_content
        )

    context = ""

    for report_name, report in reports.items():

        context += f"""
Report:
{report_name}

Type:
{report["report_type"]}

Upload Date:
{report["upload_date"]}

Relevant Content:
"""

        for chunk in report["chunks"]:

            context += f"""
{chunk}
"""

        context += "\n---\n"

    return context.strip()