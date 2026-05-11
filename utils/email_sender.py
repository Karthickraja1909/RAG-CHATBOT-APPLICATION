"""
Email notification module for evaluation reports.
Sends evaluation results to configured recipients via SMTP.
All configuration from .env — no parameters needed.
"""

import json
import smtplib
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


def send_evaluation_report(report: dict, report_path: Optional[Path] = None) -> bool:
    """
    Send evaluation report via email to all configured recipients.

    Args:
        report: Combined evaluation report dictionary.
        report_path: Optional path to attach the full JSON report file.

    Returns:
        True if email sent successfully, False otherwise.
    """
    settings = get_settings()

    if not settings.email_enabled:
        logger.info("Email notifications disabled")
        return False

    if not settings.email_recipients_list:
        logger.warning("No email recipients configured")
        return False

    if not settings.email_sender or not settings.email_sender_password:
        logger.warning("Email sender credentials not configured")
        return False

    logger.info(
        f"Preparing email: from={settings.email_sender}, "
        f"to={settings.email_recipients_list}, "
        f"smtp={settings.email_smtp_host}:{settings.email_smtp_port}"
    )

    # Build email content
    try:
        subject = _build_subject(report, settings)
        html_body = _build_html_report(report)
    except Exception as e:
        logger.error(f"Failed to build email content: {type(e).__name__}: {e}")
        return False

    # Send to each recipient
    try:
        msg = MIMEMultipart("mixed")
        msg["From"] = settings.email_sender
        msg["To"] = ", ".join(settings.email_recipients_list)
        msg["Subject"] = subject

        # HTML body
        html_part = MIMEText(html_body, "html", "utf-8")
        msg.attach(html_part)

        # Attach JSON report file
        if report_path and report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                attachment = MIMEApplication(f.read().encode("utf-8"), _subtype="json")
                attachment.add_header(
                    "Content-Disposition", "attachment", filename=report_path.name
                )
                msg.attach(attachment)

        # Send via SMTP
        with smtplib.SMTP(settings.email_smtp_host, settings.email_smtp_port) as server:
            server.starttls()
            server.login(settings.email_sender, settings.email_sender_password)
            server.send_message(msg)

        logger.info(f"Email report sent to: {settings.email_recipients_list}")
        return True

    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP authentication failed. Check EMAIL_SENDER and EMAIL_SENDER_PASSWORD in .env")
        return False
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False


def _build_subject(report: dict, settings) -> str:
    """Build email subject line with pass/fail status."""
    summary = report.get("summary", {})
    pass_rate = summary.get("overall_pass_rate", 0.0)
    threshold = settings.eval_threshold
    status = "PASSED" if pass_rate >= threshold else "FAILED"
    timestamp = report.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))

    return f"{settings.email_subject_prefix} Evaluation {status} — {pass_rate:.0%} pass rate | {timestamp}"


def _build_html_report(report: dict) -> str:
    """Build HTML email body with evaluation results."""
    summary = report.get("summary", {})
    config = report.get("configuration", {})
    results = report.get("results", {})

    pass_rate = summary.get("overall_pass_rate", 0.0) or 0.0
    threshold = config.get("threshold", 0.7) or 0.7
    status_color = "#28a745" if pass_rate >= threshold else "#dc3545"
    status_text = "PASSED" if pass_rate >= threshold else "FAILED"

    total_samples = summary.get("total_samples", 0) or 0
    total_passed = summary.get("total_passed", 0) or 0
    total_failed = summary.get("total_failed", 0) or 0
    duration = summary.get("duration_seconds", 0) or 0

    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; border-radius: 8px; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #1a1a2e; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            h2 {{ color: #16213e; margin-top: 25px; }}
            .status {{ display: inline-block; padding: 8px 20px; border-radius: 4px; color: white; font-weight: bold; font-size: 18px; background: {status_color}; }}
            .metric-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            .metric-table th {{ background: #1a1a2e; color: white; padding: 10px; text-align: left; }}
            .metric-table td {{ padding: 8px 10px; border-bottom: 1px solid #eee; }}
            .metric-table tr:hover {{ background: #f8f9fa; }}
            .pass {{ color: #28a745; font-weight: bold; }}
            .fail {{ color: #dc3545; font-weight: bold; }}
            .config-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
            .config-table td {{ padding: 5px 10px; border-bottom: 1px solid #f0f0f0; }}
            .config-table td:first-child {{ font-weight: bold; color: #555; width: 200px; }}
            .footer {{ margin-top: 30px; padding-top: 15px; border-top: 1px solid #eee; color: #666; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>GenAI RAG System — LLM Evaluation Report</h1>

            <p><span class="status">{status_text}: {pass_rate:.1%} Pass Rate</span></p>

            <h2>Summary</h2>
            <table class="config-table">
                <tr><td>Overall Pass Rate</td><td>{pass_rate:.1%}</td></tr>
                <tr><td>Threshold</td><td>{threshold:.1%}</td></tr>
                <tr><td>Total Samples</td><td>{total_samples}</td></tr>
                <tr><td>Passed</td><td>{total_passed}</td></tr>
                <tr><td>Failed</td><td>{total_failed}</td></tr>
                <tr><td>Duration</td><td>{duration:.1f}s</td></tr>
            </table>

            <h2>Configuration</h2>
            <table class="config-table">
                <tr><td>Eval Model</td><td>{config.get('eval_model', 'N/A')}</td></tr>
                <tr><td>Frameworks</td><td>{', '.join(config.get('frameworks', []))}</td></tr>
                <tr><td>DeepEval Metrics</td><td>{', '.join(config.get('deepeval_metrics', []))}</td></tr>
                <tr><td>RAGAS Metrics</td><td>{', '.join(config.get('ragas_metrics', []))}</td></tr>
            </table>
    """

    # Per-framework results
    for framework, data in results.items():
        fw_pass_rate = data.get("pass_rate", 0.0) or 0.0
        fw_status = "pass" if fw_pass_rate >= threshold else "fail"

        html += f"""
            <h2>{framework.upper()} Results <span class="{fw_status}">({fw_pass_rate:.1%})</span></h2>
            <table class="metric-table">
                <thead>
                    <tr><th>Metric</th><th>Average Score</th><th>Min</th><th>Max</th><th>Pass Rate</th><th>Status</th></tr>
                </thead>
                <tbody>
        """

        metrics_summary = data.get("metrics_summary", {}) or {}
        for metric_name, metric_data in metrics_summary.items():
            if not isinstance(metric_data, dict):
                continue
            avg = metric_data.get("average_score", 0.0) or 0.0
            min_s = metric_data.get("min_score", 0.0) or 0.0
            max_s = metric_data.get("max_score", 0.0) or 0.0
            m_pass_rate = metric_data.get("pass_rate", 0.0) or 0.0
            m_status_class = "pass" if avg >= threshold else "fail"
            m_status_text = "PASS" if avg >= threshold else "FAIL"

            html += f"""
                    <tr>
                        <td>{metric_name}</td>
                        <td>{avg:.3f}</td>
                        <td>{min_s:.3f}</td>
                        <td>{max_s:.3f}</td>
                        <td>{m_pass_rate:.1%}</td>
                        <td class="{m_status_class}">{m_status_text}</td>
                    </tr>
            """

        html += """
                </tbody>
            </table>
        """

    # Per-sample details
    for framework, data in results.items():
        per_sample = data.get("per_sample_results", []) or []
        if per_sample:
            html += f"""
            <h2>{framework.upper()} — Per-Sample Details</h2>
            <table class="metric-table">
                <thead>
                    <tr><th>Sample ID</th><th>Avg Score</th><th>Status</th></tr>
                </thead>
                <tbody>
            """
            for sample in per_sample:
                s_passed = sample.get("overall_passed", False)
                s_class = "pass" if s_passed else "fail"
                s_text = "PASS" if s_passed else "FAIL"
                s_score = sample.get("average_score", 0.0) or 0.0
                html += f"""
                    <tr>
                        <td>{sample.get('sample_id', 'N/A')}</td>
                        <td>{s_score:.3f}</td>
                        <td class="{s_class}">{s_text}</td>
                    </tr>
                """
            html += """
                </tbody>
            </table>
            """

    html += f"""
            <div class="footer">
                <p>Generated by GenAI RAG System — Automated LLM Evaluation Pipeline</p>
                <p>Report ID: {report.get('report_id', 'N/A')} | Timestamp: {report.get('timestamp', 'N/A')}</p>
            </div>
        </div>
    </body>
    </html>
    """

    return html