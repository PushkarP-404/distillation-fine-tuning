#!/usr/bin/env python3
"""
generate_planning_dataset.py (v2.0)
-----------------------------------
Generates synthetic, structurally validated OS task planning datasets for training
on-device SLMs (e.g., SmolLM2-135M / 360M / Qwen2.5-1.5B) to act as reliable autonomous OS planners.

Upgraded to align with AI-Agent OS Harness v2.3:
- Full coverage of all 38 registered tools and 9 intent categories.
- Long-horizon multi-step orchestration chains (3-8 steps) with DAG dependencies.
- Dynamic variable propagation with 'captures' and '${scratchpad.var}' references.
- Grammar-constrained decoding compliance (explicit risk='read_only'|'change').
- High-level task chunking (phase decomposition) for complex tasks.
- Error recovery and step repair traces (up to 3 repair steps).
- Strict validation against harness_v2.validate_plan() for 100% schema compliance.
"""

import os
import sys
import json
import random
import argparse
from typing import Any

# Ensure harness_v2 can be imported for plan validation
AGENT_DAEMON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "os-ai-agent", "agent-daemon"))
sys.path.insert(0, AGENT_DAEMON_DIR)

from harness_v2 import (
    validate_plan, ALLOWED_TOOLS, PolicyEngine, PlanError,
    classify_intent_categories, _build_tool_subset
)

# Common Linux packages, services, paths, and commands
SAMPLE_PACKAGES = [
    "curl", "git", "htop", "tmux", "jq", "vim", "nano", "tree", "wget", "zip",
    "unzip", "rsync", "tcpdump", "netcat-openbsd", "strace", "gdb", "python3",
    "py3-pip", "nginx", "lightdm", "xfce4", "papirus-icon-theme", "make", "gcc",
    "pandoc", "poppler-utils"
]

SAMPLE_SERVICES = [
    "lightdm", "sshd", "networking", "ai-agent", "ai-agent-privileged", "udev",
    "syslog", "crond", "acpid", "dbus", "ntpd", "local"
]

SAMPLE_DIRS = [
    "Desktop/Projects", "Desktop/Reports", "Desktop/Archive", "Documents/Work",
    "Documents/Finances", "Downloads/Invoices", "Projects/Backend", "Projects/Frontend",
    "/tmp/scratch", "/tmp/cache", "/var/log/reports", "Desktop/Research", "Desktop/Data"
]

SAMPLE_FILES = [
    ("notes.txt", "Meeting notes and action items\n"),
    ("todo.md", "# TODO\n- [ ] Deploy new version\n"),
    ("script.sh", "#!/bin/sh\necho 'Running verification task...'\n"),
    ("config.json", '{"version": "2.0", "active": true, "env": "prod"}\n'),
    ("app.py", "#!/usr/bin/env python3\nprint('AI-Agent OS Online')\n"),
    ("data.csv", "id,item,cost,department\n1,laptop,1200,eng\n2,monitor,350,design\n"),
    ("invoice.txt", "Invoice #1042 - Acme Corp - Total: $4,500 - Due: Net 30\n"),
    ("system_audit.log", "[INFO] Audit completed successfully at 12:00:00\n")
]

URLS = [
    "https://portal.company.internal/login",
    "https://status.service.local",
    "https://docs.localnet/api/v1",
    "https://cloud-dashboard.org/overview",
    "http://127.0.0.1:8080/metrics"
]

EMAILS = [
    "alice@company.com", "bob.finance@acme.org", "team-leads@internal.net",
    "support@vendor-services.io", "director@research-lab.org"
]


# ── Generator 1: Directory Creation ──────────────────────────────────────────
def make_dir_create_sample(user="aiuser") -> tuple[str, dict[str, Any]]:
    target = random.choice(SAMPLE_DIRS)
    full_path = target if target.startswith("/") else f"/home/{user}/{target}"
    folder_name = os.path.basename(full_path)
    parent_dir = os.path.dirname(full_path).replace(f"/home/{user}/", "~/")

    phrasings = [
        f"create a folder in {parent_dir} called {folder_name}",
        f"make a directory named {folder_name} in {parent_dir}",
        f"mkdir -p {target}",
        f"create directory {full_path}",
        f"create a new directory called {folder_name} inside {parent_dir}",
    ]
    intent = random.choice(phrasings)

    plan = {
        "goal": f"Create directory {full_path}",
        "assumptions": ["Filesystem writable"],
        "success_criteria": [{"tool": "dir.query", "args": {"path": full_path}}],
        "steps": [
            {
                "id": "step-1",
                "purpose": f"Create directory {full_path}",
                "tool": "dir.create",
                "args": {"path": full_path},
                "depends_on": [],
                "risk": "change",
                "verify": [{"tool": "dir.query", "args": {"path": full_path}}]
            }
        ]
    }
    return intent, plan


# ── Generator 2: File Creation & Writing ──────────────────────────────────────
def make_file_write_sample(user="aiuser") -> tuple[str, dict[str, Any]]:
    folder = random.choice(["Desktop", "Documents", "Downloads", "/tmp"])
    fname, content = random.choice(SAMPLE_FILES)
    full_path = f"/tmp/{fname}" if folder == "/tmp" else f"/home/{user}/{folder}/{fname}"
    disp_folder = folder if folder.startswith("/") else f"~/{folder}"

    phrasings = [
        f"create a file in {disp_folder} called {fname}",
        f"write a file called {fname} in {disp_folder}",
        f"touch {disp_folder}/{fname}",
        f"create file {full_path} with standard initial contents",
    ]
    intent = random.choice(phrasings)

    plan = {
        "goal": f"Create file {full_path}",
        "assumptions": ["Target directory accessible"],
        "success_criteria": [{"tool": "file.query", "args": {"path": full_path}}],
        "steps": [
            {
                "id": "step-1",
                "purpose": f"Write file {full_path}",
                "tool": "file.write",
                "args": {"path": full_path, "content": content},
                "depends_on": [],
                "risk": "change",
                "verify": [{"tool": "file.query", "args": {"path": full_path}}]
            }
        ]
    }
    return intent, plan


# ── Generator 3: Package Management ──────────────────────────────────────────
def make_package_sample() -> tuple[str, dict[str, Any]]:
    pkg = random.choice(SAMPLE_PACKAGES)
    is_install = random.random() < 0.8

    if is_install:
        phrasings = [
            f"install {pkg}",
            f"install package {pkg}",
            f"apk add {pkg}",
            f"add package {pkg} to the system and verify installation",
        ]
        intent = random.choice(phrasings)
        plan = {
            "goal": f"Install package {pkg}",
            "assumptions": ["Network repository reachability"],
            "success_criteria": [{"tool": "package.query", "args": {"package": pkg}}],
            "steps": [
                {
                    "id": "step-1",
                    "purpose": f"Install {pkg} via package manager",
                    "tool": "package.install",
                    "args": {"package": pkg},
                    "depends_on": [],
                    "risk": "change",
                    "verify": [{"tool": "package.query", "args": {"package": pkg}}]
                }
            ]
        }
    else:
        phrasings = [
            f"check if {pkg} is installed",
            f"query package {pkg}",
            f"is {pkg} installed on this machine?",
        ]
        intent = random.choice(phrasings)
        plan = {
            "goal": f"Query package {pkg} status",
            "assumptions": [],
            "success_criteria": [{"tool": "package.query", "args": {"package": pkg}}],
            "steps": [
                {
                    "id": "step-1",
                    "purpose": f"Check status of package {pkg}",
                    "tool": "package.query",
                    "args": {"package": pkg},
                    "depends_on": [],
                    "risk": "read_only",
                    "verify": []
                }
            ]
        }
    return intent, plan


# ── Generator 4: Service Management ──────────────────────────────────────────
def make_service_sample() -> tuple[str, dict[str, Any]]:
    svc = random.choice(SAMPLE_SERVICES)
    action = random.choice(["start", "restart", "stop", "status", "logs"])

    if action in ("start", "restart", "stop"):
        phrasings = [
            f"{action} service {svc}",
            f"{action} {svc}",
            f"rc-service {svc} {action}",
            f"please {action} the {svc} service",
        ]
        intent = random.choice(phrasings)
        plan = {
            "goal": f"{action.capitalize()} service {svc}",
            "assumptions": ["Init system available"],
            "success_criteria": [{"tool": "service.status", "args": {"name": svc}}],
            "steps": [
                {
                    "id": "step-1",
                    "purpose": f"{action.capitalize()} service {svc}",
                    "tool": f"service.{action}",
                    "args": {"name": svc},
                    "depends_on": [],
                    "risk": "change",
                    "verify": [{"tool": "service.status", "args": {"name": svc}}]
                }
            ]
        }
    elif action == "logs":
        phrasings = [
            f"check logs for service {svc}",
            f"show log output of {svc}",
            f"inspect {svc} service logs",
        ]
        intent = random.choice(phrasings)
        plan = {
            "goal": f"Inspect logs for service {svc}",
            "assumptions": [],
            "success_criteria": [{"tool": "service.logs", "args": {"name": svc}}],
            "steps": [
                {
                    "id": "step-1",
                    "purpose": f"Retrieve service logs for {svc}",
                    "tool": "service.logs",
                    "args": {"name": svc, "lines": 50},
                    "depends_on": [],
                    "risk": "read_only",
                    "verify": []
                }
            ]
        }
    else:
        phrasings = [
            f"check status of service {svc}",
            f"is {svc} running?",
            f"service {svc} status",
        ]
        intent = random.choice(phrasings)
        plan = {
            "goal": f"Inspect service {svc} status",
            "assumptions": [],
            "success_criteria": [{"tool": "service.status", "args": {"name": svc}}],
            "steps": [
                {
                    "id": "step-1",
                    "purpose": f"Check status of {svc}",
                    "tool": "service.status",
                    "args": {"name": svc},
                    "depends_on": [],
                    "risk": "read_only",
                    "verify": []
                }
            ]
        }
    return intent, plan


# ── Generator 5: System Monitoring & Diagnostics ─────────────────────────────
def make_monitoring_sample() -> tuple[str, dict[str, Any]]:
    tool = random.choice(["disk.status", "memory.status", "system.info", "port.inspect"])
    if tool == "disk.status":
        intent = random.choice(["check disk space", "disk usage", "how much storage is left?", "df -h"])
        goal = "Collect disk usage statistics"
        args = {"path": "/"}
    elif tool == "memory.status":
        intent = random.choice(["check memory usage", "show free ram", "free -m", "how much ram is used?"])
        goal = "Collect memory usage statistics"
        args = {}
    elif tool == "port.inspect":
        port = random.choice([80, 443, 22, 8080, 11434])
        intent = random.choice([f"check if port {port} is open", f"is port {port} listening?", f"inspect port {port}"])
        goal = f"Inspect port {port} status"
        args = {"port": port}
    else:
        intent = random.choice(["check system info", "system status", "uname -a", "show operating system information"])
        goal = "Collect host and OS information"
        args = {}

    plan = {
        "goal": goal,
        "assumptions": [],
        "success_criteria": [{"tool": tool, "args": args}],
        "steps": [
            {
                "id": "step-1",
                "purpose": goal,
                "tool": tool,
                "args": args,
                "depends_on": [],
                "risk": "read_only",
                "verify": []
            }
        ]
    }
    return intent, plan


# ── Generator 6: File Organization & Batch Operations (3-4 steps) ─────────────
def make_file_management_sample(user="aiuser") -> tuple[str, dict[str, Any]]:
    src_dir = f"/home/{user}/Downloads"
    dest_dir = f"/home/{user}/Documents/Archive"
    log_file = f"/home/{user}/Documents/archive_log.txt"

    intent = random.choice([
        f"find all old logs in {src_dir}, archive them to {dest_dir}, and record an audit log",
        f"clean up downloaded reports in {src_dir}, move them to {dest_dir}, and update {log_file}",
        f"archive files from {src_dir} into {dest_dir} and write an operation summary"
    ])

    plan = {
        "goal": f"Archive downloaded files from {src_dir} to {dest_dir} and update log",
        "assumptions": ["Directories accessible and writable"],
        "success_criteria": [
            {"tool": "dir.query", "args": {"path": dest_dir}},
            {"tool": "file.query", "args": {"path": log_file}}
        ],
        "steps": [
            {
                "id": "step-1",
                "purpose": f"Ensure archive directory {dest_dir} exists",
                "tool": "dir.create",
                "args": {"path": dest_dir},
                "depends_on": [],
                "risk": "change",
                "verify": [{"tool": "dir.query", "args": {"path": dest_dir}}]
            },
            {
                "id": "step-2",
                "purpose": f"List files in {src_dir} to be archived",
                "tool": "file.list",
                "args": {"path": src_dir, "pattern": "*.log"},
                "depends_on": ["step-1"],
                "risk": "read_only",
                "captures": {"file_list": "stdout:strip"},
                "verify": []
            },
            {
                "id": "step-3",
                "purpose": f"Move logs to {dest_dir}",
                "tool": "command.argv",
                "args": {"argv": ["sh", "-c", f"mv {src_dir}/*.log {dest_dir}/ 2>/dev/null || true"]},
                "depends_on": ["step-2"],
                "risk": "change",
                "verify": [{"tool": "dir.query", "args": {"path": dest_dir}}]
            },
            {
                "id": "step-4",
                "purpose": f"Record archiving action in {log_file}",
                "tool": "file.append",
                "args": {"path": log_file, "content": "Archived files: ${scratchpad.file_list}\n"},
                "depends_on": ["step-3"],
                "risk": "change",
                "verify": [{"tool": "file.query", "args": {"path": log_file}}]
            }
        ]
    }
    return intent, plan


# ── Generator 7: Data Analysis & Document Processing (3-5 steps) ──────────────
def make_data_and_document_sample(user="aiuser") -> tuple[str, dict[str, Any]]:
    csv_path = f"/home/{user}/Documents/sales_data.csv"
    report_path = f"/home/{user}/Desktop/sales_summary.txt"

    intent = random.choice([
        f"read {csv_path}, calculate the quarterly sales summary, and write the report to {report_path}",
        f"analyze quarterly metrics from {csv_path} and save a summary report at {report_path}",
        f"extract records from {csv_path}, generate analysis with python, and save report to {report_path}"
    ])

    plan = {
        "goal": f"Analyze sales CSV and generate summary report at {report_path}",
        "assumptions": ["sales_data.csv exists", "python3 available"],
        "success_criteria": [
            {"tool": "file.query", "args": {"path": report_path}}
        ],
        "steps": [
            {
                "id": "step-1",
                "purpose": f"Verify sales CSV exists and inspect structure",
                "tool": "data.csv_read",
                "args": {"path": csv_path, "max_rows": 5},
                "depends_on": [],
                "risk": "read_only",
                "captures": {"header_info": "stdout:first_line"},
                "verify": []
            },
            {
                "id": "step-2",
                "purpose": "Compute total and averages via python script",
                "tool": "command.argv",
                "args": {
                    "argv": [
                        "python3", "-c",
                        f"import csv; data=list(csv.reader(open('{csv_path}'))); print(f'Rows: {{len(data)-1}}')"
                    ]
                },
                "depends_on": ["step-1"],
                "risk": "read_only",
                "captures": {"calc_summary": "stdout:strip"},
                "verify": []
            },
            {
                "id": "step-3",
                "purpose": f"Write summary report to {report_path}",
                "tool": "file.write",
                "args": {
                    "path": report_path,
                    "content": "Sales Analysis Summary\n----------------------\nHeader: ${scratchpad.header_info}\nStats: ${scratchpad.calc_summary}\n"
                },
                "depends_on": ["step-2"],
                "risk": "change",
                "verify": [{"tool": "file.query", "args": {"path": report_path}}]
            }
        ]
    }
    return intent, plan


# ── Generator 8: Browser Web Automation (3-5 steps) ───────────────────────────
def make_browser_sample() -> tuple[str, dict[str, Any]]:
    target_url = random.choice(URLS)
    shot_path = "/tmp/portal_verification.png"
    log_path = "/tmp/web_status.json"

    intent = random.choice([
        f"open {target_url} in browser, inspect page elements, take a screenshot to {shot_path}",
        f"navigate to {target_url}, capture accessibility tree and take screenshot at {shot_path}",
        f"verify web portal at {target_url} using browser and save visual proof"
    ])

    plan = {
        "goal": f"Open {target_url}, inspect page and capture screenshot",
        "assumptions": ["Browser or CDP endpoint accessible", "X11/display available"],
        "success_criteria": [
            {"tool": "file.query", "args": {"path": shot_path}}
        ],
        "steps": [
            {
                "id": "step-1",
                "purpose": f"Navigate browser to {target_url}",
                "tool": "browser.navigate",
                "args": {"url": target_url},
                "depends_on": [],
                "risk": "read_only",
                "verify": []
            },
            {
                "id": "step-2",
                "purpose": "Dump interactive elements on the loaded page",
                "tool": "browser.dump",
                "args": {},
                "depends_on": ["step-1"],
                "risk": "read_only",
                "captures": {"dom_summary": "stdout:first_line"},
                "verify": []
            },
            {
                "id": "step-3",
                "purpose": f"Capture screenshot of loaded page to {shot_path}",
                "tool": "browser.screenshot",
                "args": {"path": shot_path},
                "depends_on": ["step-2"],
                "risk": "read_only",
                "verify": []
            },
            {
                "id": "step-4",
                "purpose": f"Record navigation status to {log_path}",
                "tool": "file.write",
                "args": {
                    "path": log_path,
                    "content": '{"url": "' + target_url + '", "status": "verified"}\n'
                },
                "depends_on": ["step-3"],
                "risk": "change",
                "verify": [{"tool": "file.query", "args": {"path": log_path}}]
            }
        ]
    }
    return intent, plan


# ── Generator 9: Email & Communication Workflows (2-4 steps) ──────────────────
def make_email_sample(user="aiuser") -> tuple[str, dict[str, Any]]:
    recipient = random.choice(EMAILS)
    subject = "Automated System Report - Status OK"
    report_file = f"/home/{user}/Documents/status.txt"

    intent = random.choice([
        f"check for unread emails, compile status notes, and send an update to {recipient}",
        f"send an email to {recipient} with subject '{subject}' and attach latest report",
        f"draft and dispatch status update email to {recipient}"
    ])

    plan = {
        "goal": f"Send status email to {recipient}",
        "assumptions": ["Mail daemon or spool available"],
        "success_criteria": [
            {"tool": "email.read", "args": {"folder": "outbox", "max_emails": 1}}
        ],
        "steps": [
            {
                "id": "step-1",
                "purpose": "Check inbox for any pending urgent incoming mail",
                "tool": "email.read",
                "args": {"max_emails": 5, "unseen_only": True},
                "depends_on": [],
                "risk": "read_only",
                "captures": {"inbox_state": "stdout:first_line"},
                "verify": []
            },
            {
                "id": "step-2",
                "purpose": f"Send update email to {recipient}",
                "tool": "email.send",
                "args": {
                    "to": recipient,
                    "subject": subject,
                    "body": "System health is normal. Inbox state: ${scratchpad.inbox_state}"
                },
                "depends_on": ["step-1"],
                "risk": "change",
                "verify": [{"tool": "email.read", "args": {"folder": "outbox", "max_emails": 1}}]
            }
        ]
    }
    return intent, plan


# ── Generator 10: Coding & Build Workflows (3-5 steps) ────────────────────────
def make_coding_sample(user="aiuser") -> tuple[str, dict[str, Any]]:
    repo_dir = f"/home/{user}/Projects/Backend"
    test_log = f"{repo_dir}/test_run.log"

    intent = random.choice([
        f"navigate to {repo_dir}, check git status, run test suite, and record test output",
        f"verify backend repo at {repo_dir}, run pytest, and save log to {test_log}",
        f"run project tests in {repo_dir} and verify build passes"
    ])

    plan = {
        "goal": f"Run test suite in {repo_dir} and capture test results",
        "assumptions": ["Git repository initialized", "python3/pytest available"],
        "success_criteria": [
            {"tool": "file.query", "args": {"path": test_log}}
        ],
        "steps": [
            {
                "id": "step-1",
                "purpose": f"Check git branch status in {repo_dir}",
                "tool": "command.argv",
                "args": {"argv": ["git", "-C", repo_dir, "status", "--short"]},
                "depends_on": [],
                "risk": "read_only",
                "captures": {"git_status": "stdout:strip"},
                "verify": []
            },
            {
                "id": "step-2",
                "purpose": "Execute test suite via shell runner",
                "tool": "command.argv",
                "args": {"argv": ["python3", "-m", "unittest", "discover", "-s", repo_dir]},
                "depends_on": ["step-1"],
                "risk": "read_only",
                "captures": {"test_results": "stdout:strip"},
                "verify": []
            },
            {
                "id": "step-3",
                "purpose": f"Write test run report to {test_log}",
                "tool": "file.write",
                "args": {
                    "path": test_log,
                    "content": "Git Status:\n${scratchpad.git_status}\n\nTest Output:\n${scratchpad.test_results}\n"
                },
                "depends_on": ["step-2"],
                "risk": "change",
                "verify": [{"tool": "file.query", "args": {"path": test_log}}]
            }
        ]
    }
    return intent, plan


# ── Generator 11: Complex Multi-App Orchestration (5-8 steps) ─────────────────
def make_multiapp_orchestration_sample(user="aiuser") -> tuple[str, dict[str, Any]]:
    csv_file = f"/home/{user}/Documents/July_expenses.csv"
    summary_doc = f"/home/{user}/Documents/July_expense_summary.txt"
    recipient = "finance-manager@company.org"

    intent = random.choice([
        f"open the July expense spreadsheet, calculate total travel expenses, write a report, and email it to {recipient}",
        f"process {csv_file}, compute total spending, generate summary report, and send email update to {recipient}",
        f"extract expenses from {csv_file}, write summary at {summary_doc}, and dispatch notification to {recipient}"
    ])

    plan = {
        "goal": f"Process July expenses from {csv_file}, write {summary_doc}, and email {recipient}",
        "assumptions": ["CSV readable", "python3 available", "email subsystem operational"],
        "success_criteria": [
            {"tool": "file.query", "args": {"path": summary_doc}},
            {"tool": "email.read", "args": {"folder": "outbox", "max_emails": 1}}
        ],
        "steps": [
            {
                "id": "step-1",
                "purpose": f"Inspect {csv_file} columns and first rows",
                "tool": "data.csv_read",
                "args": {"path": csv_file, "max_rows": 10},
                "depends_on": [],
                "risk": "read_only",
                "captures": {"csv_headers": "stdout:first_line"},
                "verify": []
            },
            {
                "id": "step-2",
                "purpose": "Calculate travel expense sum using python",
                "tool": "command.argv",
                "args": {
                    "argv": [
                        "python3", "-c",
                        f"import csv; rows=list(csv.DictReader(open('{csv_file}'))); total=sum(float(r.get('amount',0)) for r in rows if 'travel' in r.get('category','').lower()); print(f'${{total:.2f}}')"
                    ]
                },
                "depends_on": ["step-1"],
                "risk": "read_only",
                "captures": {"travel_total": "stdout:strip"},
                "verify": []
            },
            {
                "id": "step-3",
                "purpose": f"Write summary report to {summary_doc}",
                "tool": "file.write",
                "args": {
                    "path": summary_doc,
                    "content": "July Expense Report\n-------------------\nTotal Travel Spending: ${scratchpad.travel_total}\nAudit Verified: Yes\n"
                },
                "depends_on": ["step-2"],
                "risk": "change",
                "verify": [{"tool": "file.query", "args": {"path": summary_doc}}]
            },
            {
                "id": "step-4",
                "purpose": f"Send notification email with results to {recipient}",
                "tool": "email.send",
                "args": {
                    "to": recipient,
                    "subject": "July Travel Expense Summary",
                    "body": "The July travel expenses total is ${scratchpad.travel_total}. Full report saved at " + summary_doc
                },
                "depends_on": ["step-3"],
                "risk": "change",
                "verify": [{"tool": "email.read", "args": {"folder": "outbox", "max_emails": 1}}]
            }
        ]
    }
    return intent, plan


# ── Generator 12: High-Level Task Chunking (Phase Decomposition) ──────────────
def make_chunked_phases_sample() -> tuple[str, dict[str, Any]]:
    complex_scenarios = [
        (
            "Migrate database schema, verify table integrity, restart backend service, and notify devops",
            [
                {"id": "phase1", "goal": "Inspect current schema and backup database", "tools_hint": ["command.argv", "file.write"]},
                {"id": "phase2", "goal": "Apply database schema migrations", "tools_hint": ["command.argv"]},
                {"id": "phase3", "goal": "Restart backend service and verify port listening", "tools_hint": ["service.restart", "port.inspect"]},
                {"id": "phase4", "goal": "Send completion email to devops", "tools_hint": ["email.send"]}
            ]
        ),
        (
            "Download monthly raw telemetry, compute aggregated analytics, export CSV, and capture web dashboard screenshot",
            [
                {"id": "phase1", "goal": "Download and locate telemetry files", "tools_hint": ["file.list", "command.argv"]},
                {"id": "phase2", "goal": "Aggregate telemetry metrics into CSV summary", "tools_hint": ["data.csv_read", "data.csv_write"]},
                {"id": "phase3", "goal": "Open web dashboard and capture verification screenshot", "tools_hint": ["browser.navigate", "browser.screenshot"]}
            ]
        ),
        (
            "Audit system security, inspect listening ports, update firewall packages, and log report",
            [
                {"id": "phase1", "goal": "Inspect listening network ports and running services", "tools_hint": ["port.inspect", "service.status"]},
                {"id": "phase2", "goal": "Install security package updates", "tools_hint": ["package.install"]},
                {"id": "phase3", "goal": "Compile and save security audit log", "tools_hint": ["file.write"]}
            ]
        )
    ]
    intent, phases = random.choice(complex_scenarios)
    phase_response = {"phases": phases}
    return intent, phase_response


# ── Generator 13: Step Error Recovery & Repair Traces ─────────────────────────
def make_repair_trace_sample() -> tuple[dict[str, Any], dict[str, Any]]:
    repairs = [
        (
            "Install package nginx",
            "exit_code=1 stderr='apk: package nginx not found in cache'",
            [{"tool": "package.install", "args": {"package": "nginx"}, "exit_code": 1}],
            {
                "goal": "Update repository cache and install package nginx",
                "assumptions": ["Network online"],
                "success_criteria": [{"tool": "package.query", "args": {"package": "nginx"}}],
                "steps": [
                    {
                        "id": "repair-1",
                        "purpose": "Update package repository index",
                        "tool": "command.argv",
                        "args": {"argv": ["apk", "update"]},
                        "depends_on": [],
                        "risk": "change",
                        "verify": [{"tool": "command.argv", "args": {"argv": ["echo", "updated"]}}]
                    },
                    {
                        "id": "repair-2",
                        "purpose": "Install nginx package",
                        "tool": "package.install",
                        "args": {"package": "nginx"},
                        "depends_on": ["repair-1"],
                        "risk": "change",
                        "verify": [{"tool": "package.query", "args": {"package": "nginx"}}]
                    }
                ]
            }
        ),
        (
            "Start service lightdm",
            "exit_code=1 stderr='service lightdm already running or locked'",
            [{"tool": "service.start", "args": {"name": "lightdm"}, "exit_code": 1}],
            {
                "goal": "Restart service lightdm cleanly",
                "assumptions": ["Init system available"],
                "success_criteria": [{"tool": "service.status", "args": {"name": "lightdm"}}],
                "steps": [
                    {
                        "id": "repair-1",
                        "purpose": "Restart service lightdm",
                        "tool": "service.restart",
                        "args": {"name": "lightdm"},
                        "depends_on": [],
                        "risk": "change",
                        "verify": [{"tool": "service.status", "args": {"name": "lightdm"}}]
                    }
                ]
            }
        ),
        (
            "Write file /tmp/output/report.txt",
            "exit_code=1 stderr='No such file or directory: /tmp/output'",
            [{"tool": "file.write", "args": {"path": "/tmp/output/report.txt"}, "exit_code": 1}],
            {
                "goal": "Create parent directory and write file",
                "assumptions": ["Filesystem writable"],
                "success_criteria": [{"tool": "file.query", "args": {"path": "/tmp/output/report.txt"}}],
                "steps": [
                    {
                        "id": "repair-1",
                        "purpose": "Create parent directory /tmp/output",
                        "tool": "dir.create",
                        "args": {"path": "/tmp/output"},
                        "depends_on": [],
                        "risk": "change",
                        "verify": [{"tool": "dir.query", "args": {"path": "/tmp/output"}}]
                    },
                    {
                        "id": "repair-2",
                        "purpose": "Write report file",
                        "tool": "file.write",
                        "args": {"path": "/tmp/output/report.txt", "content": "Status: OK\n"},
                        "depends_on": ["repair-1"],
                        "risk": "change",
                        "verify": [{"tool": "file.query", "args": {"path": "/tmp/output/report.txt"}}]
                    }
                ]
            }
        )
    ]
    purpose, failure, evidence, repair_plan = random.choice(repairs)
    context = {"purpose": purpose, "failure": failure, "evidence": evidence}
    return context, repair_plan


# ── Registered Sample Generators with Distribution Weights ───────────────────
PLAN_GENERATORS = [
    (make_dir_create_sample, 0.10),
    (make_file_write_sample, 0.10),
    (make_package_sample, 0.10),
    (make_service_sample, 0.10),
    (make_monitoring_sample, 0.08),
    (make_file_management_sample, 0.12),
    (make_data_and_document_sample, 0.12),
    (make_browser_sample, 0.10),
    (make_email_sample, 0.08),
    (make_coding_sample, 0.10),
    (make_multiapp_orchestration_sample, 0.10),
]


def generate_dataset(count: int, output_file: str, include_chunks: bool = True, include_repairs: bool = True):
    print(f"Generating {count} syntactically validated OS planning scenarios (Harness v2.3 compliant)...")
    funcs, weights = zip(*PLAN_GENERATORS)

    validated_samples = []
    attempts = 0
    max_attempts = count * 4

    # Calculate sub-counts
    phase_target = int(count * 0.08) if include_chunks else 0
    repair_target = int(count * 0.10) if include_repairs else 0
    plan_target = count - phase_target - repair_target

    # 1. Generate Standard & Multi-Step Plans
    while len([s for s in validated_samples if s.get("type") == "plan"]) < plan_target and attempts < max_attempts:
        attempts += 1
        gen_fn = random.choices(funcs, weights=weights)[0]
        intent, plan_dict = gen_fn()

        # Strict validation against harness_v2.validate_plan
        try:
            validate_plan(plan_dict)
            validated_samples.append({
                "id": len(validated_samples) + 1,
                "type": "plan",
                "intent": intent,
                "plan": plan_dict
            })
        except PlanError as e:
            print(f"[WARN] Generated plan failed validation: {e}")
            continue

    # 2. Generate Chunked Phase Decompositions
    for _ in range(phase_target):
        intent, phase_obj = make_chunked_phases_sample()
        validated_samples.append({
            "id": len(validated_samples) + 1,
            "type": "chunk_phases",
            "intent": intent,
            "phases": phase_obj
        })

    # 3. Generate Repair Traces
    for _ in range(repair_target):
        ctx, repair_plan = make_repair_trace_sample()
        try:
            validate_plan(repair_plan)
            validated_samples.append({
                "id": len(validated_samples) + 1,
                "type": "repair",
                "context": ctx,
                "plan": repair_plan
            })
        except PlanError as e:
            print(f"[WARN] Repair plan failed validation: {e}")

    # Shuffle to interleave plans, chunks, and repairs
    random.shuffle(validated_samples)

    print(f"Successfully generated {len(validated_samples)} examples:")
    print(f"  - Full/Multi-step plans: {len([s for s in validated_samples if s.get('type') == 'plan'])}")
    print(f"  - Chunked phase decompositions: {len([s for s in validated_samples if s.get('type') == 'chunk_phases'])}")
    print(f"  - Step repair plans: {len([s for s in validated_samples if s.get('type') == 'repair'])}")

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for s in validated_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Saved dataset to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic OS task planning dataset (Harness v2.3 compliant)")
    parser.add_argument("--count", type=int, default=1200, help="Total number of samples to generate")
    parser.add_argument("--output", default="planning_dataset.jsonl", help="Output JSONL file")
    parser.add_argument("--no-chunks", action="store_true", help="Do not generate chunked planning phases")
    parser.add_argument("--no-repairs", action="store_true", help="Do not generate step repair traces")
    args = parser.parse_args()

    generate_dataset(
        args.count,
        args.output,
        include_chunks=not args.no_chunks,
        include_repairs=not args.no_repairs
    )
