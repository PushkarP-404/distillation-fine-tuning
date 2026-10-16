#!/usr/bin/env python3
"""
generate_planning_dataset.py
----------------------------
Generates synthetic, structurally validated OS task planning datasets for training
on-device SLMs (e.g., SmolLM2-135M / 360M) to act as reliable autonomous OS planners.

Every generated sample is strictly verified against harness_v2.validate_plan()
to guarantee 0% syntax errors, valid tool names, and mandatory verification steps.
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

from harness_v2 import validate_plan, ALLOWED_TOOLS, PolicyEngine, PlanError

# Common Linux packages, services, paths, and commands
SAMPLE_PACKAGES = [
    "curl", "git", "htop", "tmux", "jq", "vim", "nano", "tree", "wget", "zip",
    "unzip", "rsync", "tcpdump", "netcat-openbsd", "strace", "gdb", "python3",
    "py3-pip", "nginx", "lightdm", "xfce4", "papirus-icon-theme", "make", "gcc"
]

SAMPLE_SERVICES = [
    "lightdm", "sshd", "networking", "ai-agent", "ai-agent-privileged", "udev",
    "syslog", "crond", "acpid", "dbus", "ntpd", "local"
]

SAMPLE_DIRS = [
    "Desktop/Test", "Desktop/Projects", "Desktop/Notes", "Desktop/Archive",
    "Documents/Work", "Documents/Reports", "Downloads/Temp", "Projects/Python",
    "Projects/Web", "Projects/Rust", "/tmp/cache", "/tmp/scratch", "/var/log/custom",
    "Desktop/Research", "Desktop/Scripts", "Desktop/Backup", "Desktop/Data"
]

SAMPLE_FILES = [
    ("notes.txt", "Meeting notes and action items\n"),
    ("todo.md", "# TODO\n- [ ] Initial task\n"),
    ("script.sh", "#!/bin/sh\necho 'Running task...'\n"),
    ("config.json", '{"version": "1.0", "active": true}\n'),
    ("app.py", "#!/usr/bin/env python3\nprint('AI-Agent OS')\n"),
    ("server.py", "import http.server\n"),
    ("data.csv", "id,name,status\n1,alpha,ok\n"),
    ("readme.txt", "Automated system report\n"),
    ("info.log", "Initialized agent runtime\n")
]


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
        f"create a new folder called {folder_name} on {parent_dir}",
        f"please make folder {target}",
        f"create folder {folder_name} in {parent_dir}",
    ]
    intent = random.choice(phrasings)

    plan = {
        "goal": f"Create directory {full_path}",
        "assumptions": ["Parent path accessible"],
        "success_criteria": [{"tool": "dir.query", "args": {"path": full_path}}],
        "steps": [
            {
                "id": "step-1",
                "purpose": f"Create directory {full_path}",
                "tool": "dir.create",
                "args": {"path": full_path},
                "depends_on": [],
                "verify": [{"tool": "dir.query", "args": {"path": full_path}}]
            }
        ]
    }
    return intent, plan


def make_file_write_sample(user="aiuser") -> tuple[str, dict[str, Any]]:
    folder = random.choice(["Desktop", "Documents", "Downloads", "/tmp"])
    fname, content = random.choice(SAMPLE_FILES)
    full_path = f"/tmp/{fname}" if folder == "/tmp" else f"/home/{user}/{folder}/{fname}"
    disp_folder = folder if folder.startswith("/") else f"~/{folder}"

    phrasings = [
        f"create a file in {disp_folder} called {fname}",
        f"write a file called {fname} in {disp_folder}",
        f"touch {disp_folder}/{fname}",
        f"make a file called {fname} in {disp_folder}",
        f"create file {full_path}",
        f"create a new empty file in {disp_folder} named {fname}",
    ]
    intent = random.choice(phrasings)

    plan = {
        "goal": f"Create file {full_path}",
        "assumptions": ["Directory accessible"],
        "success_criteria": [{"tool": "file.query", "args": {"path": full_path}}],
        "steps": [
            {
                "id": "step-1",
                "purpose": f"Write file {full_path}",
                "tool": "file.write",
                "args": {"path": full_path, "content": content},
                "depends_on": [],
                "verify": [{"tool": "file.query", "args": {"path": full_path}}]
            }
        ]
    }
    return intent, plan


def make_package_sample() -> tuple[str, dict[str, Any]]:
    pkg = random.choice(SAMPLE_PACKAGES)
    is_install = random.random() < 0.8

    if is_install:
        phrasings = [
            f"install {pkg}",
            f"install package {pkg}",
            f"apk add {pkg}",
            f"please install {pkg} and verify it works",
            f"add package {pkg} to the system",
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
                    "verify": [{"tool": "package.query", "args": {"package": pkg}}]
                }
            ]
        }
    else:
        phrasings = [
            f"check if {pkg} is installed",
            f"query package {pkg}",
            f"is {pkg} installed on this machine?",
            f"verify {pkg} installation status",
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
                    "verify": []
                }
            ]
        }
    return intent, plan


def make_service_sample() -> tuple[str, dict[str, Any]]:
    svc = random.choice(SAMPLE_SERVICES)
    action = random.choice(["start", "restart", "stop", "status"])

    if action in ("start", "restart", "stop"):
        phrasings = [
            f"{action} service {svc}",
            f"{action} {svc}",
            f"rc-service {svc} {action}",
            f"please {action} the {svc} service",
        ]
        intent = random.choice(phrasings)
        tool_name = f"service.{action}"
        verify_tool = "service.status"
        plan = {
            "goal": f"{action.capitalize()} service {svc}",
            "assumptions": ["Init system available"],
            "success_criteria": [{"tool": verify_tool, "args": {"name": svc}}],
            "steps": [
                {
                    "id": "step-1",
                    "purpose": f"{action.capitalize()} service {svc}",
                    "tool": tool_name,
                    "args": {"name": svc},
                    "depends_on": [],
                    "verify": [{"tool": verify_tool, "args": {"name": svc}}]
                }
            ]
        }
    else:
        phrasings = [
            f"check status of service {svc}",
            f"is {svc} running?",
            f"service {svc} status",
            f"inspect service {svc}",
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
                    "verify": []
                }
            ]
        }
    return intent, plan


def make_monitoring_sample() -> tuple[str, dict[str, Any]]:
    tool = random.choice(["disk.status", "memory.status", "system.info"])
    if tool == "disk.status":
        phrasings = [
            "check disk space", "disk usage", "show free disk space",
            "df -h", "how much storage is left?", "inspect disk status"
        ]
        goal = "Collect disk usage statistics"
    elif tool == "memory.status":
        phrasings = [
            "check memory usage", "show free ram", "free -m",
            "how much ram is used?", "inspect memory status"
        ]
        goal = "Collect memory usage statistics"
    else:
        phrasings = [
            "check system info", "system status", "uname -a",
            "show operating system information", "inspect host environment"
        ]
        goal = "Collect host and OS information"

    intent = random.choice(phrasings)
    plan = {
        "goal": goal,
        "assumptions": [],
        "success_criteria": [{"tool": tool, "args": {}}],
        "steps": [
            {
                "id": "step-1",
                "purpose": goal,
                "tool": tool,
                "args": {},
                "depends_on": [],
                "verify": []
            }
        ]
    }
    return intent, plan


def make_composed_multistep_sample(user="aiuser") -> tuple[str, dict[str, Any]]:
    types = ["dir_and_file", "pkg_and_service"]
    selected = random.choice(types)

    if selected == "dir_and_file":
        folder = random.choice(["Desktop/Project", "Projects/WebApp", "Desktop/Workspace", "Documents/Notes"])
        dir_path = f"/home/{user}/{folder}"
        file_path = f"{dir_path}/README.md"

        phrasings = [
            f"create folder {folder} and make README.md inside it",
            f"make directory ~/{folder} and create file README.md",
            f"setup project directory ~/{folder} with a README.md file",
            f"create {dir_path} and initialize README.md",
        ]
        intent = random.choice(phrasings)

        plan = {
            "goal": f"Create directory {dir_path} and initialize README.md",
            "assumptions": ["Filesystem writable"],
            "success_criteria": [
                {"tool": "dir.query", "args": {"path": dir_path}},
                {"tool": "file.query", "args": {"path": file_path}}
            ],
            "steps": [
                {
                    "id": "step-1",
                    "purpose": f"Create directory {dir_path}",
                    "tool": "dir.create",
                    "args": {"path": dir_path},
                    "depends_on": [],
                    "verify": [{"tool": "dir.query", "args": {"path": dir_path}}]
                },
                {
                    "id": "step-2",
                    "purpose": f"Initialize {file_path}",
                    "tool": "file.write",
                    "args": {"path": file_path, "content": "# Project Initialized\n"},
                    "depends_on": ["step-1"],
                    "verify": [{"tool": "file.query", "args": {"path": file_path}}]
                }
            ]
        }
        return intent, plan

    else:
        pkg = random.choice(["nginx", "lightdm", "sshd"])
        phrasings = [
            f"install {pkg} and start it",
            f"install package {pkg} and start service {pkg}",
            f"setup {pkg} and make sure service is active",
        ]
        intent = random.choice(phrasings)
        plan = {
            "goal": f"Install package {pkg} and start service {pkg}",
            "assumptions": ["Network repository online"],
            "success_criteria": [
                {"tool": "package.query", "args": {"package": pkg}},
                {"tool": "service.status", "args": {"name": pkg}}
            ],
            "steps": [
                {
                    "id": "step-1",
                    "purpose": f"Install {pkg}",
                    "tool": "package.install",
                    "args": {"package": pkg},
                    "depends_on": [],
                    "verify": [{"tool": "package.query", "args": {"package": pkg}}]
                },
                {
                    "id": "step-2",
                    "purpose": f"Start {pkg} service",
                    "tool": "service.start",
                    "args": {"name": pkg},
                    "depends_on": ["step-1"],
                    "verify": [{"tool": "service.status", "args": {"name": pkg}}]
                }
            ]
        }
        return intent, plan


GENERATORS = [
    (make_dir_create_sample, 0.25),
    (make_file_write_sample, 0.25),
    (make_package_sample, 0.20),
    (make_service_sample, 0.15),
    (make_monitoring_sample, 0.10),
    (make_composed_multistep_sample, 0.05),
]


def generate_dataset(count: int, output_file: str):
    print(f"Generating {count} syntactically validated OS planning scenarios...")
    funcs, weights = zip(*GENERATORS)
    
    validated_samples = []
    attempts = 0
    max_attempts = count * 3

    while len(validated_samples) < count and attempts < max_attempts:
        attempts += 1
        gen_fn = random.choices(funcs, weights=weights)[0]
        intent, plan_dict = gen_fn()

        # Strict validation against harness_v2.validate_plan
        try:
            validate_plan(plan_dict)
            validated_samples.append({
                "id": len(validated_samples) + 1,
                "intent": intent,
                "plan": plan_dict
            })
        except PlanError as e:
            print(f"[WARN] Generated plan failed validation: {e}")
            continue

    print(f"Successfully generated and validated {len(validated_samples)} planning pairs.")
    
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for s in validated_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Saved dataset to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic OS task planning dataset")
    parser.add_argument("--count", type=int, default=1000, help="Number of planning samples to generate")
    parser.add_argument("--output", default="planning_dataset.jsonl", help="Output JSONL file")
    args = parser.parse_args()

    generate_dataset(args.count, args.output)
