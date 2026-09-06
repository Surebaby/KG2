#!/usr/bin/env python3
"""
Pre-flight check for Legacy KG Coverage Audit

验证所有依赖和配置是否就绪
"""

import sys
from pathlib import Path
from typing import List, Tuple

def check_file_exists(path: str, description: str) -> Tuple[bool, str]:
    """Check if a file exists."""
    p = Path(path)
    if p.exists():
        return True, f"✓ {description}: {path}"
    else:
        return False, f"✗ {description} NOT FOUND: {path}"

def check_directory_exists(path: str, description: str) -> Tuple[bool, str]:
    """Check if a directory exists."""
    p = Path(path)
    if p.exists() and p.is_dir():
        return True, f"✓ {description}: {path}"
    else:
        return False, f"✗ {description} NOT FOUND: {path}"

def check_python_import(module: str) -> Tuple[bool, str]:
    """Check if a Python module can be imported."""
    try:
        __import__(module)
        return True, f"✓ Python module: {module}"
    except ImportError:
        return False, f"✗ Python module NOT FOUND: {module}"

def main():
    print("="*80)
    print("Legacy KG Coverage Audit - Pre-flight Check")
    print("="*80)
    print()

    checks: List[Tuple[bool, str]] = []

    # 1. Check core scripts
    print("[ 1. Core Scripts ]")
    checks.append(check_file_exists(
        "scripts/diagnose/legacy_kg_coverage_audit.py",
        "Audit script"
    ))
    checks.append(check_file_exists(
        "scripts/run_legacy_kg_audit.sh",
        "Launch script"
    ))
    checks.append(check_file_exists(
        "tests/test_legacy_kg_coverage_audit.py",
        "Test suite"
    ))
    print()

    # 2. Check documentation
    print("[ 2. Documentation ]")
    checks.append(check_file_exists(
        "docs/legacy_kg_audit_guide.md",
        "Usage guide"
    ))
    checks.append(check_file_exists(
        "docs/legacy_kg_repair_executive_summary.md",
        "Executive summary"
    ))
    checks.append(check_file_exists(
        "configs/experiments/legacy_kg_repair_comparison.yaml",
        "Comparison config"
    ))
    print()

    # 3. Check data availability
    print("[ 3. Data Availability ]")
    checks.append(check_file_exists(
        "data/hotpotqa/dev.jsonl",
        "HotpotQA dev"
    ))
    checks.append(check_file_exists(
        "data/musique/dev.jsonl",
        "MuSiQue dev"
    ))
    checks.append(check_file_exists(
        "indexes/kg_cache/question_kg_index_v2.json",
        "Legacy KG index"
    ))
    print()

    # 4. Check KG infrastructure
    print("[ 4. KG Infrastructure ]")
    checks.append(check_directory_exists(
        "indexes/kg_cache",
        "KG cache directory"
    ))
    checks.append(check_file_exists(
        "kgproweight/kg/entity_linker.py",
        "Entity linker"
    ))
    checks.append(check_file_exists(
        "kgproweight/kg/kg_filter.py",
        "KG filter"
    ))
    checks.append(check_file_exists(
        "kgproweight/kg/wikidata_retriever.py",
        "Wikidata retriever"
    ))
    print()

    # 5. Check Python dependencies
    print("[ 5. Python Dependencies ]")
    for module in ["json", "argparse", "dataclasses", "pathlib", "collections"]:
        checks.append(check_python_import(module))
    print()

    # 6. Check project-specific modules
    print("[ 6. Project Modules ]")
    checks.append(check_python_import("kgproweight"))
    checks.append(check_python_import("kgproweight.kg.entity_linker"))
    checks.append(check_python_import("kgproweight.kg.kg_filter"))
    checks.append(check_python_import("kgproweight.kg.cache"))
    print()

    # Print all results
    print("="*80)
    print("Check Results:")
    print("="*80)

    for passed, message in checks:
        print(message)

    # Summary
    passed_count = sum(1 for p, _ in checks if p)
    total_count = len(checks)

    print()
    print("="*80)
    if passed_count == total_count:
        print(f"✓ All checks passed ({passed_count}/{total_count})")
        print("="*80)
        print()
        print("Ready to run audit:")
        print("  bash scripts/run_legacy_kg_audit.sh")
        print()
        return 0
    else:
        print(f"✗ {total_count - passed_count} checks failed ({passed_count}/{total_count} passed)")
        print("="*80)
        print()
        print("Please fix the failed checks before running the audit.")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
