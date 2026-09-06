#!/bin/bash
# 运行 Legacy KG Coverage Audit
# 用法: bash scripts/run_legacy_kg_audit.sh

set -euo pipefail

# 配置
DATASETS="hotpotqa musique"
N_SAMPLES=100
SEED=46
SPLIT=dev
KG_INDEX="indexes/kg_cache/question_kg_index_v2.json"
OUTPUT_DIR="reports/legacy_kg_audit"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/audit_n${N_SAMPLES}_seed${SEED}_${TIMESTAMP}.json"

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Legacy KG Coverage Audit"
echo "=========================================="
echo "Datasets:    ${DATASETS}"
echo "Sample:      ${N_SAMPLES} questions per dataset"
echo "Seed:        ${SEED}"
echo "Split:       ${SPLIT}"
echo "KG Index:    ${KG_INDEX}"
echo "Output:      ${OUTPUT_FILE}"
echo "=========================================="
echo ""

# 检查依赖
if [ ! -f "${KG_INDEX}" ]; then
    echo "❌ Error: KG index not found: ${KG_INDEX}"
    exit 1
fi

# 运行审计
python scripts/diagnose/legacy_kg_coverage_audit.py \
    --datasets ${DATASETS} \
    --n_samples ${N_SAMPLES} \
    --seed ${SEED} \
    --split ${SPLIT} \
    --kg_index "${KG_INDEX}" \
    --output "${OUTPUT_FILE}" \
    --max_mentions 5 \
    --offline

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Audit completed successfully"
    echo "=========================================="
    echo "Results saved to:"
    echo "  - JSON: ${OUTPUT_FILE}"
    echo "  - MD:   ${OUTPUT_FILE%.json}.md"
    echo ""
    echo "Next steps:"
    echo "  1. Review the markdown summary: ${OUTPUT_FILE%.json}.md"
    echo "  2. Check bottleneck distribution"
    echo "  3. Decide on repair strategy based on repair_feasibility"
    echo ""
else
    echo ""
    echo "❌ Audit failed with exit code ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi
