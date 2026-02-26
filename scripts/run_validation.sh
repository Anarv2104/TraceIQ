#!/usr/bin/env bash
# scripts/run_validation.sh
#
# CI-safe validation runner with SMOKE/PROOF modes.
# SMOKE: never fails on statistical FAIL (except pytest), only on real crashes.
# PROOF: fails on any experiment FAIL.

set -uo pipefail  # NO -e, we handle exit codes manually

MODE="${1:-all}"  # smoke, proof, all

# Crash exit codes
CRASH_CODES=(2 126 127 137 143)

# Crash markers in output (specific Python exceptions)
# Note: AssertionError excluded - treated as FAIL not CRASH
CRASH_MARKERS=(
    "Traceback (most recent call last):"
    "ModuleNotFoundError:"
    "ImportError:"
    "SyntaxError:"
    "NameError:"
    "TypeError:"
    "ValueError:"
    "AttributeError:"
    "FileNotFoundError:"
    "RuntimeError:"
    "KeyboardInterrupt"
)

# Arrays to track results
declare -a NAMES=() MODES=() EXITS=() TIMES=() STATUSES=() SUMMARIES=()
FAILED=0

is_crash_code() {
    local code="$1"
    for c in "${CRASH_CODES[@]}"; do
        [[ "$code" -eq "$c" ]] && return 0
    done
    return 1
}

has_crash_marker() {
    local output_file="$1"
    for marker in "${CRASH_MARKERS[@]}"; do
        if grep -qF "$marker" "$output_file" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

find_summary_json() {
    local output_file="$1"
    local start_time="$2"

    # Try to parse from output first
    local parsed
    parsed=$(grep -oE '[^ ]*summary\.json' "$output_file" 2>/dev/null | head -1)
    if [[ -n "$parsed" && -f "$parsed" ]]; then
        echo "$parsed"
        return
    fi

    # Fallback: find newest summary.json in results directories
    local newest=""
    local newest_time=0
    for dir in experiments/results research/outputs; do
        if [[ -d "$dir" ]]; then
            while IFS= read -r -d '' f; do
                local mtime
                mtime=$(stat -f %m "$f" 2>/dev/null || stat -c %Y "$f" 2>/dev/null)
                if [[ "$mtime" -ge "$start_time" && "$mtime" -gt "$newest_time" ]]; then
                    newest="$f"
                    newest_time="$mtime"
                fi
            done < <(find "$dir" -name "summary.json" -print0 2>/dev/null)
        fi
    done

    if [[ -n "$newest" ]]; then
        echo "$newest"
    else
        echo "-"
    fi
}

# Run a command and record results
# Args: name cmd mode [always_fail_on_nonzero] [skip_summary]
run_command() {
    local name="$1"
    local cmd="$2"
    local mode="$3"
    local always_fail="${4:-false}"  # If true, non-zero always fails (for pytest)
    local skip_summary="${5:-false}" # If true, don't search for summary.json

    echo ""
    echo "========================================"
    echo "[$mode] $name"
    echo "========================================"
    echo "CMD: $cmd"
    echo ""

    local start_time
    start_time=$(date +%s)
    local output_file
    output_file=$(mktemp)

    # Run command via bash -c (avoids eval)
    local exit_code=0
    bash -c "$cmd" > "$output_file" 2>&1 || exit_code=$?

    local end_time
    end_time=$(date +%s)
    local runtime=$((end_time - start_time))

    # Determine status based on exit code AND crash markers
    local status
    local is_crash=0

    if [[ "$exit_code" -eq 0 ]]; then
        status="PASS"
    elif is_crash_code "$exit_code"; then
        status="CRASH"
        is_crash=1
    elif has_crash_marker "$output_file"; then
        status="CRASH"
        is_crash=1
    elif [[ "$always_fail" == "true" ]]; then
        # pytest and similar: always fail on non-zero
        status="FAIL"
    elif [[ "$mode" == "smoke" ]]; then
        status="INCONCLUSIVE"
    else
        status="FAIL"
    fi

    # Mark failed if CRASH or FAIL
    if [[ "$is_crash" -eq 1 ]] || [[ "$status" == "FAIL" ]]; then
        FAILED=1
    fi

    # Find summary.json (skip for pytest and similar)
    local summary_path="-"
    if [[ "$skip_summary" != "true" ]]; then
        summary_path=$(find_summary_json "$output_file" "$start_time")
    fi

    # Store results
    NAMES+=("$name")
    MODES+=("$mode")
    EXITS+=("$exit_code")
    TIMES+=("${runtime}s")
    STATUSES+=("$status")
    SUMMARIES+=("$summary_path")

    # Print output
    local lines=15
    [[ "$status" != "PASS" ]] && lines=30
    echo "--- Output (last $lines lines) ---"
    tail -n "$lines" "$output_file"
    echo "--- End output ---"
    echo "Exit: $exit_code | Status: $status | Time: ${runtime}s"

    rm -f "$output_file"

    # Return exit code for flag probing
    return "$exit_code"
}

# Run log_replay with flag probing (retry without --no-strict if argparse fails)
run_log_replay_smoke() {
    local output_file
    output_file=$(mktemp)
    local exit_code=0

    echo ""
    echo "========================================"
    echo "[smoke] log_replay (probing flags)"
    echo "========================================"

    local start_time
    start_time=$(date +%s)

    # Try with --no-strict first
    echo "Trying: python -m experiments.exp_log_replay --no-strict"
    bash -c "python -m experiments.exp_log_replay --no-strict" > "$output_file" 2>&1 || exit_code=$?

    if [[ "$exit_code" -eq 2 ]]; then
        # Argparse error - retry without flag
        echo "Flag --no-strict not supported (exit 2), retrying without..."
        exit_code=0
        bash -c "python -m experiments.exp_log_replay" > "$output_file" 2>&1 || exit_code=$?
    fi

    local end_time
    end_time=$(date +%s)
    local runtime=$((end_time - start_time))

    # Now classify the final result
    local status
    local is_crash=0

    if [[ "$exit_code" -eq 0 ]]; then
        status="PASS"
    elif is_crash_code "$exit_code"; then
        status="CRASH"
        is_crash=1
    elif has_crash_marker "$output_file"; then
        status="CRASH"
        is_crash=1
    else
        status="INCONCLUSIVE"
    fi

    if [[ "$is_crash" -eq 1 ]]; then
        FAILED=1
    fi

    local summary_path
    summary_path=$(find_summary_json "$output_file" "$start_time")

    NAMES+=("log_replay")
    MODES+=("smoke")
    EXITS+=("$exit_code")
    TIMES+=("${runtime}s")
    STATUSES+=("$status")
    SUMMARIES+=("$summary_path")

    local lines=15
    [[ "$status" != "PASS" ]] && lines=30
    echo "--- Output (last $lines lines) ---"
    tail -n "$lines" "$output_file"
    echo "--- End output ---"
    echo "Exit: $exit_code | Status: $status | Time: ${runtime}s"

    rm -f "$output_file"
}

sanity_check() {
    echo ""
    echo "========================================"
    echo "Sanity Check"
    echo "========================================"
    if ! python -c "import traceiq; print(f'traceiq version: {traceiq.__version__}')"; then
        echo "FATAL: Cannot import traceiq"
        exit 1
    fi
    echo "Sanity check PASSED"
}

run_smoke() {
    echo ""
    echo "######################################## "
    echo "#            SMOKE MODE                #"
    echo "######################################## "

    sanity_check

    run_command "causal_chain" \
        "python -m experiments.exp_causal_chain --seeds 3 --quick" "smoke"

    run_command "false_positive" \
        "python -m experiments.exp_false_positive --seeds 3 --quick" "smoke"

    run_command "scaling" \
        "python -m experiments.exp_scaling --seeds 3 --quick" "smoke"

    # exp_log_replay: special handling with flag probing
    run_log_replay_smoke

    # pytest: always fail on non-zero (not INCONCLUSIVE)
    # pytest: always fail on non-zero, skip summary detection
    run_command "pytest" "pytest -q --tb=short" "smoke" "true" "true"
}

run_proof() {
    echo ""
    echo "######################################## "
    echo "#            PROOF MODE                #"
    echo "######################################## "

    run_command "causal_chain" \
        "python -m experiments.exp_causal_chain --seeds 50" "proof"

    run_command "false_positive" \
        "python -m experiments.exp_false_positive --seeds 50" "proof"

    run_command "scaling" \
        "python -m experiments.exp_scaling --seeds 20" "proof"

    run_command "log_replay" \
        "python -m experiments.exp_log_replay" "proof"

    # Plot only if results CSVs exist
    if [[ -d "experiments/results" ]] && \
       find experiments/results -name "*.csv" -print -quit 2>/dev/null | grep -q .; then
        run_command "plot_all" \
            "python -m experiments.plot_all" "proof"
    fi
}

print_table() {
    echo ""
    echo "========================================"
    echo "                SUMMARY                 "
    echo "========================================"
    printf "%-20s %-8s %-6s %-8s %-14s %s\n" \
        "NAME" "MODE" "EXIT" "TIME" "STATUS" "SUMMARY"
    printf "%-20s %-8s %-6s %-8s %-14s %s\n" \
        "----" "----" "----" "----" "------" "-------"
    for i in "${!NAMES[@]}"; do
        printf "%-20s %-8s %-6s %-8s %-14s %s\n" \
            "${NAMES[$i]}" "${MODES[$i]}" "${EXITS[$i]}" \
            "${TIMES[$i]}" "${STATUSES[$i]}" "${SUMMARIES[$i]}"
    done
    echo ""
    if [[ "$FAILED" -eq 1 ]]; then
        echo "OVERALL: FAILED"
    else
        echo "OVERALL: PASSED"
    fi
}

main() {
    case "$MODE" in
        smoke)
            run_smoke
            ;;
        proof)
            run_proof
            ;;
        all)
            run_smoke
            run_proof
            ;;
        *)
            echo "Usage: $0 {smoke|proof|all}"
            exit 2
            ;;
    esac

    print_table

    exit "$FAILED"
}

main
