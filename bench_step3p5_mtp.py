#!/usr/bin/env python3
"""Step3.5-Flash MTP offline benchmark.

Usage:
    # 基础跑法：启动服务器 → 跑 benchmark → 关闭
    python bench_step3p5_mtp.py

    # 如果服务器已经在跑（比如你用 test.sh 启动了），直接连上去跑
    python bench_step3p5_mtp.py --server-already-running --port 8000

    # 调整参数
    python bench_step3p5_mtp.py --tp 4 --num-rounds 5 --max-new-tokens 2048

    # 跑 GSM8K 准确率测试
    python bench_step3p5_mtp.py --run-gsm8k
"""

import argparse
import json
import os
import signal
import sys
import time
from types import SimpleNamespace

import requests


def parse_args():
    parser = argparse.ArgumentParser(description="Step3.5-Flash MTP Benchmark")
    parser.add_argument(
        "--model-path",
        type=str,
        default="stepfun-ai/Step-3.5-Flash",
        help="Model path",
    )
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallelism size")
    parser.add_argument("--port", type=int, default=30000, help="Server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument(
        "--num-rounds", type=int, default=3, help="Number of benchmark rounds"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=2048, help="Max new tokens per request"
    )
    parser.add_argument(
        "--server-already-running",
        action="store_true",
        help="Skip launching server, connect to existing one",
    )
    parser.add_argument(
        "--run-gsm8k",
        action="store_true",
        help="Also run GSM8K 5-shot accuracy test",
    )
    parser.add_argument(
        "--gsm8k-num-questions",
        type=int,
        default=200,
        help="Number of GSM8K questions",
    )
    parser.add_argument(
        "--speculative-num-steps", type=int, default=3, help="Speculative num steps"
    )
    parser.add_argument(
        "--speculative-eagle-topk", type=int, default=1, help="Speculative eagle topk"
    )
    parser.add_argument(
        "--speculative-num-draft-tokens",
        type=int,
        default=4,
        help="Speculative num draft tokens",
    )
    return parser.parse_args()


def launch_server(args):
    """Launch sglang server and wait until ready."""
    from sglang.srt.utils import kill_process_tree
    from sglang.test.test_utils import (
        DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        popen_launch_server,
    )

    base_url = f"http://{args.host}:{args.port}"
    other_args = [
        "--tp",
        str(args.tp),
        "--trust-remote-code",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        str(args.speculative_num_steps),
        "--speculative-eagle-topk",
        str(args.speculative_eagle_topk),
        "--speculative-num-draft-tokens",
        str(args.speculative_num_draft_tokens),
        "--enable-multi-layer-eagle",
        "--port",
        str(args.port),
    ]
    env = os.environ.copy()
    env["SGLANG_ENABLE_SPEC_V2"] = "1"
    print(f"[Bench] Launching server: {args.model_path} tp={args.tp} ...")
    process = popen_launch_server(
        args.model_path,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
        other_args=other_args,
        env=env,
    )
    print(f"[Bench] Server ready at {base_url}")
    return process, base_url


def flush_cache(base_url):
    try:
        requests.get(f"{base_url}/flush_cache", timeout=10)
    except Exception:
        pass


def get_server_spec_info(base_url):
    """Get avg_spec_accept_length from server."""
    try:
        resp = requests.get(f"{base_url}/get_server_info", timeout=10)
        data = resp.json()
        return data["internal_states"][0].get("avg_spec_accept_length", 0)
    except Exception:
        return 0


def run_speed_bench(base_url, max_new_tokens, prompt=None):
    """Send one prompt and measure speed + acceptance length."""
    if prompt is None:
        prompt = (
            "Human: Give me a fully functional FastAPI server. "
            "Show the python code.\n\nAssistant:"
        )

    json_data = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "stop": ["Question", "Assistant:", "<|separator|>", "<|eos|>"],
        },
    }

    start = time.perf_counter()
    response = requests.post(f"{base_url}/generate", json=json_data)
    elapsed = time.perf_counter() - start

    if response.status_code != 200:
        print(f"  [Error] status={response.status_code}")
        return 0, 0, 0, 0

    ret = response.json()
    meta = ret["meta_info"]
    tokens = meta["completion_tokens"]
    speed = tokens / elapsed

    if "spec_verify_ct" in meta and meta["spec_verify_ct"] > 0:
        acc_length = tokens / meta["spec_verify_ct"]
    else:
        acc_length = 0

    return elapsed, tokens, acc_length, speed


def run_gsm8k_bench(base_url, port, num_questions):
    """Run GSM8K 5-shot evaluation."""
    from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k

    gsm_args = SimpleNamespace(
        num_shots=5,
        data_path=None,
        num_questions=num_questions,
        max_new_tokens=512,
        parallel=128,
        host="http://127.0.0.1",
        port=port,
    )
    metrics = run_eval_few_shot_gsm8k(gsm_args)
    return metrics


def print_table(headers, rows):
    """Simple table printer."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    fmt = " | ".join(f"{{:^{w}}}" for w in col_widths)
    sep = "-+-".join("-" * w for w in col_widths)

    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


def main():
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"
    process = None

    if not args.server_already_running:
        process, base_url = launch_server(args)

    try:
        # ================================================================
        # Speed benchmark
        # ================================================================
        print(f"\n{'='*60}")
        print(f" Step3.5-Flash MTP Speed Benchmark")
        print(f" Rounds: {args.num_rounds}  |  Max tokens: {args.max_new_tokens}")
        print(f"{'='*60}\n")

        flush_cache(base_url)

        # Warmup
        print("[Bench] Warmup round ...")
        run_speed_bench(base_url, args.max_new_tokens)
        flush_cache(base_url)

        results = []
        for i in range(args.num_rounds):
            elapsed, tokens, acc_length, speed = run_speed_bench(
                base_url, args.max_new_tokens
            )
            results.append((elapsed, tokens, acc_length, speed))
            print(
                f"  Round {i+1}: {tokens} tokens, "
                f"acc_len={acc_length:.2f}, "
                f"speed={speed:.1f} tok/s, "
                f"latency={elapsed:.2f}s"
            )
            flush_cache(base_url)

        # Summary
        avg_acc = sum(r[2] for r in results) / len(results)
        avg_speed = sum(r[3] for r in results) / len(results)
        avg_tokens = sum(r[1] for r in results) / len(results)
        max_draft = args.speculative_num_draft_tokens
        accept_rate = avg_acc / max_draft * 100 if max_draft > 0 else 0

        # Server-side acceptance length (aggregated across all requests)
        server_acc = get_server_spec_info(base_url)

        print(f"\n{'='*60}")
        print(f" Results Summary")
        print(f"{'='*60}")
        print_table(
            ["Metric", "Value"],
            [
                ["Avg Acceptance Length", f"{avg_acc:.2f} / {max_draft}"],
                ["Acceptance Rate", f"{accept_rate:.1f}%"],
                ["Avg Speed (tok/s)", f"{avg_speed:.1f}"],
                ["Avg Tokens Generated", f"{avg_tokens:.0f}"],
                ["Server Avg Acc Length", f"{server_acc:.2f}"],
                ["Spec Steps", f"{args.speculative_num_steps}"],
                ["Eagle Topk", f"{args.speculative_eagle_topk}"],
                ["Draft Tokens", f"{args.speculative_num_draft_tokens}"],
            ],
        )

        # ================================================================
        # GSM8K accuracy benchmark (optional)
        # ================================================================
        if args.run_gsm8k:
            print(f"\n{'='*60}")
            print(f" GSM8K 5-shot Accuracy Test ({args.gsm8k_num_questions} questions)")
            print(f"{'='*60}\n")

            flush_cache(base_url)
            metrics = run_gsm8k_bench(base_url, args.port, args.gsm8k_num_questions)
            gsm8k_server_acc = get_server_spec_info(base_url)

            print(f"\n{'='*60}")
            print(f" GSM8K Results")
            print(f"{'='*60}")
            print_table(
                ["Metric", "Value"],
                [
                    ["Accuracy", f'{metrics["accuracy"]:.3f}'],
                    [
                        "Correct / Total",
                        f'{metrics.get("num_correct", "?")}/{metrics.get("num_total", "?")}',
                    ],
                    ["Server Avg Acc Length", f"{gsm8k_server_acc:.2f}"],
                ],
            )

        # ================================================================
        # Final one-liner for easy copy-paste comparison
        # ================================================================
        print(f"\n[One-liner] acc_len={avg_acc:.2f} speed={avg_speed:.1f} tok/s"
              f" server_acc={server_acc:.2f}")

    finally:
        if process is not None:
            from sglang.srt.utils import kill_process_tree

            print("\n[Bench] Shutting down server ...")
            kill_process_tree(process.pid)
            print("[Bench] Done.")


if __name__ == "__main__":
    main()
