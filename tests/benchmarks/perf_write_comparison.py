import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

import ffwf as fw


def generate_benchmark_data(num_rows=200_000, num_cols=100):
    print(f"Generating data with {num_rows} rows and {num_cols} columns...")
    data = {}
    for i in range(num_cols):
        if i % 3 == 0:  # Ints
            data[f"col_{i}"] = np.random.randint(0, 10**9, num_rows)
        elif i % 3 == 1:  # Floats
            data[f"col_{i}"] = np.random.uniform(0, 10**6, num_rows)
        else:  # Strings
            data[f"col_{i}"] = [f"val_{j}" for j in range(num_rows)]

    df_pl = pl.DataFrame(data)
    df_pd = df_pl.to_pandas()
    return df_pl, df_pd


def run_write_benchmark():
    """
    Compare ffwf write performance against Polars and Pandas CSV writers.
    """
    # Rationale: FWF is structurally simpler than CSV (no delimiters to escape, no complex quoting).
    # A high-performance FWF writer should theoretically match or exceed the speed of a CSV writer.
    # We use Polars CSV as the "speed of light" baseline for optimized IO.
    print("\n--- Write Performance Comparison ---")
    print("Rationale: FWF is structurally simpler than CSV (no escaping/quoting).")
    print("A high-performance FWF writer should match or beat Polars' CSV speed.")
    print("------------------------------------\n")

    num_rows = 500_000
    num_cols = 100
    df_pl, df_pd = generate_benchmark_data(num_rows, num_cols)

    results = {}
    os.makedirs("data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # 1. ffwf write_fwf_pl
    path_fwf = "data/bench_write.fwf"
    print("Benchmarking ffwf.write_fwf_pl...")
    start = time.perf_counter()
    fw.write_fwf_pl(df_pl, path_fwf)
    results["ffwf (FWF)"] = time.perf_counter() - start
    print(f"ffwf: {results['ffwf (FWF)']:.4f}s")

    # 2. Polars write_csv
    path_csv_pl = "data/bench_write_pl.csv"
    print("Benchmarking Polars.write_csv...")
    start = time.perf_counter()
    df_pl.write_csv(path_csv_pl)
    results["Polars (CSV)"] = time.perf_counter() - start
    print(f"Polars CSV: {results['Polars (CSV)']:.4f}s")

    # 3. Pandas to_csv
    path_csv_pd = "data/bench_write_pd.csv"
    print("Benchmarking Pandas.to_csv...")
    start = time.perf_counter()
    df_pd.to_csv(path_csv_pd, index=False)
    results["Pandas (CSV)"] = time.perf_counter() - start
    print(f"Pandas CSV: {results['Pandas (CSV)']:.4f}s")

    # Cleanup large files
    for p in [path_fwf, path_csv_pl, path_csv_pd]:
        if os.path.exists(p):
            os.remove(p)

    plot_write_results(results, num_rows, num_cols)


def plot_write_results(results, rows, cols):
    names = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=["blue", "green", "red"])
    plt.ylabel("Time (seconds)")
    plt.title(f"Write Performance Comparison ({rows:,} rows, {cols} columns)")

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            f"{yval:.4f}s",
            va="bottom",
            ha="center",
        )

    plt.tight_layout()
    out_path = "plots/write_comparison_benchmark.png"
    plt.savefig(out_path)
    print(f"Chart saved as {out_path}")


if __name__ == "__main__":
    run_write_benchmark()
