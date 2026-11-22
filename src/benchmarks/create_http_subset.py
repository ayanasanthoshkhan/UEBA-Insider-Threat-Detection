
from pathlib import Path
import argparse
import os

DATA_ROOT = Path(
    os.environ.get(
        "UEBA_DATA_ROOT",
        r"D:\UEBA_Capstone\data"
    )
)

SOURCE = DATA_ROOT / "raw" / "r4.2" / "http.csv"
OUTPUT_DIR = DATA_ROOT / "benchmarks" / "http_subsets"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--size-mib",
    type=int,
    choices=[256, 512, 1024],
    required=True
)

args = parser.parse_args()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output = OUTPUT_DIR / f"http_{args.size_mib}mib.csv"
target_bytes = args.size_mib * 1024 * 1024

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

if SOURCE.stat().st_size < target_bytes:
    raise ValueError("Source is smaller than requested subset")

if output.exists():
    raise FileExistsError(
        f"Output already exists: {output}"
    )

print("\n=== HTTP SUBSET CREATION ===")
print("Source:", SOURCE)
print("Target size:", args.size_mib, "MiB")
print("Output:", output)

written_bytes = 0
records = 0

try:
    with SOURCE.open("rb") as src, output.open("wb") as dst:

        header = src.readline()

        expected_header = (
            b"id,date,user,pc,url,content"
        )

        if header.strip() != expected_header:
            raise ValueError("Unexpected HTTP CSV header")

        dst.write(header)
        written_bytes += len(header)

        while written_bytes < target_bytes:
            line = src.readline()

            if not line:
                break

            dst.write(line)
            written_bytes += len(line)
            records += 1

    actual_mib = output.stat().st_size / (1024 * 1024)

    print("\n=== SUBSET RESULT ===")
    print("Data records:", records)
    print("Actual size (MiB):", round(actual_mib, 3))
    print("Saved to:", output)
    print("\nHTTP subset creation successful!")

except Exception:
    if output.exists():
        output.unlink()
    raise
