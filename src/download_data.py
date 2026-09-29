from pathlib import Path

from datasets import load_dataset


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. ReMedQA
# Primary perturbation / robustness dataset
# ============================================================

def download_remedqa():
    print("\n" + "=" * 70)
    print("1/3 — Downloading ReMedQA")
    print("=" * 70)

    dataset = load_dataset(
        "disi-unibo-nlp/ReMedQA"
    )

    output_path = DATA_DIR / "remedqa"
    dataset.save_to_disk(output_path)

    print("\nReMedQA structure:")
    print(dataset)

    print("\nSplits:")
    for split in dataset.keys():
        print(f"  {split:<40} {len(dataset[split]):>8,}")

    print(f"\nSaved to: {output_path}")


# ============================================================
# 2. CareQA
# External biomedical validation dataset
# ============================================================

def download_careqa():
    print("\n" + "=" * 70)
    print("2/3 — Downloading CareQA")
    print("=" * 70)

    # Closed-ended English CareQA
    dataset = load_dataset(
        "HPAI-BSC/CareQA",
        "CareQA_en",
    )

    output_path = DATA_DIR / "careqa_en"
    dataset.save_to_disk(output_path)

    print("\nCareQA structure:")
    print(dataset)

    print("\nSplits:")
    for split in dataset.keys():
        print(f"  {split:<40} {len(dataset[split]):>8,}")

    print(f"\nSaved to: {output_path}")


# ============================================================
# 3. MedQA
# Standard USMLE medical QA benchmark
# ============================================================

def download_medqa():
    print("\n" + "=" * 70)
    print("3/3 — Downloading MedQA")
    print("=" * 70)

    dataset = load_dataset(
        "premmahadik05/medqa",
        "questions",
    )

    output_path = DATA_DIR / "medqa"
    dataset.save_to_disk(output_path)

    print("\nMedQA structure:")
    print(dataset)

    print("\nSplits:")
    for split in dataset.keys():
        print(f"  {split:<40} {len(dataset[split]):>8,}")

    print(f"\nSaved to: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("\n")
    print("=" * 70)
    print("NAACL 2027 — Clinical LLM Reliability Study")
    print("Dataset Downloader")
    print("=" * 70)

    download_remedqa()
    download_careqa()
    download_medqa()

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print("\nDatasets:")
    print(f"  ReMedQA : {DATA_DIR / 'remedqa'}")
    print(f"  CareQA  : {DATA_DIR / 'careqa_en'}")
    print(f"  MedQA   : {DATA_DIR / 'medqa'}")


if __name__ == "__main__":
    main()