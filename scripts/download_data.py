from pathlib import Path
import json
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

dataset = "realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv"
urls = {
    "ec2_cpu_utilization_53ea38.csv":
        "https://raw.githubusercontent.com/numenta/NAB/master/data/" + dataset,
    "combined_windows.json":
        "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json",
}

for filename, url in urls.items():
    print(f"Downloading {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    (RAW / filename).write_bytes(response.content)

(RAW / "metadata.json").write_text(json.dumps({
    "dataset": dataset,
    "source": "Numenta Anomaly Benchmark",
    "source_repository": "https://github.com/numenta/NAB"
}, indent=2))

print("Dataset download complete.")
