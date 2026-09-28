"""results.json + benchmarks.json -> cmp2016/dashboard/index.html (한 파일짜리 대시보드).

사용 예:
    python cmp2016/run_cmp.py && python cmp2016/build_cmp_dashboard.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    results = json.loads((HERE / "results.json").read_text())
    bench = json.loads((HERE / "benchmarks.json").read_text())
    data = {"results": results, "benchmarks": bench["rows"], "bench_note": bench["note"], "insights": bench["insights"]}
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = (HERE / "dashboard" / "template.html").read_text().replace("__DATA__", payload)
    out = HERE / "dashboard" / "index.html"
    out.write_text(html)
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
