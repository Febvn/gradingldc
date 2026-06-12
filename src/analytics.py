"""
Analisis Data Operasional Kopi — pilar kedua proposal.

Proposal KP menyebut dua pilar: (1) Smart Grading dan (2) "Analisis terhadap
aliran data operasional di unit kopi LDC untuk mengidentifikasi peluang
optimasi". Modul ini mewujudkan pilar kedua: setiap sesi grading dicatat ke
basis data (SQLite), lalu diolah menjadi KPI & tren operasional:

    - Distribusi mutu (Mutu 1..6) & yield ekspor premium
    - Tren rata-rata nilai cacat / tingkat cacat harian
    - Throughput (jumlah biji & sampel yang digrading per hari)
    - Pareto jenis cacat (kontribusi tiap defect terhadap total)
    - Perbandingan antar lini produksi

Tanpa dependensi berat (cukup sqlite3 + pustaka standar) agar ringan & portabel.
"""
from __future__ import annotations

import os
import json
import sqlite3
import math
import statistics
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional

DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_store", "grading_sessions.db")

GRADE_ORDER = ["Mutu 1", "Mutu 2", "Mutu 3", "Mutu 4a", "Mutu 4b", "Mutu 5", "Mutu 6", "Di luar mutu"]


class GradingStore:
    """Penyimpanan sesi grading (SQLite)."""

    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _conn(self):
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self):
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    source TEXT,
                    line TEXT,
                    operator TEXT,
                    total_beans INTEGER,
                    defect_beans INTEGER,
                    normal_beans INTEGER,
                    defect_rate REAL,
                    defect_value_300g REAL,
                    grade_code TEXT,
                    grade_label TEXT,
                    premium INTEGER,
                    sample_weight_g REAL,
                    counts_json TEXT,
                    simulated INTEGER DEFAULT 0
                )
                """
            )

    # ------------------------------------------------------------------
    def add_session(self, grade_result: Dict, source: str = "upload",
                    line: str = "Line-1", operator: str = "PoC",
                    ts: Optional[str] = None, simulated: bool = False) -> int:
        """Catat satu hasil grading (grade_result = GradeResult.to_dict())."""
        ts = ts or datetime.now().isoformat(timespec="seconds")
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO sessions (ts, source, line, operator, total_beans,
                       defect_beans, normal_beans, defect_rate, defect_value_300g,
                       grade_code, grade_label, premium, sample_weight_g, counts_json, simulated)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    ts, source, line, operator,
                    grade_result.get("total_beans", 0),
                    grade_result.get("defect_beans", 0),
                    grade_result.get("normal_beans", 0),
                    grade_result.get("defect_rate_pct", 0.0),
                    grade_result.get("defect_value_per_300g", 0.0),
                    grade_result.get("grade_code", ""),
                    grade_result.get("grade_label", ""),
                    1 if grade_result.get("export_premium_eligible") else 0,
                    grade_result.get("estimated_sample_weight_g", 0.0),
                    json.dumps(grade_result.get("class_counts", {})),
                    1 if simulated else 0,
                ),
            )
            return cur.lastrowid

    def all_sessions(self) -> List[sqlite3.Row]:
        with self._conn() as c:
            return list(c.execute("SELECT * FROM sessions ORDER BY ts ASC"))

    def recent(self, limit: int = 15) -> List[Dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM sessions ORDER BY ts DESC LIMIT ?", (limit,))
            return [dict(r) for r in rows]

    def count(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"]

    def clear(self):
        with self._conn() as c:
            c.execute("DELETE FROM sessions")


class OperationalAnalytics:
    """Mengolah sesi grading menjadi KPI & tren untuk dashboard."""

    def __init__(self, store: GradingStore):
        self.store = store

    # ------------------------------------------------------------------
    def overview(self) -> Dict:
        rows = self.store.all_sessions()
        n = len(rows)
        if n == 0:
            return {"empty": True, "total_sessions": 0}

        total_beans = sum(r["total_beans"] for r in rows)
        defect_beans = sum(r["defect_beans"] for r in rows)
        premium = sum(1 for r in rows if r["premium"])
        defect_rates = [r["defect_rate"] for r in rows]
        defect_values = [r["defect_value_300g"] for r in rows]

        # Distribusi mutu.
        grade_dist = {g: 0 for g in GRADE_ORDER}
        for r in rows:
            grade_dist[r["grade_code"]] = grade_dist.get(r["grade_code"], 0) + 1
        grade_dist = {g: c for g, c in grade_dist.items() if c > 0}

        return {
            "empty": False,
            "total_sessions": n,
            "total_beans": total_beans,
            "avg_defect_rate": round(statistics.mean(defect_rates), 2),
            "avg_defect_value": round(statistics.mean(defect_values), 1),
            "premium_yield_pct": round(premium / n * 100.0, 1),
            "premium_sessions": premium,
            "grade_distribution": grade_dist,
            "best_grade": min((r["grade_code"] for r in rows),
                              key=lambda g: GRADE_ORDER.index(g) if g in GRADE_ORDER else 99),
            "date_range": (rows[0]["ts"][:10], rows[-1]["ts"][:10]),
        }

    # ------------------------------------------------------------------
    def daily_trend(self) -> List[Dict]:
        """Agregasi per hari: rata-rata defect rate, throughput, yield premium."""
        rows = self.store.all_sessions()
        buckets: Dict[str, List[sqlite3.Row]] = {}
        for r in rows:
            day = r["ts"][:10]
            buckets.setdefault(day, []).append(r)

        out = []
        for day in sorted(buckets):
            day_rows = buckets[day]
            n = len(day_rows)
            out.append({
                "date": day,
                "sessions": n,
                "beans": sum(r["total_beans"] for r in day_rows),
                "avg_defect_rate": round(statistics.mean([r["defect_rate"] for r in day_rows]), 2),
                "avg_defect_value": round(statistics.mean([r["defect_value_300g"] for r in day_rows]), 1),
                "premium_pct": round(sum(1 for r in day_rows if r["premium"]) / n * 100.0, 1),
            })
        return out

    # ------------------------------------------------------------------
    def defect_pareto(self) -> List[Dict]:
        """Total biji per jenis cacat di seluruh sesi (diurut menurun)."""
        agg: Dict[str, int] = {}
        for r in self.store.all_sessions():
            try:
                counts = json.loads(r["counts_json"] or "{}")
            except json.JSONDecodeError:
                counts = {}
            for label, c in counts.items():
                if label in ("Normal", "Uncertain"):
                    continue
                agg[label] = agg.get(label, 0) + int(c)
        total = sum(agg.values()) or 1
        items = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
        cum = 0.0
        out = []
        for label, c in items:
            cum += c
            out.append({
                "label": label,
                "count": c,
                "pct": round(c / total * 100.0, 1),
                "cum_pct": round(cum / total * 100.0, 1),
            })
        return out

    # ------------------------------------------------------------------
    def by_line(self) -> List[Dict]:
        rows = self.store.all_sessions()
        buckets: Dict[str, List[sqlite3.Row]] = {}
        for r in rows:
            buckets.setdefault(r["line"] or "N/A", []).append(r)
        out = []
        for line in sorted(buckets):
            lr = buckets[line]
            n = len(lr)
            out.append({
                "line": line,
                "sessions": n,
                "beans": sum(r["total_beans"] for r in lr),
                "avg_defect_rate": round(statistics.mean([r["defect_rate"] for r in lr]), 2),
                "premium_pct": round(sum(1 for r in lr if r["premium"]) / n * 100.0, 1),
            })
        return out


# ---------------------------------------------------------------------------
def seed_simulated_history(store: GradingStore, days: int = 30, per_day: int = 6,
                           seed: int = 42) -> int:
    """
    Isi basis data dengan riwayat sesi grading SIMULASI (ditandai simulated=1)
    agar dashboard analitik punya data saat demo. Pola dibuat realistis:
    mutu membaik perlahan dari waktu ke waktu, dengan variasi antar lini.

    Mengembalikan jumlah sesi yang dibuat.
    """
    import random
    rng = random.Random(seed)

    # Import engine grading agar nilai cacat konsisten dengan standar.
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from grading_standard import SNIGradeEngine
    engine = SNIGradeEngine()

    lines = ["Line-A", "Line-B", "Line-C"]
    operators = ["Andi", "Budi", "Citra", "Dewi"]
    created = 0
    start = datetime.now() - timedelta(days=days - 1)

    for d in range(days):
        day = start + timedelta(days=d)
        # Tren membaik: defect base menurun pelan seiring waktu.
        improve = d / max(1, days - 1)  # 0..1
        for _ in range(rng.randint(per_day - 2, per_day + 2)):
            line = rng.choice(lines)
            # Bias kualitas per lini (Line-A terbaik).
            line_bias = {"Line-A": 0.7, "Line-B": 1.0, "Line-C": 1.4}[line]
            total = rng.randint(1400, 1600)  # ~ jumlah biji dalam 300 g
            # Tingkat cacat dasar 1%..9%, menurun seiring tren & lini.
            base_rate = (0.09 - 0.05 * improve) * line_bias * rng.uniform(0.7, 1.3)
            base_rate = max(0.004, min(0.4, base_rate))
            defect_total = int(total * base_rate)

            # Sebar cacat ke 5 jenis dengan bobot tertentu.
            weights = {"Biji Hitam": 0.18, "Biji Cokelat": 0.22,
                       "Berlubang": 0.25, "Pecah": 0.27, "Berjamur": 0.08}
            counts = {"Normal": total - defect_total}
            allocated = 0
            labels = list(weights)
            for i, lab in enumerate(labels):
                if i == len(labels) - 1:
                    c = defect_total - allocated
                else:
                    c = int(round(defect_total * weights[lab] * rng.uniform(0.6, 1.4)))
                c = max(0, c)
                counts[lab] = c
                allocated += c
            counts["Normal"] = max(0, total - sum(v for k, v in counts.items() if k != "Normal"))

            result = engine.grade_sample(counts, sample_weight_g=300.0)
            ts = (day.replace(hour=rng.randint(7, 17), minute=rng.randint(0, 59),
                              second=rng.randint(0, 59))).isoformat(timespec="seconds")
            store.add_session(result.to_dict(), source="simulasi",
                              line=line, operator=rng.choice(operators),
                              ts=ts, simulated=True)
            created += 1
    return created


if __name__ == "__main__":
    # Buat DB demo & cetak ringkasan.
    db = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_store", "demo.db")
    if os.path.exists(db):
        os.remove(db)
    store = GradingStore(db)
    n = seed_simulated_history(store, days=21, per_day=6)
    an = OperationalAnalytics(store)
    ov = an.overview()
    print(f"Seed {n} sesi simulasi.")
    print("Overview:", json.dumps(ov, indent=2, ensure_ascii=False))
    print("Pareto cacat:", json.dumps(an.defect_pareto(), indent=2, ensure_ascii=False))
    print("Per lini:", json.dumps(an.by_line(), indent=2, ensure_ascii=False))
