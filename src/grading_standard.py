"""
Mesin Grading Mutu Biji Kopi — SNI 01-2907-2008 (Sistem Nilai Cacat).

Modul ini adalah "keystone" Smart Grading yang menjembatani hasil
KLASIFIKASI CACAT PER-BIJI (model EfficientNet milik tim) menjadi
PENILAIAN MUTU LOT sesuai standar resmi Indonesia.

Standar acuan:
    SNI 01-2907-2008 "Biji kopi" (menggantikan SNI 01-2907-1999).
    Penggolongan mutu didasarkan pada SISTEM NILAI CACAT yang dihitung
    dari contoh (sample) seberat 300 gram.

    Tabel 1 — Penggolongan mutu (jumlah nilai cacat per 300 g):
        Mutu 1  : maksimum 11
        Mutu 2  : 12 - 25
        Mutu 3  : 26 - 44
        Mutu 4a : 45 - 60
        Mutu 4b : 61 - 80
        Mutu 5  : 81 - 150
        Mutu 6  : 151 - 225

    Tabel 2 — Bobot nilai cacat per jenis cacat (sebagian, yang relevan
    dengan 6 kelas yang dideteksi sistem visi komputer):
        1 biji hitam                 = 1
        1 biji hitam sebagian        = 1/2
        1 biji hitam pecah           = 1/2
        1 kopi gelondong             = 1
        1 biji cokelat               = 1/4
        1 kulit kopi besar           = 1
        1 biji pecah                 = 1/5
        1 biji muda                  = 1/5
        1 biji berlubang satu        = 1/10
        1 biji berlubang > satu      = 1/5
        1 biji bertutul-tutul (jamur)= 1/10
        1 ranting/tanah/batu kecil   = 1   (besar = 5, sedang = 2)

Syarat mutu umum (SNI): bebas serangga hidup; bebas biji berbau busuk /
berbau kapang; kadar air maks. 12,5%; kadar kotoran maks. 0,5%.

Catatan implementasi:
    Sebuah gambar tunggal jarang berisi tepat 300 g biji. Engine ini
    MENGEKSTRAPOLASI nilai cacat sampel ke basis 300 g menggunakan estimasi
    massa rata-rata biji (default Robusta ~0,19 g/biji) sehingga keputusan
    mutu tetap bermakna meski sampel kecil. Semua parameter dapat dikonfigurasi.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Tabel 2 SNI 01-2907-2008 — bobot nilai cacat per jenis cacat.
# Kunci di-selaraskan dengan Config.GRADE_LABELS (6 kelas defect tim).
# 'Normal' tidak menyumbang nilai cacat.
# Untuk "Berlubang" dipakai 1/5 (asumsi konservatif: berlubang lebih dari satu).
# ---------------------------------------------------------------------------
SNI_DEFECT_VALUES: Dict[str, float] = {
    "Normal": 0.0,
    "Biji Hitam": 1.0,      # biji hitam penuh
    "Biji Cokelat": 0.25,   # 1/4 (over-fermentasi / cokelat)
    "Berlubang": 0.2,       # 1/5 (berlubang lebih dari satu); 1/10 bila satu lubang
    "Pecah": 0.2,           # 1/5 (biji pecah)
    "Berjamur": 0.1,        # 1/10 (biji bertutul-tutul / berjamur)
}

# Tabel jenis cacat SNI lengkap (referensi & bila ingin grading manual lebih detail).
SNI_DEFECT_VALUES_FULL: Dict[str, float] = {
    "biji hitam": 1.0,
    "biji hitam sebagian": 0.5,
    "biji hitam pecah": 0.5,
    "kopi gelondong": 1.0,
    "biji cokelat": 0.25,
    "kulit kopi besar": 1.0,
    "kulit kopi sedang": 0.5,
    "kulit kopi kecil": 0.2,
    "biji berkulit tanduk": 0.5,
    "kulit tanduk besar": 0.5,
    "kulit tanduk sedang": 0.2,
    "kulit tanduk kecil": 0.1,
    "biji pecah": 0.2,
    "biji muda": 0.2,
    "biji berlubang satu": 0.1,
    "biji berlubang lebih dari satu": 0.2,
    "biji bertutul-tutul": 0.1,
    "ranting/tanah/batu besar": 5.0,
    "ranting/tanah/batu sedang": 2.0,
    "ranting/tanah/batu kecil": 1.0,
}

# Tabel 1 SNI — (batas_atas_inklusif, kode, nama, layak_ekspor_premium)
# Diurut dari mutu terbaik ke terendah.
SNI_GRADE_TABLE = [
    (11.0, "Mutu 1", "Grade 1", True),
    (25.0, "Mutu 2", "Grade 2", True),
    (44.0, "Mutu 3", "Grade 3", False),
    (60.0, "Mutu 4a", "Grade 4a", False),
    (80.0, "Mutu 4b", "Grade 4b", False),
    (150.0, "Mutu 5", "Grade 5", False),
    (225.0, "Mutu 6", "Grade 6", False),
]

# Massa rata-rata satu biji kopi (gram). Robusta kering ~0,18-0,20 g.
DEFAULT_BEAN_MASS_G = 0.19
SNI_SAMPLE_WEIGHT_G = 300.0


@dataclass
class GradeResult:
    """Hasil penilaian mutu satu sampel/lot."""
    grade_code: str                       # mis. "Mutu 1"
    grade_label: str                      # mis. "Grade 1"
    defect_value_sample: float            # nilai cacat pada sampel terdeteksi
    defect_value_per_300g: float          # nilai cacat diekstrapolasi ke 300 g
    total_beans: int
    defect_beans: int
    normal_beans: int
    defect_rate_pct: float                # % biji cacat
    estimated_sample_weight_g: float
    class_counts: Dict[str, int] = field(default_factory=dict)
    defect_value_breakdown: Dict[str, float] = field(default_factory=dict)
    export_premium_eligible: bool = False
    recommendation: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class SNIGradeEngine:
    """
    Menerjemahkan jumlah biji per-kelas-cacat menjadi mutu SNI 01-2907-2008.

    Contoh:
        engine = SNIGradeEngine()
        result = engine.grade_sample({"Normal": 280, "Biji Hitam": 3, "Pecah": 10})
        print(result.grade_code, result.defect_value_per_300g)
    """

    def __init__(
        self,
        defect_values: Optional[Dict[str, float]] = None,
        grade_table=None,
        bean_mass_g: float = DEFAULT_BEAN_MASS_G,
        sample_weight_g: float = SNI_SAMPLE_WEIGHT_G,
    ):
        self.defect_values = dict(defect_values or SNI_DEFECT_VALUES)
        self.grade_table = list(grade_table or SNI_GRADE_TABLE)
        self.bean_mass_g = float(bean_mass_g)
        self.sample_weight_g = float(sample_weight_g)

    # ------------------------------------------------------------------
    def defect_value_for_class(self, class_label: str) -> float:
        """Bobot nilai cacat untuk satu kelas (0 bila tidak dikenal / normal)."""
        return float(self.defect_values.get(class_label, 0.0))

    # ------------------------------------------------------------------
    def _grade_from_value(self, defect_value_300g: float):
        """Petakan nilai cacat per 300 g ke (kode, label, layak_premium)."""
        for upper, code, label, premium in self.grade_table:
            if defect_value_300g <= upper:
                return code, label, premium
        # Di atas batas tabel (>225) — di luar mutu SNI.
        return "Di luar mutu", "Reject", False

    # ------------------------------------------------------------------
    def grade_sample(
        self,
        class_counts: Dict[str, int],
        sample_weight_g: Optional[float] = None,
        extrapolate_to_300g: bool = True,
    ) -> GradeResult:
        """
        Hitung mutu dari jumlah biji per kelas.

        Args:
            class_counts: {label_kelas: jumlah_biji}. Label harus sesuai
                Config.GRADE_LABELS (mis. "Normal", "Biji Hitam", ...).
            sample_weight_g: berat sampel sebenarnya (gram). Bila None,
                diestimasi dari jumlah biji × massa rata-rata biji.
            extrapolate_to_300g: bila True, nilai cacat diskalakan ke basis
                300 g (sesuai metode SNI). Bila False, dipakai apa adanya.

        Returns:
            GradeResult.
        """
        class_counts = {k: int(v) for k, v in class_counts.items() if int(v) > 0}
        total_beans = sum(class_counts.values())

        # Nilai cacat pada sampel terdeteksi + rincian kontribusi.
        breakdown: Dict[str, float] = {}
        defect_value_sample = 0.0
        defect_beans = 0
        for label, count in class_counts.items():
            unit_value = self.defect_value_for_class(label)
            contribution = unit_value * count
            if unit_value > 0:
                breakdown[label] = round(contribution, 4)
                defect_beans += count
            defect_value_sample += contribution

        normal_beans = total_beans - defect_beans
        defect_rate = (defect_beans / total_beans * 100.0) if total_beans else 0.0

        # Estimasi berat sampel.
        if sample_weight_g is None or sample_weight_g <= 0:
            est_weight = total_beans * self.bean_mass_g
        else:
            est_weight = float(sample_weight_g)

        # Ekstrapolasi nilai cacat ke 300 g.
        if extrapolate_to_300g and est_weight > 0:
            scale = self.sample_weight_g / est_weight
            defect_value_300g = defect_value_sample * scale
        else:
            defect_value_300g = defect_value_sample

        grade_code, grade_label, premium = self._grade_from_value(defect_value_300g)

        notes: List[str] = []
        if total_beans == 0:
            notes.append("Tidak ada biji terdeteksi pada sampel.")
        if sample_weight_g is None:
            notes.append(
                f"Berat sampel diestimasi dari {total_beans} biji × "
                f"{self.bean_mass_g:g} g = {est_weight:.1f} g, lalu "
                f"diskalakan ke basis 300 g sesuai SNI."
            )
        if total_beans < 100:
            notes.append(
                "Sampel < 100 biji: hasil bersifat indikatif. Untuk grading "
                "resmi gunakan contoh 300 g (umumnya ~1500 biji)."
            )

        return GradeResult(
            grade_code=grade_code,
            grade_label=grade_label,
            defect_value_sample=round(defect_value_sample, 3),
            defect_value_per_300g=round(defect_value_300g, 2),
            total_beans=total_beans,
            defect_beans=defect_beans,
            normal_beans=normal_beans,
            defect_rate_pct=round(defect_rate, 1),
            estimated_sample_weight_g=round(est_weight, 1),
            class_counts=class_counts,
            defect_value_breakdown=breakdown,
            export_premium_eligible=premium,
            recommendation=self._recommendation(grade_code, premium, defect_rate),
            notes=notes,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _recommendation(grade_code: str, premium: bool, defect_rate: float) -> str:
        if grade_code in ("Mutu 1", "Mutu 2"):
            return (
                "Lolos kualifikasi mutu ekspor tinggi (premium). "
                "Lanjutkan ke proses pengemasan/ekspor."
            )
        if grade_code in ("Mutu 3", "Mutu 4a", "Mutu 4b"):
            return (
                "Mutu komersial standar. Pertimbangkan sortasi ulang untuk "
                "menaikkan grade bila target pasar premium."
            )
        if grade_code in ("Mutu 5", "Mutu 6"):
            return (
                "Mutu rendah. Disarankan re-sortasi/re-processing sebelum "
                "didistribusikan. Tidak memenuhi syarat ekspor premium."
            )
        return "Di luar penggolongan mutu SNI — tolak / proses ulang menyeluruh."


# Util cepat untuk dipakai modul lain ---------------------------------------
def grade_from_counts(class_counts: Dict[str, int], **kwargs) -> GradeResult:
    """Shortcut: buat engine default lalu nilai sampel."""
    return SNIGradeEngine().grade_sample(class_counts, **kwargs)


if __name__ == "__main__":
    # Demonstrasi cepat tiga skenario.
    engine = SNIGradeEngine()
    demos = {
        "Lot premium (sedikit cacat)": {"Normal": 1490, "Pecah": 8, "Biji Cokelat": 4},
        "Lot komersial": {"Normal": 1400, "Biji Hitam": 20, "Pecah": 40, "Berlubang": 30},
        "Lot mutu rendah": {"Normal": 1000, "Biji Hitam": 200, "Berjamur": 150, "Pecah": 150},
    }
    for name, counts in demos.items():
        r = engine.grade_sample(counts, sample_weight_g=300.0)
        print(f"\n{name}")
        print(f"  Nilai cacat / 300g : {r.defect_value_per_300g}")
        print(f"  Mutu               : {r.grade_code} ({r.grade_label})")
        print(f"  % biji cacat       : {r.defect_rate_pct}%")
        print(f"  Layak premium      : {r.export_premium_eligible}")
        print(f"  Rekomendasi        : {r.recommendation}")
