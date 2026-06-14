"""
SCAA (Specialty Coffee Association of America) Defect Score Calculator.
Modul ini bertugas menghitung total 'Full Defects' berdasarkan
jumlah cacat individu dalam suatu sampel kopi (misal 300 gram).
"""

class SCAACalculator:
    def __init__(self):
        # Pembobotan (Berapa banyak cacat individu untuk menjadi 1 Full Defect)
        # Sesuai standar SCAA Green Arabica Defect Handbook (simplified untuk 6 kelas)
        self.defect_weights = {
            'Biji Hitam': 1.0,     # 1 Full Black = 1 Full Defect
            'Berjamur': 1.0,       # 1 Fungus = 1 Full Defect (kategori berat)
            'Biji Cokelat': 0.2,   # 5 Partial Sour/Cokelat = 1 Full Defect
            'Berlubang': 0.2,      # 5 Insect Damage = 1 Full Defect
            'Pecah': 0.2           # 5 Broken/Chipped = 1 Full Defect
            # Normal tidak memiliki bobot cacat
        }
    
    def calculate_full_defects(self, defect_counts):
        """
        Menghitung total Full Defects dari kamus perhitungan cacat.
        
        Args:
            defect_counts (dict): Kamus berisi {'Nama Cacat': jumlah}
            
        Returns:
            float: Total nilai Full Defects
        """
        total_defects = 0.0
        for defect_name, count in defect_counts.items():
            if defect_name in self.defect_weights:
                total_defects += count * self.defect_weights[defect_name]
        return total_defects
        
    def determine_grade(self, total_full_defects):
        """
        Menentukan kelas/grade akhir berdasarkan Total Full Defects.
        (Berdasarkan skala simplifikasi SCAA untuk sampel 300g)
        
        Args:
            total_full_defects (float): Skor total cacat
            
        Returns:
            str: Label kelas (Specialty, Premium, Exchange, dll)
        """
        # Dalam standar SCAA murni, Specialty Grade harus memiliki 0 Primary Defects.
        # Untuk menyederhanakan, kita gunakan cut-off berdasarkan jumlah Full Defects.
        if total_full_defects <= 5.0:
            return "Specialty Grade"
        elif total_full_defects <= 8.0:
            return "Premium Grade"
        elif total_full_defects <= 23.0:
            return "Exchange Grade"
        else:
            return "Below Standard (Reject)"

    def is_primary_defect(self, defect_name):
        """
        Memeriksa apakah sebuah cacat masuk kategori Primary Defect.
        Primary defects sangat fatal dan tidak ditolerir di Specialty Grade.
        """
        primary_defects = ['Biji Hitam', 'Berjamur']
        return defect_name in primary_defects
