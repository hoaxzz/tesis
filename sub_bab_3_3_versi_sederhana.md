# 3.3 Konversi Sinyal ke Citra 2D (Hilbert RGBA PNG)

Pada tahapan ini, sinyal EEG satu dimensi (1D) dengan panjang $N = 512$ sampel dikonversi menjadi representasi citra dua dimensi (2D) berformat **Hilbert RGBA PNG** berukuran $32 \times 32 \times 4$ piksel. Metode ini dirancang agar data deret waktu dapat direpresentasikan dalam bentuk citra berdimensi tetap secara *lossless* (*bit-exact*, tanpa kehilangan informasi sedikit pun) dan dapat dikembalikan persis ke bentuk sinyal aslinya.

---

### 3.3.1 Spesifikasi dan Format Citra

Format citra yang dibangun memiliki spesifikasi teknis sebagai berikut:

1. **Dimensi Citra:** $32 \times 32$ piksel dengan 4 kanal warna (**Red, Green, Blue, Alpha** / RGBA), menghasilkan total 1024 titik koordinat piksel.
2. **Kapasitas Sinyal:** Dapat menampung sinyal hingga $N_{\max} = 1023$ sampel ($32^2 - 1$). Pada penelitian ini, panjang sinyal yang dikonversi adalah $N = 512$ sampel.
3. **Mekanisme *Bit-Packing* dan Tipe Data `float32` (IEEE-754):**
   - **Definisi `float32` (*Single-Precision Floating-Point*):** Sinyal deret waktu EEG direpresentasikan menggunakan tipe data `float32` berbasis standar internasional **IEEE-754**. Format ini mengalokasikan lebar memori tepat **32 bit (setara 4 byte)** per sampel untuk merekam nilai amplitudo desimal kontinu, yang tersusun atas 1 bit tanda (*sign*), 8 bit eksponen (*exponent*), dan 23 bit mantisa (*fraction/significand*).
   - **Pemetaan Satu-ke-Satu (*One-to-One Bit-Casting*):** Karena 1 sampel `float32` berukuran tepat 4 byte, representasi biner mentah 4 byte tersebut dipetakan langsung (*bit-casting*) ke dalam 4 kanal warna `uint8` ($[R, G, B, A]$, rentang nilai 0 s.d. 255, masing-masing 1 byte) pada satu piksel. Pendekatan ini tidak menggunakan pembulatan nilai desimal maupun kuantisasi diskret, sehingga resolusi mikrovolt kontinu sinyal EEG terjaga secara utuh (*bit-exact*).
4. **Piksel Metadata Panjang Sinyal:**  
   Piksel terakhir pada kurva Hilbert (indeks ke-1023) dialokasikan khusus untuk menyimpan panjang asli sinyal ($N$) dalam format bilangan bulat 32-bit (`uint32`).
5. **Format Berkas:**  
   Citra disimpan dalam format berkas `.png` menggunakan kompresi *lossless* (DEFLATE).

---

### 3.3.2 Langkah-Langkah Pembuatan Citra (*Encoding: Sinyal 1D ke Citra 2D*)

Tahapan pembuatan citra dari sinyal EEG 1D dilakukan melalui 4 langkah operasional:

1. **Langkah 1 — Pembangkitan Jalur Koordinat Hilbert (`d2xy`):**  
   Menghitung pemetaan dari indeks urutan 1D ($d = 0, 1, \dots, 1023$) ke pasangan koordinat 2D $(\text{baris}[d], \text{kolom}[d])$ pada grid $32 \times 32$ menggunakan algoritma kurva pengisi ruang Hilbert. Algoritma ini memastikan bahwa titik-titik sampel yang bersebelahan pada domain waktu tetap berada pada posisi yang berdekatan di ruang citra 2D.

2. **Langkah 2 — Pengemasan Sampel Sinyal ke Kanal RGBA:**  
   Sebanyak $N = 512$ sampel sinyal (`float32`) dikemas ke dalam byte mentah `uint8`, kemudian ditempatkan pada koordinat kurva Hilbert dari indeks $d = 0$ hingga $d = 511$:
   $$\mathbf{I}_{\text{RGBA}}[\text{baris}[d], \text{kolom}[d], :] = \text{byte}_{\text{float32}}(x_d), \quad \forall d \in \{0, 1, \dots, 511\}$$

3. **Langkah 3 — Penyimpanan Metadata dan Pengisian Padding:**  
   - **Penyimpanan Metadata:** Nilai panjang sinyal $N = 512$ dikonversi menjadi format `uint32` (4 byte: `[0, 2, 0, 0]`) dan disimpan pada piksel terakhir kurva Hilbert (indeks $d = 1023$):
     $$\mathbf{I}_{\text{RGBA}}[\text{baris}[1023], \text{kolom}[1023], :] = \text{byte}_{\text{uint32}}(512)$$
   - **Pengisian Padding:** Koordinat kurva yang tidak terpakai (indeks $d = 512$ sampai $1022$) diisi dengan nilai nol $[0, 0, 0, 0]$ (*zero-padding*).

4. **Langkah 4 — Ekspor Berkas PNG:**  
   Matriks array RGBA berukuran $32 \times 32 \times 4$ diekspor dan disimpan ke media penyimpanan sebagai berkas citra PNG mode warna `RGBA`.

---

### 3.3.3 Langkah-Langkah Rekonstruksi (*Decoding: Citra 2D ke Sinyal 1D*)

Untuk mengembalikan citra Hilbert RGBA PNG menjadi sinyal deret waktu 1D asli, dilakukan 3 langkah inversi:

1. **Langkah 1 — Pembacaan Panjang Sinyal dari Metadata:**  
   Membaca berkas PNG, lalu mengakses 4 byte pada piksel terakhir kurva Hilbert (indeks $d = 1023$). Empat byte tersebut diubah kembali (*bit-casting*) menjadi bilangan bulat `uint32` untuk memperoleh nilai panjang sinyal asli $N = 512$.

2. **Langkah 2 — Ekstraksi Piksel Aktif:**  
   Mengambil data byte secara berurutan dari $N = 512$ piksel pertama sepanjang jalur koordinat kurva Hilbert ($d = 0$ sampai $511$).

3. **Langkah 3 — Rekonstruksi ke Sinyal 1D (`float32`):**  
   Array byte berdimensi $(512, 4)$ di-*cast* balik menjadi bilangan desimal `float32` sehingga menghasilkan kembali sinyal deret waktu 1D:
   $$\hat{\mathbf{x}} = \text{float32}(\text{byte aktif}) \quad \in \mathbb{R}^{512}$$

---

### 3.3.4 Verifikasi Integritas Rekonstruksi (*Round-Trip Test*)

Pengujian dilakukan untuk membuktikan bahwa proses pembentukan citra dan rekonstruksi baliknya ($\text{Sinyal 1D} \rightarrow \text{Citra RGBA PNG} \rightarrow \text{Sinyal 1D}$) tidak menyebabkan distorsi maupun perubahan nilai data sedikit pun:

| Parameter Pengujian | Nilai Hasil Uji | Keterangan |
| :--- | :---: | :--- |
| **Root Mean Square Error (RMSE)** | **0.000** | Galat antara sinyal awal dan sinyal hasil rekonstruksi bernilai nol. |
| **Max Absolute Error ($\max \lvert x - \hat{x} \rvert$)** | **0.000** | Seluruh titik sampel bernilai identik pada tingkat bit (*bit-exact*). |
| **Status Kesetaraan Bit (`np.array_equal`)** | **True** | Tidak ada kehilangan informasi selama proses simpan dan baca PNG. |
| **Uji Panjang Dinamis ($N = 300$)** | **RMSE = 0.000** | Format citra tetap konsisten dan dapat direkonstruksi secara eksak. |

---

### 3.3.5 Diagram Alur Konversi dan Rekonstruksi Citra

```text
Sinyal EEG 1D (512 sampel, float32)
       │
       ▼
[ Langkah 1: Hitung Koordinat Kurva Hilbert 32x32 ]
  ──► baris[d] dan kolom[d] untuk d = 0..1023
       │
       ▼
[ Langkah 2 & 3: Bit-Packing ke Matriks RGBA uint8 ]
  • Titik d = 0..511  : 4 byte per sampel float32
  • Titik d = 512..1022: Zero-padding [0, 0, 0, 0]
  • Titik d = 1023    : Metadata panjang N = 512 (uint32)
       │
       ▼
[ Langkah 4: Simpan ke Berkas Citra ]
  ──► File Citra Hilbert RGBA PNG (32 × 32 × 4)
       │
       ▼
[ Proses Rekonstruksi Sinyal (Decoding) ]
  1. Baca N = 512 dari piksel d = 1023
  2. Ekstrak byte dari 512 titik kurva Hilbert aktif
  3. Konversi balik byte uint8 -> float32
       │
       ▼
Sinyal EEG 1D Hasil Rekonstruksi (RMSE = 0.0, Bit-Exact)
```
