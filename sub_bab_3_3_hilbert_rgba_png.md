# 3.3 Konversi Sinyal ke Citra 2D (Hilbert RGBA PNG)

Pada tahapan pra-pemrosesan ini, sinyal *Electroencephalogram* (EEG) satu dimensi (1D) ditransformasikan menjadi representasi citra dua dimensi (2D) berformat **Hilbert RGBA PNG** berdimensi tetap $32 \times 32 \times 4$ piksel. Transformasi ini dirancang agar sinyal 1D dapat diproses oleh model konvolusi citra 2D (ResUNet2D) dan dapat direkonstruksi kembali ke domain sinyal 1D secara *lossless* (*bit-exact*). 

Subbab ini menjabarkan spesifikasi teknis representasi data, langkah-langkah operasional pengkodean (*encoding*), konstruksi tensor masukan untuk pelatihan, prosedur pembacaan kembali (*decoding*), serta protokol verifikasi integritas data.

---

## 3.3.1 Parameter dan Spesifikasi Teknis Representasi

Proses konversi sinyal ke citra dikonfigurasi berdasarkan parameter dan spesifikasi teknis berikut:

1. **Dimensi Grid dan Kapasitas Titik:**
   - Orde kurva Hilbert: $k = 5$.
   - Dimensi grid spasial: $M \times M = 32 \times 32$ piksel (total 1024 titik koordinat).
   - Indeks aktif data sinyal: $d \in \{0, 1, \dots, N-1\}$.
   - Indeks metadata panjang sinyal: $d = M^2 - 1 = 1023$ (titik koordinat terakhir kurva Hilbert).
   - Kapasitas sampel maksimum: $N_{\max} = M^2 - 1 = 1023$ sampel.
   - Panjang sinyal aktual dataset: $N = 512$ sampel (baik untuk pipeline artefak EOG maupun EMG).

2. **Spesifikasi Tipe Data dan Format Memori:**
   - Format sinyal 1D masukan: *single-precision floating-point* 32-bit (`float32`, *little-endian* `<f4`).
   - Format piksel citra: 4 kanal warna (**R, G, B, A**) dengan tipe *unsigned 8-bit integer* (`uint8`, rentang nilai $0\text{--}255$).
   - Hubungan byte: 1 sampel `float32` (4 byte) dikemas tepat ke dalam 4 byte kanal warna $[R, G, B, A]$ pada satu koordinat piksel tanpa proses kuantisasi.
   - Format penyimpanan berkas: PNG (*Portable Network Graphics*) dengan mode warna `RGBA` dan kompresi *lossless* (algoritma DEFLATE).

---

## 3.3.2 Prosedur Konversi Sinyal 1D ke Citra Hilbert RGBA PNG (*Encoding*)

Tahapan operasional pengkodean dari sinyal deret waktu 1D $\mathbf{x} = [x_0, x_1, \dots, x_{N-1}]^\top \in \mathbb{R}^N$ menjadi berkas citra Hilbert RGBA PNG dilaksanakan melalui empat langkah sistematis:

### Langkah 1: Validasi dan Penyiapan Format Memori
1. Memastikan array sinyal $\mathbf{x}$ tersusun secara berurutan dan kontinu pada memori (*C-contiguous memory layout*).
2. Memastikan tipe data sinyal berada pada format *little-endian float32* (`<f4`).
3. Memvalidasi bahwa panjang sampel memenuhi batas kapasitas: $1 \le N \le 1023$.

### Langkah 2: Pembangkitan Jalur Koordinat Hilbert (`d2xy`)
Setiap indeks urutan 1D $d \in \{0, 1, \dots, 1023\}$ dipetakan ke pasangan koordinat spasial 2D $(x_d, y_d)$ pada grid $32 \times 32$. Algoritma dijalankan secara bertingkat untuk skala $s \in \{1, 2, 4, 8, 16\}$ dengan inisialisasi $t = d$, $x = 0$, $y = 0$:

1. Ekstraksi bit kuadran:
   $$r_x = 1 \ \& \ \lfloor t / 2 \rfloor$$
   $$r_y = 1 \ \& \ (t \oplus r_x)$$
2. Penerapan transformasi orientasi kuadran (`_rot`):
   - Jika $r_y = 0$ dan $r_x = 1$:
     $$x \leftarrow s - 1 - x, \qquad y \leftarrow s - 1 - y$$
   - Jika $r_y = 0$:
     $$\text{tukar}(x, y)$$
3. Akumulasi posisi dan pembaruan skala:
   $$x \leftarrow x + s \cdot r_x$$
   $$y \leftarrow y + s \cdot r_y$$
   $$t \leftarrow \lfloor t / 4 \rfloor$$
   $$s \leftarrow 2s$$

Keluaran langkah ini menghasilkan dua array pemetaan koordinat berukuran 1024 elemen:
$$\text{rows}[d] = y_d, \qquad \text{cols}[d] = x_d, \quad \forall d \in \{0, 1, \dots, 1023\}$$

### Langkah 3: Pengemasan Bit (*Bit-Packing*) ke Matriks RGBA ($32 \times 32 \times 4$)
Matriks berdimensi $32 \times 32 \times 4$ bertipe `uint8` ($\mathbf{I}_{\text{RGBA}}$) diinisialisasi dengan nilai 0, kemudian diisi melalui dua tahap:

1. **Pengisian Sinyal Aktif ($d = 0 \dots N-1$):**
   Setiap nilai sampel $x_d$ (`float32`) dikonversi secara biner (*bit-casting*, tanpa pembulatan) menjadi 4 byte `uint8` dan diposisikan pada koordinat kurva Hilbert ke-$d$:
   $$\mathbf{I}_{\text{RGBA}}[\text{rows}[d], \text{cols}[d], :] = \text{view}_{\text{uint8[4]}}(x_d), \quad \forall d \in \{0, 1, \dots, N-1\}$$

2. **Pengisian Metadata Panjang Sinyal ($d = 1023$):**
   Panjang sinyal $N$ dikonversi menjadi format `uint32` *little-endian* (`<u4`) dan disimpan pada koordinat piksel terakhir kurva Hilbert:
   $$\mathbf{I}_{\text{RGBA}}[\text{rows}[1023], \text{cols}[1023], :] = \text{view}_{\text{uint8[4]}}(N)$$
   Untuk sinyal berdurasi $N = 512$ sampel, nilai 4 byte yang tersimpan pada piksel ini adalah $[0, 2, 0, 0]$ karena:
   $$512 = 0 \times 256^0 + 2 \times 256^1 + 0 \times 256^2 + 0 \times 256^3$$

3. **Wilayah Padding ($d = N \dots 1022$):**
   Piksel pada rentang $d = 512$ hingga $1022$ tetap bernilai $[0, 0, 0, 0]$ sebagai batas akhir sinyal (*zero-padding*).

### Langkah 4: Penyimpanan Berkas Citra PNG
Matriks $\mathbf{I}_{\text{RGBA}}$ disimpan ke media penyimpanan menggunakan kompresi *lossless* PNG mode `RGBA`. Seluruh nilai byte tersimpan secara eksak tanpa mengalami perubahan nilai piksel.

---

## 3.3.3 Prosedur Pembentukan Tensor Input untuk Pelatihan Jaringan (Tensor 2-Kanal)

Untuk melatih model konvolusi 2D ResUNet2D, citra RGBA PNG dibaca kembali ke memori dan dikonversi menjadi **tensor grid kontinu 2-kanal** $\mathbf{G} \in \mathbb{R}^{2 \times 32 \times 32}$ bertipe `float32`:

1. **Konstruksi Kanal 0 — Nilai Amplitudo Sinyal Kontinu:**
   Setiap piksel aktif sepanjang kurva Hilbert diisi nilai amplitudo kontinu $x_d$ (`float32`):
   $$\mathbf{G}[0, \text{rows}[d], \text{cols}[d]] = \begin{cases} x_d, & 0 \le d < N \\ 0.0, & N \le d < 1024 \end{cases}$$

2. **Konstruksi Kanal 1 — Masker Aktif Sinyal (*Active Mask*):**
   Menandai posisi piksel aktif pembawa sinyal dengan nilai $1.0$ dan area padding dengan nilai $0.0$:
   $$\mathbf{G}[1, \text{rows}[d], \text{cols}[d]] = \begin{cases} 1.0, & 0 \le d < N \\ 0.0, & N \le d < 1024 \end{cases}$$

Tensor $\mathbf{G}$ inilah yang diumpankan ke model ResUNet2D selama proses pelatihan dan inferensi.

---

## 3.3.4 Prosedur Rekonstruksi Sinyal 1D dari Citra Hilbert RGBA PNG (*Decoding*)

Setelah model ResUNet2D menghasilkan citra hasil prediksi pembersihan $\hat{\mathbf{G}} \in \mathbb{R}^{1 \times 32 \times 32}$, sinyal EEG 1D bersih $\hat{\mathbf{x}} \in \mathbb{R}^N$ direkonstruksi kembali melalui tiga langkah berikut:

### Langkah 1: Pembacaan Berkas Citra dan Ekstraksi Metadata Panjang ($N$)
1. Berkas citra PNG dibaca ke memori sebagai array matriks $\mathbf{I}_{\text{RGBA}} \in \{0, \dots, 255\}^{32 \times 32 \times 4}$.
2. Koordinat baris dan kolom untuk piksel terakhir diakses: $r_{\text{last}} = \text{rows}[1023]$, $c_{\text{last}} = \text{cols}[1023]$.
3. Nilai 4 byte pada posisi tersebut di-*cast* balik menjadi bilangan bulat `uint32` (`<u4`) untuk memperoleh parameter panjang sinyal asli $N$:
   $$N = \text{view}_{\text{uint32}}(\mathbf{I}_{\text{RGBA}}[r_{\text{last}}, c_{\text{last}}, :])$$

### Langkah 2: Pengambilan Byte Sampel Aktif
Mengambil array byte pada koordinat kurva Hilbert dari indeks $0$ sampai $N-1$:
$$\mathbf{B} = \mathbf{I}_{\text{RGBA}}[\text{rows}[0:N], \text{cols}[0:N], :] \quad \in \{0, \dots, 255\}^{N \times 4}$$
Array $\mathbf{B}$ dipastikan tersusun kontinu pada memori komputer (*C-contiguous*).

### Langkah 3: *Bit-Casting* Balik ke Array Sinyal 1D (`float32`)
Array byte $\mathbf{B}$ berdimensi $(N, 4)$ di-*cast* balik secara biner menjadi array satu dimensi bertipe `float32` berukuran $N$:
$$\hat{\mathbf{x}} = \text{view}_{\text{float32}}(\mathbf{B}) \quad \in \mathbb{R}^N$$

Proses ini mengembalikan bentuk gelombang waktu 1D $\hat{\mathbf{x}} = [\hat{x}_0, \hat{x}_1, \dots, \hat{x}_{N-1}]$ secara presisi.

---

## 3.3.5 Prosedur Verifikasi Integritas dan Pengujian Rekonstruksi

Sebelum data digunakan dalam eksperimen pelatihan model, dilakukan prosedur pengujian integritas siklus bolak-balik (*round-trip verification*: $\mathbf{x} \rightarrow \text{Hilbert RGBA PNG} \rightarrow \hat{\mathbf{x}}$) menggunakan dua metrik evaluasi:

1. **Root Mean Square Error (RMSE) Rekonstruksi:**
   $$\text{RMSE}(\mathbf{x}, \hat{\mathbf{x}}) = \sqrt{\frac{1}{N} \sum_{d=0}^{N-1} (x_d - \hat{x}_d)^2}$$

2. **Kesetaraan Eksak Bit (*Bit-Exact Equality*):**
   $$\text{Exact Equality} = \begin{cases} \text{True}, & \forall d \in \{0, \dots, N-1\}: x_d \equiv \hat{x}_d \\ \text{False}, & \text{lainnya} \end{cases}$$

### Hasil Uji Integritas Siklus Konversi:
- **RMSE Siklus Bolak-Balik (*Round-Trip*):** $0.000 \times 10^0$ (tepat `0.0`).
- **Galat Maksimum Absolut ($\max |x_d - \hat{x}_d|$):** $0.0$.
- **Status Kesetaraan Bit:** `True` (100% identik tanpa kehilangan informasi).
- **Uji Panjang Dinamis ($N = 300$ vs $N = 512$):** Keduanya berhasil dikonversi dan direkonstruksi kembali dengan $\text{RMSE} = 0.0$.
- **Kesetaraan Isometrik:** Nilai $\text{RMSE}$ antar-sinyal pada domain 1D terbukti bernilai persis sama dengan nilai $\text{RMSE}$ pada $N = 512$ piksel aktif pada grid 2D kanal 0.

---

## 3.3.6 Diagram Alur Prosedur Konversi dan Rekonstruksi

Bagan alur tahapan konversi sinyal 1D ke citra 2D hingga rekonstruksi kembali disajikan sebagai berikut:

```text
[ Sinyal EEG 1D: x in R^N, dtype=float32 (<f4) ]
                     │
                     ▼
  (Langkah 1) Validasi format memori & panjang (1 <= N <= 1023)
                     │
                     ▼
  (Langkah 2) Pembangkitan koordinat Hilbert: d2xy(32, d)
              ──► rows[d], cols[d] untuk d = 0..1023
                     │
                     ▼
  (Langkah 3) Bit-Packing ke kanvas matriks uint8 (32 x 32 x 4):
              • d = 0..(N-1) : I_RGBA[rows[d], cols[d], :] = bytes_<f4>(x_d)
              • d = 1023     : I_RGBA[rows[1023], cols[1023], :] = bytes_<u4>(N)
              • d = N..1022  : Padding nilai nol [0, 0, 0, 0]
                     │
                     ▼
  (Langkah 4) Simpan sebagai berkas citra PNG (Lossless DEFLATE)
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
  [ Modul Pelatihan ]     [ Modul Inferensi / Rekonstruksi ]
  Dekode ke Tensor 2D     1. Baca N dari piksel d = 1023
  G in R^(2 x 32 x 32):   2. Ambil N piksel aktif sepanjang kurva
  • Ch 0: Amplitudo (f32) 3. Bit-cast balik uint8[N, 4] -> float32[N]
  • Ch 1: Masker aktif                   │
                                         ▼
                          [ Sinyal EEG 1D Rekonstruksi: x_hat ]
                          (RMSE round-trip = 0.0, Bit-Exact)
```
