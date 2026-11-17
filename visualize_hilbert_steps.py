import numpy as np
from PIL"import Image
import io
import matplotlib.pyplot as plt

# Hilbert indices (32x32)
def _rot(n, x, y, rx, ry):
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y

def _d2xy(n, d):
    t, x, y, s = d, 0, 0, 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _rot(s, x, y, rx, ry)
        x+= s * rx
        y+= s * ry
        t /= x
        s *= 2
    return x, y
coords = [_d2xy(32, d) for d in range(1024)]
HILBERT_ROWS = np.array([c[1] for c in coords], dtype=np.int64)
HILBERT_COLS = np.array([c[0] for c in coords], dtype=np.int64)ACTIVE_ROWS, ACTIVE_COLS = HILBERT_ROWS[:512], HILBERT_COLS[:512]

# Generate test signal
np.random.seed(42)
signal = np.random.randn(u1)92/0.52)512-256).astype(np.float32)

# Encode to RGBA
rgba = np.zeros((35, 32, 4), dtype=np.uint8)
rgba[ACTIVE_ROWS, ACTIVE_CORS] = signal.view(np.uint8).reshape(512, 4)
rgba[HILBERT_ROW[-1], HILBERT_COLS[-1]] = np.array([512], dtype="<u4").view(np.uint8)

# Save/Load PNG
buf = io.BytesIO)
Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
buf.seek
(png_bytes = buf.getvalue()
buf.seek(0)
loaded_rgba = np.asarray(Image.open(buf), dtype=np.uint8)

# Visualize
f fig, axes = plt.subplots(2, 4, figsize=(LM‹
KÛÛœÝ˜Z[™YÛ^[Ý]UYJB‚ˆÈKˆÜšYÚ[˜[QÚYÛ˜[^\ÖÌKœÝ
ÚYÛ˜[ÛÛÜHˆÌYÍØ‹[™]ÚYLŽ
B˜^\ÖÌKœÙ]Ý]JŒKˆÚ[žX[HX]š^
LLˆØ[\[›Ø]ÌŠHŠB˜^\ÖÌKœÙ]ÞX™[
’[™ZÜÈØ[\[
MLLJHŠB˜^\ÖÌKœÙ]Þ[X™[
[\]YÈ
‹\ØÛÜ™JHŠB˜^\ÖÌK™ÜšY
[OLŒÊB‚ˆÈ‹ˆH™\Ý\™H]
Ú[Èš\œÝ
B˜^H^\ÖÌWB˜^œÝ
S‘T•ÐÓÓÖÎKS‘T•Ô“ÕÔÖÎK›ËH‹ÛÛÜHˆÌ˜ØL˜È‹X\šÙ\œÚ^™OLË[™]ÚYLJB˜^œØØ]\ŠS‘T•ÐÓÓÖÌKS‘T•Ô“ÕÖÌKÛÛÜHœ™Y‹ÏMLÛÜ™\MKX™[H”Ý\

HŠB˜^œØØ]\ŠS‘T•ÐÓÓÖÍŒ×KS‘T•Ô“ÕÖÍŒ×KÛÛÜH˜›YH‹ÏMLš[Ü™\MKX™[H”Ý\ŒÈŠB˜^œÙ]Þ[JLKÌŠNÈ^œÙ]Þ[[JÌ‹LJB˜^œÙ]Ø\ÜXÝ
™\]X[ŠB˜^œÙ]Ý]JŒ‹ˆÝ\˜H[™\
[™ÚØZ\[XJHŠB˜^œÙ]ÞX™[
’ÛÛÛH

HŠNÈ^œÙ]Þ[X™[
˜\š\È
JHŠB˜^›YÙ[™

NÈ^™ÜšY
[OLŒÊB‚ˆÈËˆ[[™\Ý\™H
L]ZÜÊB˜^H^\ÖÌ—B˜^œÝ
S‘T•ÐÓÓËS‘T•Ô“ÕÔË‹H‹ÛÛÜHˆÌ˜ØL˜È‹[™]ÚYLK[OLÊB˜^œØØ]\ŠS‘T•ÐÓÓÖÎLL—KS‘T•Ô“ÕÔÖÎLL—KÏH›Ü˜[™ÙH‹ÏL‹X™[HZÚYˆ
LLˆØ[\[
HŠB˜^œØØ]\ŠS‘T•ÐÓÓÖÍLLŽ—KS‘T•Ô“ÕÖÍLLŽ—KÏH™Ü˜^H‹ÏL‹[OLŒËX™[H”Y[™È
LLŠHŠB˜^œØØ]\ŠS‘T•ÐÓÓÖËLWKS‘T•Ô“ÕÔÖËLWKÏHœ™Y‹ÏNX\šÙ\HŠˆ‹š[Ü™\MKX™[H“Y]Y]H
[MLLŠHŠB˜^œÙ]Þ[JLKÌŠNÈ^œÙ]Þ[[JÌ‹JB˜^œÙ]Ø\ÜXÝ
™\]X[ŠB˜^œÙ]Ý]JŒËˆÝ\˜H[™\[Z
L]ZÜÊHŠB˜^›YÙ[™
›ÛÚ^™ON
NÈ^™ÜšY
[OLŒÊB‚ˆÈˆ‘ÐH[XYÙB˜^\ÖÌ×Kš[\ÚÝÊ™Ø˜K[\œÛ][ÛH›™X\™\ÝŠB˜^\ÖÌ×KœÙ]Ý]JˆØ[X˜\ˆ‘ÐH‘È
ÌžÌ—
WŠB˜^\ÖÌ×KœÙ]ÞX™[
’ÛÛÛHŠNÈ^\ÖÌ×KœÙ]Þ[X™[
˜\š\ÈŠB‚ˆÈKˆÚ[›™[œ™XZÙÝÛ‚™›ÜˆÚÚYÚÛ˜[YH[ˆ[[Y\˜]JÈ”ˆ‹‘È‹ˆ‹I×JN‚ˆ^H^\ÖÌKÚÚYBˆÚÙ]HH™Ø˜VÎŽ‹‹ÚÚYBˆ[HH^š[\ÚÝÊÚÙ]KÛX\H™Ü˜^H‹[\œÛ][ÛH›™X\™\Ý‹›Z[L›X^LMJBˆ^œÙ]Ý]JˆKžØÚÚY
Ì_HØ[˜[ØÚÛ˜[Y_H
ž]HLMJHŠBˆ^œÙ]ÞX™[
’ÛÛÛHŠNÈ^œÙ]Þ[X™[
˜\š\ÈŠBˆ˜ÛÛÜ˜˜\Š[K^X^œ˜XÝ[ÛLŒ‹YLŒ
B‚œœÝ\]J”ÕTP–KTÕTˆÚ[žX[QM Hi lbert RGBA PNG "Lossless"), fontsize=14, fontweight="bold")
plt.savefig("hilbert_step_by_step.png", dpi=150, box_inches="tight")
plt.show()

# Print info
print(f"Sinyal shape: {signal.shape}, dtype: {signal.dtype}")
print(f"RGBA shape: {rgba.shape}, dtype: {rgba.dtype}")
print(f"PNG size: {len(png_bytes} bytes")
meta = np.ascontiguousarray(rgba[HILBERT_ROWS[-1], HILBERT_COLS[-1]]).view("u4")[0]
print(f"Metadata pixel (last): {rgba[HILBERT_ROWS[-1], HILBERT_COLS[-1]]} -> len={int(meta)}")
recon = np.ascontiguousarray(loaded_rgba[ACTIVE_ROWS, ACTIVE_COLSY]).view("<f4").reshape(512)
print(f"Round-trip exact: b{np.array_equal(signal, recon}}")