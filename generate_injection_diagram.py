import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(16, 11), dpi=200)
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.axis("off")

fig.patch.set_facecolor("#F8FAFC")
ax.set_facecolor("#F8FAFC")

# Title
ax.text(8, 10.4, "PIPELINE SINTESIS DATASET EEGDENOISENET: INJEKSI NOISE & ALOKASI EPOCH", 
        ha="center", va="center", fontsize=15, fontweight="bold", color="#0F172A")
ax.text(8, 10.05, "Visualisasi Pairing 1-to-1, Pengacakan Bebas Duplikasi (Permutasi), & Penanganan Sisa Epoch", 
        ha="center", va="center", fontsize=11, color="#475569")

x_emg = 2.4
x_eeg = 8.0
x_eog = 13.6

def draw_dataset_block(ax, x, y_top, y_bottom, width, color, edge_color, title, subtitle):
    rect = patches.FancyBboxPatch((x - width/2, y_bottom), width, y_top - y_bottom,
                                  boxstyle="round,pad=0.1,rounding_size=0.15",
                                  facecolor=color, edgecolor=edge_color, linewidth=2, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y_top + 0.35, title, ha="center", va="center", fontsize=12.5, fontweight="bold", color="#1E293B")
    ax.text(x, y_top + 0.1, subtitle, ha="center", va="center", fontsize=9.5, color="#64748B")

y_top_all = 9.2
h_4514 = 5.2    # represents 4514 epochs
h_3400 = h_4514 * (3400 / 4514)  # ~3.91 represents 3400 epochs
h_1114 = h_4514 - h_3400          
h_5598 = h_4514 * (5598 / 4514)  # represents 5598 epochs
h_1084 = h_5598 - h_4514          

# --- 1. MIDDLE: EEG CLEAN COLUMN ---
draw_dataset_block(ax, x_eeg, y_top_all, y_top_all - h_4514, 2.8, "#EFF6FF", "#2563EB", 
                   "EEG Clean (Ground Truth)", "Total: 4.514 Epochs (Indeks 0 s.d. 4513)")

y_eeg_split = y_top_all - h_3400
rect_eeg_used_eog = patches.Rectangle((x_eeg - 1.3, y_eeg_split), 2.6, h_3400,
                                      facecolor="#DBEAFE", edgecolor="#3B82F6", linestyle="--", linewidth=1.5, zorder=3)
ax.add_patch(rect_eeg_used_eog)
ax.text(x_eeg, y_top_all - 0.4, "Epoch 0 (Pasang ke-1)", ha="center", va="center", fontsize=8.5, color="#1E3A8A", zorder=4)
ax.text(x_eeg, y_top_all - 0.8, "Epoch 1 (Pasang ke-2)", ha="center", va="center", fontsize=8.5, color="#1E3A8A", zorder=4)
ax.text(x_eeg, y_top_all - 1.2, "Epoch 2 (Pasang ke-3)", ha="center", va="center", fontsize=8.5, color="#1E3A8A", zorder=4)
ax.text(x_eeg, y_top_all - 1.6, "...", ha="center", va="center", fontsize=10, fontweight="bold", color="#1E3A8A", zorder=4)
ax.text(x_eeg, y_eeg_split + 0.8, "Epoch 3398", ha="center", va="center", fontsize=8.5, color="#1E3A8A", zorder=4)
ax.text(x_eeg, y_eeg_split + 0.4, "Epoch 3399 (Pasang ke-3400)", ha="center", va="center", fontsize=8.5, color="#1E3A8A", zorder=4)

ax.text(x_eeg, (y_top_all + y_eeg_split)/2, "3.400 Epoch Dipakai\n(untuk EOG & EMG)", 
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1D4ED8",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#EFF6FF", edgecolor="#3B82F6", alpha=0.85), zorder=5)

rect_eeg_rem_eog = patches.Rectangle((x_eeg - 1.3, y_top_all - h_4514), 2.6, h_1114,
                                     facecolor="#FEF3C7", edgecolor="#D97706", linestyle=":", linewidth=1.5, zorder=3)
ax.add_patch(rect_eeg_rem_eog)
ax.text(x_eeg, y_eeg_split - 0.35, "Epoch 3400", ha="center", va="center", fontsize=8.5, color="#92400E", zorder=4)
ax.text(x_eeg, y_eeg_split - 0.65, "...", ha="center", va="center", fontsize=9, fontweight="bold", color="#92400E", zorder=4)
ax.text(x_eeg, y_top_all - h_4514 + 0.3, "Epoch 4513 (Pasang ke-4514)", ha="center", va="center", fontsize=8.5, color="#92400E", zorder=4)
ax.text(x_eeg, y_top_all - h_4514 + h_1114/2, "1.114 Epoch Sisa EEG\nDipakai di EMG, TIDAK di EOG", 
        ha="center", va="center", fontsize=8.5, fontweight="bold", color="#B45309",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFBEB", edgecolor="#F59E0B", alpha=0.9), zorder=5)

# --- 2. LEFT: EMG COLUMN (Total 5598) ---
y_emg_bottom = y_top_all - h_5598
draw_dataset_block(ax, x_emg, y_top_all, y_emg_bottom, 2.6, "#FEE2E2", "#DC2626", 
                   "EMG Artefak (Otot)", "Total: 5.598 Epochs (Indeks 0 s.d. 5597)")

y_emg_split = y_top_all - h_4514
rect_emg_used = patches.Rectangle((x_emg - 1.2, y_emg_split), 2.4, h_4514,
                                  facecolor="#FEE2E2", edgecolor="#EF4444", linewidth=1.5, zorder=3)
ax.add_patch(rect_emg_used)
ax.text(x_emg, y_top_all - 0.4, "EMG[401] (Acak #1)", ha="center", va="center", fontsize=8.5, color="#991B1B", zorder=4)
ax.text(x_emg, y_top_all - 0.8, "EMG[3674] (Acak #2)", ha="center", va="center", fontsize=8.5, color="#991B1B", zorder=4)
ax.text(x_emg, y_top_all - 1.2, "EMG[296] (Acak #3)", ha="center", va="center", fontsize=8.5, color="#991B1B", zorder=4)
ax.text(x_emg, y_top_all - 1.6, "...", ha="center", va="center", fontsize=10, fontweight="bold", color="#991B1B", zorder=4)
ax.text(x_emg, y_emg_split + 0.4, "EMG[327] (Acak #4514)", ha="center", va="center", fontsize=8.5, color="#991B1B", zorder=4)

ax.text(x_emg, (y_top_all + y_emg_split)/2, "4.514 Epoch Terpilih\n(Shuffled tanpa duplikasi)", 
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="#B91C1C",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2", edgecolor="#EF4444", alpha=0.9), zorder=5)

rect_emg_unused = patches.Rectangle((x_emg - 1.2, y_emg_bottom), 2.4, h_1084,
                                    facecolor="#F1F5F9", edgecolor="#94A3B8", linestyle=":", hatch="//", linewidth=1.5, zorder=3)
ax.add_patch(rect_emg_unused)
ax.text(x_emg, y_emg_bottom + h_1084/2, "1.084 Epoch Sisa EMG\nDIBUANG / TIDAK DIPAKAI\n(Cegah Data Leakage)", 
        ha="center", va="center", fontsize=8.5, fontweight="bold", color="#64748B",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.95), zorder=5)

# --- 3. RIGHT: EOG COLUMN (Total 3400) ---
y_eog_bottom = y_top_all - h_3400
draw_dataset_block(ax, x_eog, y_top_all, y_eog_bottom, 2.6, "#ECFDF5", "#059669", 
                   "EOG Artefak (Mata)", "Total: 3.400 Epochs (Indeks 0 s.d. 3399)")

rect_eog_used = patches.Rectangle((x_eog - 1.2, y_eog_bottom), 2.4, h_3400,
                                  facecolor="#D1FAE5", edgecolor="#10B981", linewidth=1.5, zorder=3)
ax.add_patch(rect_eog_used)
ax.text(x_eog, y_top_all - 0.4, "EOG[291] (Acak #1)", ha="center", va="center", fontsize=8.5, color="#065F46", zorder=4)
ax.text(x_eog, y_top_all - 0.8, "EOG[755] (Acak #2)", ha="center", va="center", fontsize=8.5, color="#065F46", zorder=4)
ax.text(x_eog, y_top_all - 1.2, "EOG[2096] (Acak #3)", ha="center", va="center", fontsize=8.5, color="#065F46", zorder=4)
ax.text(x_eog, y_top_all - 1.6, "...", ha="center", va="center", fontsize=10, fontweight="bold", color="#065F46", zorder=4)
ax.text(x_eog, y_eog_bottom + 0.4, "EOG[3174] (Acak #3400)", ha="center", va="center", fontsize=8.5, color="#065F46", zorder=4)

ax.text(x_eog, (y_top_all + y_eog_bottom)/2, "3.400 Epoch Terpilih\n(100% Seluruh EOG Terpakai\nShuffled tanpa duplikasi)", 
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="#047857",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0FDF4", edgecolor="#10B981", alpha=0.9), zorder=5)

ax.text(x_eog, y_eog_bottom - 0.35, "Sisa EOG: 0 (Habis Terpasang)", 
        ha="center", va="center", fontsize=8.5, fontweight="bold", color="#059669",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#ECFDF5", edgecolor="#34D399", linestyle="--"))

# --- INJECTION CONNECTORS (ARROWS & MATH) ---
ax.annotate("", xy=(x_eeg - 1.4, y_top_all - 0.4), xytext=(x_emg + 1.3, y_top_all - 0.4),
            arrowprops=dict(arrowstyle="<->", color="#B91C1C", lw=2, mutation_scale=15))
ax.annotate("", xy=(x_eeg - 1.4, y_emg_split + 0.4), xytext=(x_emg + 1.3, y_emg_split + 0.4),
            arrowprops=dict(arrowstyle="<->", color="#B91C1C", lw=2, mutation_scale=15))

rect_inject_emg = patches.FancyBboxPatch((4.05, 5.8), 2.3, 2.0,
                                         boxstyle="round,pad=0.1,rounding_size=0.15",
                                         facecolor="#FFF1F2", edgecolor="#E11D48", linewidth=1.5, zorder=6)
ax.add_patch(rect_inject_emg)
ax.text(5.2, 7.35, "INJEKSI EMG", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#BE123C")
ax.text(5.2, 6.85, r"$X_{\mathrm{contam}} = X_{\mathrm{EEG}} + \alpha \cdot X_{\mathrm{EMG}}$", 
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="#881337")
ax.text(5.2, 6.35, r"$\alpha = \sqrt{\frac{P_{\mathrm{EEG}}}{P_{\mathrm{EMG}} \cdot 10^{\mathrm{SNR}/10}}}$", 
        ha="center", va="center", fontsize=9, color="#9F1239")
ax.text(5.2, 5.95, "Hasil: 4.514 Pasang per SNR", ha="center", va="center", fontsize=8, fontweight="bold", color="#E11D48")

ax.annotate("", xy=(x_eog - 1.3, y_top_all - 0.4), xytext=(x_eeg + 1.4, y_top_all - 0.4),
            arrowprops=dict(arrowstyle="<->", color="#047857", lw=2, mutation_scale=15))
ax.annotate("", xy=(x_eog - 1.3, y_eog_bottom + 0.4), xytext=(x_eeg + 1.4, y_eeg_split + 0.4),
            arrowprops=dict(arrowstyle="<->", color="#047857", lw=2, mutation_scale=15))

rect_inject_eog = patches.FancyBboxPatch((9.65, 5.8), 2.3, 2.0,
                                         boxstyle="round,pad=0.1,rounding_size=0.15",
                                         facecolor="#F0FDF4", edgecolor="#059669", linewidth=1.5, zorder=6)
ax.add_patch(rect_inject_eog)
ax.text(10.8, 7.35, "INJEKSI EOG", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#065F46")
ax.text(10.8, 6.85, r"$X_{\mathrm{contam}} = X_{\mathrm{EEG}} + \alpha \cdot X_{\mathrm{EOG}}$", 
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="#064E3B")
ax.text(10.8, 6.35, r"$\alpha = \sqrt{\frac{P_{\mathrm{EEG}}}{P_{\mathrm{EOG}} \cdot 10^{\mathrm{SNR}/10}}}$", 
        ha="center", va="center", fontsize=9, color="#047857")
ax.text(10.8, 5.95, "Hasil: 3.400 Pasang per SNR", ha="center", va="center", fontsize=8, fontweight="bold", color="#059669")

# --- BOTTOM SUMMARY CARDS (OUTPUT DATASET) ---
card_emg = patches.FancyBboxPatch((0.8, 0.4), 6.5, 2.1, boxstyle="round,pad=0.1,rounding_size=0.15",
                                  facecolor="#FFFFFF", edgecolor="#DC2626", linewidth=1.5, zorder=5)
ax.add_patch(card_emg)
ax.text(4.05, 2.15, "Output Folder: processed/emg/", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#991B1B")
ax.text(4.05, 1.8, "• clean_eeg_all_epochs.npy  -> Shape: (4.514, 512)", ha="center", va="center", fontsize=9, color="#1E293B")
ax.text(4.05, 1.45, "• linear_synthetic.npy      -> Dict 10 SNR [-7 s.d. +2 dB]", ha="center", va="center", fontsize=9, color="#1E293B")
ax.text(4.05, 1.15, "  (Masing-masing SNR berisi array terkontaminasi shape 4.514 x 512)", ha="center", va="center", fontsize=8, color="#64748B")
ax.text(4.05, 0.7, "Status Sisa: 1.084 epoch EMG diabaikan agar bebas duplikasi & leakage", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#B91C1C")

card_eog = patches.FancyBboxPatch((8.7, 0.4), 6.5, 2.1, boxstyle="round,pad=0.1,rounding_size=0.15",
                                  facecolor="#FFFFFF", edgecolor="#059669", linewidth=1.5, zorder=5)
ax.add_patch(card_eog)
ax.text(11.95, 2.15, "Output Folder: processed/eog/", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#065F46")
ax.text(11.95, 1.8, "• clean_eeg_all_epochs.npy  -> Shape: (3.400, 512)", ha="center", va="center", fontsize=9, color="#1E293B")
ax.text(11.95, 1.45, "• linear_synthetic.npy      -> Dict 10 SNR [-7 s.d. +2 dB]", ha="center", va="center", fontsize=9, color="#1E293B")
ax.text(11.95, 1.15, "  (Masing-masing SNR berisi array terkontaminasi shape 3.400 x 512)", ha="center", va="center", fontsize=8, color="#64748B")
ax.text(11.95, 0.7, "Status Sisa: 1.114 epoch EEG diabaikan agar seimbang 1-ke-1 dengan EOG", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#047857")

plt.savefig("D:/Tsaqib/diagram_injeksi_dataset.png", bbox_inches="tight", dpi=200)
print("SUCCESS: Diagram saved to D:/Tsaqib/diagram_injeksi_dataset.png")
