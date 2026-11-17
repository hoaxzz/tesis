import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Setup figure
fig, ax = plt.subplots(figsize=(15, 7.5), dpi=300)
ax.set_xlim(0, 15)
ax.set_ylim(0, 8.5)
ax.axis("off")

# White background like Draw.io
fig.patch.set_facecolor("#FFFFFF")
ax.set_facecolor("#FFFFFF")

# Box coordinates
# 3 Columns: EMG (left), EEG (center), EOG (right)
box_w = 4.0
box_h = 7.4
y_bottom = 0.55

x_emg = 0.6
x_eeg = 5.5
x_eog = 10.4

# Draw 3 main rectangle boxes
rect_emg = patches.Rectangle((x_emg, y_bottom), box_w, box_h, fill=False, edgecolor="black", linewidth=1.4)
rect_eeg = patches.Rectangle((x_eeg, y_bottom), box_w, box_h, fill=False, edgecolor="black", linewidth=1.4)
rect_eog = patches.Rectangle((x_eog, y_bottom), box_w, box_h, fill=False, edgecolor="black", linewidth=1.4)

ax.add_patch(rect_emg)
ax.add_patch(rect_eeg)
ax.add_patch(rect_eog)

# Header Titles
ax.text(x_emg + box_w/2, y_bottom + box_h - 0.45, "5598 Sampel Sinyal Artefak EMG", 
        ha="center", va="center", fontsize=11.5, fontweight="bold", color="black")
ax.text(x_eeg + box_w/2, y_bottom + box_h - 0.45, "4514 Sampel Sinyal EEG Bersih", 
        ha="center", va="center", fontsize=11.5, fontweight="bold", color="black")
ax.text(x_eog + box_w/2, y_bottom + box_h - 0.45, "3400 Sampel Sinyal Artefak EOG", 
        ha="center", va="center", fontsize=11.5, fontweight="bold", color="black")

# Text items with respective y coordinates
# Top items: 1, 2, 3
y_top = [6.8, 6.2, 5.6]
emg_top = [
    "Sinyal EMG 401 #Acak 1",
    "Sinyal EMG 3674 #Acak 2",
    "Sinyal EMG ke 296 #Acak 3"
]
eeg_top = [
    "Sinyal EEG Bersih 1",
    "Sinyal EEG Bersih 2",
    "Sinyal EEG Bersih 3"
]
eog_top = [
    "Sinyal Artefak EOG 291 #acak 1",
    "Sinyal Artefak EOG 455 #acak 2",
    "Sinyal Artefak EOG 2247 #acak 3"
]

for y, emg_t, eeg_t, eog_t in zip(y_top, emg_top, eeg_top, eog_top):
    ax.text(x_emg + box_w/2, y, emg_t, ha="center", va="center", fontsize=10)
    ax.text(x_eeg + box_w/2, y, eeg_t, ha="center", va="center", fontsize=10)
    ax.text(x_eog + box_w/2, y, eog_t, ha="center", va="center", fontsize=10)
    
    # Double-headed arrow EMG <-> EEG
    ax.annotate("", xy=(x_eeg, y), xytext=(x_emg + box_w, y),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.1, mutation_scale=12))
    # Double-headed arrow EEG <-> EOG
    ax.annotate("", xy=(x_eog, y), xytext=(x_eeg + box_w, y),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.1, mutation_scale=12))

# Dots 1
y_dot1 = 4.95
ax.text(x_emg + box_w/2, 4.0, "...................", ha="center", va="center", fontsize=10)
ax.text(x_eeg + box_w/2, y_dot1, "...................", ha="center", va="center", fontsize=10)
ax.text(x_eog + box_w/2, 4.95, "...................", ha="center", va="center", fontsize=10)

# Mid items (3398, 3399, 3400)
y_mid = [4.3, 3.7, 3.1]
eeg_mid = [
    "Sinyal EEG Bersih 3398",
    "Sinyal EEG Bersih 3399",
    "Sinyal EEG Bersih 3400"
]
eog_mid = [
    "Sinyal Artefak EOG 3211 #acak 3398",
    "Sinyal Artefak EOG 462 #acak 3399",
    "Sinyal Artefak EOG 1421 #acak 3400"
]

for y, eeg_t, eog_t in zip(y_mid, eeg_mid, eog_mid):
    ax.text(x_eeg + box_w/2, y, eeg_t, ha="center", va="center", fontsize=10)
    ax.text(x_eog + box_w/2, y, eog_t, ha="center", va="center", fontsize=10)
    
    # Double-headed arrow EEG <-> EOG
    ax.annotate("", xy=(x_eog, y), xytext=(x_eeg + box_w, y),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.1, mutation_scale=12))

# Dots 2 (between 3400 and 4512 in EEG Clean)
y_dot2 = 2.45
ax.text(x_eeg + box_w/2, y_dot2, "...................", ha="center", va="center", fontsize=10)

# Bottom items (4512, 4513, 4514)
y_bot = [1.8, 1.25, 0.7]
emg_bot = [
    "Sinyal EMG ke 3412 #Acak 4512",
    "Sinyal EMG ke 5021 #Acak 4513",
    "Sinyal EMG ke 412 #Acak 4514"
]
eeg_bot = [
    "Sinyal EEG Bersih 4512",
    "Sinyal EEG Bersih 4513",
    "Sinyal EEG Bersih 4514"
]

for y, emg_t, eeg_t in zip(y_bot, emg_bot, eeg_bot):
    ax.text(x_emg + box_w/2, y, emg_t, ha="center", va="center", fontsize=10)
    ax.text(x_eeg + box_w/2, y, eeg_t, ha="center", va="center", fontsize=10)
    
    # Single-headed arrow EMG -> EEG (matching user's draw.io image)
    ax.annotate("", xy=(x_eeg, y), xytext=(x_emg + box_w, y),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.1, mutation_scale=12))

# Save image
out_path = r"D:\Tsaqib\diagram_pairing_drawio.png"
plt.savefig(out_path, bbox_inches="tight", dpi=300)
print("SUCCESS: Diagram saved to", out_path)
