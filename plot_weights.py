import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality style (IEEE standard style)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

def plot_uncertainty_weight_shifts():
    # 1. Noise levels (e.g., Gaussian Blur Kernel / Sigma levels applied to Palmprint)
    noise_levels = ['Clean (0.0)', 'Low (1.5)', 'Medium (3.0)', 'Severe (5.0)', 'Extreme (7.0)']
    
    # 2. Simulated / Evaluated weight outputs from model (alpha_p + alpha_v = 1.0)
    # As palmprint degrades, alpha_p drops while alpha_v rises dynamically
    alpha_p = [0.52, 0.43, 0.28, 0.15, 0.08]  # Palmprint Stream Weight
    alpha_v = [0.48, 0.57, 0.72, 0.85, 0.92]  # Palm Vein Stream Weight

    x = np.arange(len(noise_levels))  # Label locations
    width = 0.35  # Bar width

    # 3. Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # Plot bars
    rects1 = ax.bar(x - width/2, alpha_p, width, label=r'Palmprint Weight ($\alpha_p$)', 
                   color='#2b5c8f', edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x + width/2, alpha_v, width, label=r'Palm Vein Weight ($\alpha_v$)', 
                   color='#d95f02', edgecolor='black', linewidth=0.8)

    # 4. Styling & Labels
    ax.set_ylabel('Dynamic Gating Weight Value', fontsize=12, fontweight='bold')
    ax.set_xlabel('Palmprint Modality Degradation (Gaussian Blur $\sigma$)', fontsize=12, fontweight='bold')
    ax.set_title('UAD-Net: Dynamic Weight Adaptation Under Increasing Noise', fontsize=13, pad=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(noise_levels)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # 5. Add exact numerical value annotations on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3pt vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    # Add legend
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right', fontsize=10)

    # 6. Adjust layout and save high-res image for paper
    plt.tight_layout()
    plt.savefig('uadnet_weight_shifts.png', dpi=300)
    plt.savefig('uadnet_weight_shifts.pdf')  # PDF format preferred by IEEE LaTeX
    print("[✓] Plot successfully generated and saved as 'uadnet_weight_shifts.png' and '.pdf'!")
    plt.show()

if __name__ == "__main__":
    plot_uncertainty_weight_shifts()