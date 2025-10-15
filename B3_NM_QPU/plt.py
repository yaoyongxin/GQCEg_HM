import numpy, itertools
import matplotlib.pyplot as plt


fnames = [
        "L01/results_L01_em1",
        "L10/results_L10_em1",
        "L10/results_L10_em2",
        "L10/results_L10_em3",
        "L30/results_L30_em1",
        "L30/results_L30_em2",
        "L30/results_L30_em3",
        "L30/results_L30_h1_1le",
        "L30/results_L30_h1_1e",
        "L30/results_L30_h1_1",
        ]

errors = []
stds = []
for fname in fnames:
    data = numpy.loadtxt(fname)
    aves = data[:, 2:].mean(axis=1)
    stds.append(data[:, 2:].std(axis=1).max())
    errors.append(abs(aves - data[:, 1]))

print(numpy.asarray(stds))
print(numpy.asarray(errors).max(axis=1))


fig, ax = plt.subplots(figsize=(6, 4))
positions = numpy.arange(len(errors)) + 1

ax.boxplot(errors, positions=positions, widths=0.6, patch_artist=True,
        boxprops=dict(facecolor=f'C{0}', color=f'C{0}'),
        medianprops=dict(color='black'), whis=(0, 100))

xlabels = [
        "C1M1",
        "C2M1",
        "C2M2",
        "C2M3",
        "C3M1",
        "C3M2",
        "C3M3",
        "H1-1LE",
        "H1-1E",
        "H1-1",
        ]
ax.set_xticklabels(xlabels, rotation=0)

for x in [1.5, 4.5, 7.5]:
    ax.axvline(x=x, ls="--", color="gray")

ax.text(0.01, 0.85, 'ECR: 4', transform=ax.transAxes)
ax.text(0.2, 0.85, 'ECR: [26, 38]', transform=ax.transAxes)
ax.text(0.55, 0.85, 'ECR: [131, 142]', transform=ax.transAxes)

ax.set_xlabel('Circuit and Mitigation level')
ax.set_ylabel('Error of density matrix')

plt.tight_layout()

# Save the figure as a PDF
plt.savefig("dm_analysis.pdf")

plt.show()
