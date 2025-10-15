import numpy, itertools
import matplotlib.pyplot as plt


fnames = [
        "L30/results_L30_h1_1le",
        "L30/results_L30_h1_emulator",
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
        "H1-1LE",
        "H1-Emulator",
        "H1-1E",
        "H1-1",
        ]
ax.set_xticklabels(xlabels, rotation=0)

# ax.set_xlabel('Circuit and Mitigation level')
ax.set_ylabel('Error of density matrix')

plt.tight_layout()

# Save the figure as a PDF
plt.savefig("dm_analysis_nexus.pdf")

plt.show()
