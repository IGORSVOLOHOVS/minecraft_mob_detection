import os

# The runner has no display. matplotlib reads this once, on its first import, and
# conftest.py comes before every test module, so `plt.show()` never opens a window.
os.environ.setdefault("MPLBACKEND", "Agg")
