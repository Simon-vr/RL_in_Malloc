"""Legacy ``test.py`` demo.

Previously this module imported matplotlib and called ``plt.show()`` at import
time.  The logic now lives in :func:`rlmalloc.plotting.plot_histogram` and is
only executed under ``if __name__ == "__main__"``.
"""

from rlmalloc.plotting import plot_histogram  # noqa: F401


if __name__ == "__main__":
    import numpy as np

    from rlmalloc.workloads import sample_request_by_name

    rng = np.random.default_rng(0)
    samples = np.array(
        [sample_request_by_name("lognormal_train", rng) for _ in range(10000)]
    )
    plot_histogram(
        samples,
        "results/figures/fig3_train_hist.png",
        "results/figures/fig3_train_hist.svg",
        title="Lognormal(ln32, 0.9), clipped [1,512]",
    )
    print("wrote results/figures/fig3_train_hist.png")
