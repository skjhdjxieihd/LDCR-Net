# Third-party notices

The root MIT license covers the project contributions and modifications.
The attribution and permission notice below also apply to the indicated
upstream-derived portions. Preserve this file and the referenced license
when redistributing the review code. Third-party attribution identifies
upstream authors, not the authors of the submitted manuscript.

## CPC baseline

File: `src/CPC.py`.

The next-frame classifier's contrastive sampling/scoring pattern is adapted
from the MIT-licensed UnsupSeg implementation:

- Project: UnsupSeg, *Self-Supervised Contrastive Learning for Unsupervised
  Phoneme Segmentation*.
- Upstream: https://github.com/felixkreuk/UnsupSeg
- Relevant file: `next_frame_classifier.py`.
- Copyright: Copyright (c) 2020 Felix Kreuk.
- Full permission notice: [`LICENSES/UnsupSeg-MIT.txt`](LICENSES/UnsupSeg-MIT.txt).

The local adaptation accepts four-channel descriptor sequences, uses
median-pooling downsampling, supports local encoder settings, exposes
features for boundary scoring, and uses fixed-length training crops.

## Separately installed dependencies

NumPy, SciPy, scikit-learn, Matplotlib, PyTorch, threadpoolctl and ruptures
are installed through `requirements.txt`. Their source/distribution packages
are not bundled here and retain their own licenses and copyright notices.
The project license does not replace those dependency licenses.
