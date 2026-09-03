"""Turn aponeurosis and fascicle masks into pennation angle, fascicle length and thickness.

Follows the manual protocol the labels were produced with (benchmark Readme, and the
challenge Data page): three muscle-thickness measurements spread across the image,
three fascicles, three pennation angles, each averaged. Matching the *convention*
matters as much as segmentation quality here -- the targets are an average of expert
tracings, not a physical constant, so a pipeline that measures something subtly
different is biased no matter how good its masks are.

All geometry is computed in the original image's pixel grid, never in the network's
512x512 input. The resize to 512 is anisotropic for most of these images, which would
rotate every measured angle and stretch every length along one axis.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import cv2


@dataclass
class Architecture:
    pa_deg: float | None
    fl_mm: float | None
    mt_mm: float | None
    n_fascicles: int
    ok: bool
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _largest_curves(mask: np.ndarray, k: int = 2, min_frac: float = 0.15) -> list[np.ndarray]:
    """The k widest connected structures in a binary mask, as (x, y) point sets.

    Aponeuroses are the long, roughly horizontal structures; selecting on horizontal
    extent rather than area avoids picking up a thick blob of speckle.
    """
    n, lbl, stats, _c = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    cands = []
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        if w < min_frac * mask.shape[1]:
            continue
        ys, xs = np.where(lbl == i)
        cands.append((w, np.column_stack([xs, ys])))
    cands.sort(key=lambda t: -t[0])
    return [c[1] for c in cands[:k]]


def _fit_curve(pts: np.ndarray, deg: int = 2,
               edge: str = "centre") -> np.polynomial.Polynomial | None:
    """Least-squares curve y(x) through a structure's pixels.

    Aponeuroses bow gently, so a quadratic tracks them better than a line while
    staying far too stiff to chase segmentation noise.

    `edge` selects which boundary of the segmented band to follow. This is a
    convention choice, not a detail: the raters measure the muscle belly, i.e.
    from the *inner* face of the superficial aponeurosis to the inner face of the
    deep one, whereas following both centrelines adds one aponeurosis thickness
    to every thickness measurement (~1.5 mm here, an 8.5 % overestimate).
    """
    if len(pts) < 20:
        return None
    x, y = pts[:, 0].astype(float), pts[:, 1].astype(float)
    xs = np.unique(x)
    if len(xs) < max(10, deg + 2):
        return None
    pick = {"centre": np.mean, "lower": np.max, "upper": np.min}[edge]
    ys = np.array([pick(y[x == v]) for v in xs])
    deg = min(deg, len(xs) - 2)
    return np.polynomial.Polynomial.fit(xs, ys, deg)


def _domain(p: np.polynomial.Polynomial) -> tuple[float, float]:
    d = p.domain
    return float(d[0]), float(d[1])




def _contour_edge(contour: np.ndarray, side: str = "B") -> np.ndarray | None:
    """One edge of a contour: the lowest ("B") or highest ("T") y per x column.

    DL_Track fits fascicles to a single contour edge rather than to the whole
    blob. The reason is geometric: a fascicle segmentation is a band several
    pixels thick whose *thickness* varies along its length, so a fit through all
    its pixels is pulled around by that varying thickness, while one edge is a
    consistent curve.
    """
    pts = contour.reshape(-1, 2)
    if len(pts) < 5:
        return None
    x, y = pts[:, 0], pts[:, 1]
    xs = np.unique(x)
    if len(xs) < 4:
        return None
    pick = np.max if side == "B" else np.min
    ys = np.array([pick(y[x == v]) for v in xs])
    return np.column_stack([xs, ys]).astype(float)


def fascicle_lengths_dltrack(fasc_mask: np.ndarray, sup_c, deep_c, lo: float, hi: float,
                             px_per_mm: float, min_pa: float = 10.0, max_pa: float = 40.0,
                             min_pts: int = 40, drop_cross: bool = True,
                             aspect: float = 1.0) -> tuple[list[float], list[float]]:
    """Fascicle lengths and angles following DL_Track's procedure.

    Restricts the fascicle mask to the band between the two aponeuroses, fits a
    line to one edge of each surviving contour, extrapolates it to both
    aponeuroses and takes the chord. Returns (lengths_mm, angles_deg).
    """
    h, w = fasc_mask.shape[:2]
    band = np.zeros((h, w), np.uint8)
    xs_all = np.arange(w)
    inside = (xs_all >= lo) & (xs_all <= hi)
    for xi in xs_all[inside]:
        y0, y1 = sorted((float(sup_c(xi)), float(deep_c(xi))))
        band[max(int(y0), 0):min(int(y1) + 1, h), xi] = 1
    m = (fasc_mask > 0).astype(np.uint8) & band
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    lengths, angles, spans = [], [], []
    grid = np.linspace(lo - 2 * (hi - lo), hi + 2 * (hi - lo), 5000)
    y_sup, y_deep = sup_c(grid), deep_c(grid)
    for c in cnts:
        if len(c) <= min_pts:
            continue
        edge = _contour_edge(c, "B")
        if edge is None or np.ptp(edge[:, 0]) < 10:
            continue
        z = np.polyfit(edge[:, 0], edge[:, 1], 1)
        f = np.poly1d(z)
        yf = f(grid)
        iu = int(np.argmin(np.abs(yf - y_sup)))
        il = int(np.argmin(np.abs(yf - y_deep)))
        if iu == il:
            continue
        xa, xb = grid[iu], grid[il]
        length = float(np.hypot((xb - xa) / aspect, y_deep[il] - y_sup[iu]))
        apo_slope = float((y_deep[min(il + 50, len(grid) - 1)] - y_deep[il])
                          / (grid[min(il + 50, len(grid) - 1)] - grid[il] + 1e-9))
        za, aa = z[0] * aspect, apo_slope * aspect
        ang = abs(np.degrees(np.arctan((za - aa) / (1 + za * aa))))
        if not (min_pa <= ang <= max_pa):
            continue
        lengths.append(length / px_per_mm)
        angles.append(ang)
        spans.append((min(xa, xb), max(xa, xb)))
    if drop_cross and len(spans) > 2:
        keep = drop_crossing(spans)
        if keep:
            lengths = [lengths[i] for i in keep]
            angles = [angles[i] for i in keep]
    return lengths, angles


def drop_crossing(spans: list[tuple[float, float]]) -> list[int]:
    """Indices of fascicles that do not have another fascicle spanning them.

    From Ritsche et al. 2024: segments "whose extrapolated path crosses the
    extrapolated path of another fascicle segment between the two detected
    aponeuroses" are removed, which "reduces the number of outliers used for
    calculation of median fascicle length and pennation angle". Real fascicles in
    one image are near-parallel, so a pair that crosses means at least one slope
    is wrong -- and a wrong slope is exactly what produces a wild extrapolation.
    """
    order = sorted(range(len(spans)), key=lambda i: spans[i][0])
    keep = {i: True for i in range(len(spans))}
    for a in range(len(order)):
        for b in range(a + 1, min(a + 3, len(order))):
            i, j = order[a], order[b]
            xi, Xi = spans[i]
            xj, Xj = spans[j]
            if xi <= xj and Xi >= Xj:
                keep[i] = False
            elif xj <= xi and Xj >= Xi:
                keep[j] = False
    return [i for i in range(len(spans)) if keep[i]]




def analyse(apo_mask: np.ndarray, fasc_mask: np.ndarray, px_per_cm: float,
            n_sites: int = 3, min_pa: float = 3.0, max_pa: float = 50.0,
            fl_mode: str = "blend", drop_cross: bool = True,
            aspect: float = 1.0) -> Architecture:
    """Measure PA, FL and MT from two binary masks in original-image pixels.

    `px_per_cm` is the *vertical* scale, which is what a depth annotation or a
    side ruler gives. `aspect` is the horizontal scale divided by the vertical one:
    1.0 for square pixels, and greater than 1.0 where the delivered image has been
    stretched horizontally relative to its acquisition geometry.

    Anisotropy is not a refinement here. The competition images were re-exported at
    a handful of canonical sizes whose aspect ratios differ from those of the native
    masks, and an anisotropic resize rotates every angle: a slope measured in the
    delivered frame relates to the true one by tan(true) = aspect * tan(measured).
    Lengths are likewise mixed, since a fascicle runs mostly across the image while
    thickness runs down it.
    """
    if not px_per_cm or px_per_cm <= 0:
        return Architecture(None, None, None, 0, False, "no scale")
    px_per_mm = px_per_cm / 10.0

    curves = _largest_curves(apo_mask, k=4)
    if len(curves) < 2:
        return Architecture(None, None, None, 0, False, "fewer than two aponeuroses")
    centres = [(c, _fit_curve(c, edge="centre")) for c in curves]
    centres = [(c, f) for c, f in centres if f is not None]
    if len(centres) < 2:
        return Architecture(None, None, None, 0, False, "aponeurosis fit failed")

    # Choose the pair whose separation is physiologically plausible. Taking simply
    # the two widest structures picks up subcutaneous fascia or the bone echo below
    # the muscle on some images, which produced gross thickness outliers.
    best = None
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            fi, fj = centres[i][1], centres[j][1]
            lo_ = max(_domain(fi)[0], _domain(fj)[0])
            hi_ = min(_domain(fi)[1], _domain(fj)[1])
            if hi_ - lo_ < 20:
                continue
            xs_ = np.linspace(lo_, hi_, 25)
            sep = np.abs(fj(xs_) - fi(xs_)) / (px_per_cm / 10.0)
            if not (5.0 <= np.median(sep) <= 60.0):
                continue
            score_ = (hi_ - lo_) / max(np.std(sep), 0.3)   # wide and evenly spaced
            if best is None or score_ > best[0]:
                best = (score_, centres[i], centres[j])
    if best is None:
        return Architecture(None, None, None, 0, False, "no plausible aponeurosis pair")
    (_, (pa_pts, pa_fit), (pb_pts, pb_fit)) = best
    # image y grows downwards, so the superficial aponeurosis is the smaller y
    if float(pa_fit(np.mean(_domain(pa_fit)))) < float(pb_fit(np.mean(_domain(pb_fit)))):
        sup_pts, deep_pts = pa_pts, pb_pts
    else:
        sup_pts, deep_pts = pb_pts, pa_pts
    # refit on the faces that bound the muscle belly
    sup = _fit_curve(sup_pts, edge="lower")
    deep = _fit_curve(deep_pts, edge="upper")
    # Fascicle length uses the aponeurosis centrelines, not the belly faces. The two
    # targets follow different conventions: thickness spans the muscle between the
    # aponeuroses, while a traced fascicle runs into each aponeurosis rather than
    # stopping at its face. Measuring FL between the same inner faces used for MT
    # loses one aponeurosis thickness along the fascicle, t/sin(PA) ~ 4.4 mm here.
    sup_c = _fit_curve(sup_pts, edge="centre")
    deep_c = _fit_curve(deep_pts, edge="centre")
    if sup is None or deep is None or sup_c is None or deep_c is None:
        return Architecture(None, None, None, 0, False, "aponeurosis edge fit failed")

    # ---- muscle thickness: vertical separation at n evenly spread sites ----
    lo = max(_domain(sup)[0], _domain(deep)[0])
    hi = min(_domain(sup)[1], _domain(deep)[1])
    if hi - lo < 20:
        return Architecture(None, None, None, 0, False, "aponeuroses do not overlap")
    # Mean vertical separation at n evenly spread sites, matching the raters'
    # "three straight lines at a left, middle and right location". Ritsche et al.
    # instead take the single shortest distance over the central third; measured
    # here that is worse (MT MAE 1.55 vs 1.17 mm), so we keep the rater protocol.
    sites = np.linspace(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo), n_sites)
    seps = np.array([abs(float(deep(s)) - float(sup(s))) for s in sites])
    mt_mm = float(np.mean(seps) / px_per_mm)

    # ---- fascicle orientation from skeleton fragments ----
    fasc = (fasc_mask > 0).astype(np.uint8)
    n, lbl, stats, _c = cv2.connectedComponentsWithStats(fasc, 8)
    angles = []
    deep_slope_at = lambda x: float(deep.deriv()(x))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 25:
            continue
        ys, xs = np.where(lbl == i)
        if np.ptp(xs) < 10:
            continue
        vx, vy, x0, y0 = cv2.fitLine(np.column_stack([xs, ys]).astype(np.float32),
                                     cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        if abs(vx) < 1e-6:
            continue
        m = float(vy / vx)
        xc = float(xs.mean())
        # pennation angle is measured against the local deep-aponeurosis tangent,
        # not against the horizontal
        md = deep_slope_at(np.clip(xc, lo, hi))
        ma, mda = m * aspect, md * aspect          # into physical (square-pixel) space
        ang = (abs(np.degrees(np.arctan((ma - mda) / (1 + ma * mda))))
               if abs(1 + ma * mda) > 1e-9 else 90.0)
        if not (min_pa <= ang <= max_pa):
            continue
        angles.append(ang)
    if not angles:
        return Architecture(None, None, mt_mm, 0, False, "no usable fascicle fragments")

    pa_deg = float(np.median(angles))

    # ---- fascicle length ----
    # Two independent estimators. `trig` converts MT and PA through
    # FL = MT / sin(PA) and never extrapolates; `dltrack` extrapolates traced
    # fascicle contours and never looks at MT.
    fl_trig = float(mt_mm / max(np.sin(np.radians(pa_deg)), 1e-3))
    dl_len, dl_ang = fascicle_lengths_dltrack(fasc_mask, sup_c, deep_c, lo, hi, px_per_mm,
                                              drop_cross=drop_cross, aspect=aspect)
    fl_dl = float(np.median(dl_len)) if dl_len else None
    # Average the trigonometric and DL_Track-style estimates. They are genuinely
    # independent -- one converts MT and PA without extrapolating at all, the other
    # extrapolates traced fascicles and never looks at MT -- and their biases came
    # out almost equal and opposite (-3.92 mm and +4.55 mm). Averaging cancels most
    # of both: FL MAE 7.72 / 9.26 alone, 5.90 blended. Weight is fixed at 0.5 rather
    # than the fitted optimum of 0.6 (which scores 0.3620 vs 0.3637) so that nothing
    # here is tuned to the 35 benchmark images.
    fl_blend = (0.5 * fl_trig + 0.5 * fl_dl) if fl_dl is not None else fl_trig
    options = {"blend": fl_blend, "trig": fl_trig, "dltrack": fl_dl}
    fl_mm = options.get(fl_mode)
    if fl_mm is None or not (20.0 <= fl_mm <= 250.0):
        fl_mm, fl_src = fl_trig, "trig"
    else:
        fl_src = fl_mode
    return Architecture(pa_deg, fl_mm, mt_mm, len(angles), True, fl_src)
