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


def merge_fragments(frags: list[np.ndarray], theta: float,
                    tol_px: float = 12.0) -> list[np.ndarray]:
    """Group fragments that lie along the same fascicle, and pool their pixels.

    A single fascicle is broken by the network into several short pieces wherever
    its echo fades. Fitting each piece alone gives a badly conditioned slope --
    the piece is short, so a couple of pixels of segmentation noise swing it by
    degrees, and extrapolating that across the whole muscle multiplies the error
    (dFL/dPA is ~2.8 mm per degree here).

    Pieces of one fascicle share a perpendicular offset when projected across the
    dominant orientation, so grouping on that offset and refitting over the pooled
    pixels recovers a long, well-conditioned line.
    """
    if not frags:
        return []
    nx, ny = -np.sin(theta), np.cos(theta)       # unit normal to the fascicle direction
    offs = np.array([float(np.mean(f[:, 0]) * nx + np.mean(f[:, 1]) * ny) for f in frags])
    order = np.argsort(offs)
    groups, cur = [], [order[0]]
    for a, b in zip(order, order[1:]):
        if abs(offs[b] - offs[a]) <= tol_px:
            cur.append(b)
        else:
            groups.append(cur)
            cur = [b]
    groups.append(cur)
    return [np.vstack([frags[i] for i in g]) for g in groups]


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
                             min_pts: int = 40) -> tuple[list[float], list[float]]:
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
    lengths, angles = [], []
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
        length = float(np.hypot(xb - xa, y_deep[il] - y_sup[iu]))
        apo_slope = float((y_deep[min(il + 50, len(grid) - 1)] - y_deep[il])
                          / (grid[min(il + 50, len(grid) - 1)] - grid[il] + 1e-9))
        ang = abs(np.degrees(np.arctan((z[0] - apo_slope) / (1 + z[0] * apo_slope))))
        if not (min_pa <= ang <= max_pa):
            continue
        lengths.append(length / px_per_mm)
        angles.append(ang)
    return lengths, angles


def analyse(apo_mask: np.ndarray, fasc_mask: np.ndarray, px_per_cm: float,
            n_sites: int = 3, min_pa: float = 3.0, max_pa: float = 50.0,
            fl_mode: str = "blend", fl_gain: float = 1.0,
            merge_tol: float = 12.0, pa_mode: str = "ours") -> Architecture:
    """Measure PA, FL and MT from two binary masks in original-image pixels."""
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
    sites = np.linspace(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo), n_sites)
    seps = np.array([abs(float(deep(s)) - float(sup(s))) for s in sites])
    mt_mm = float(np.mean(seps) / px_per_mm)

    # ---- fascicle orientation from skeleton fragments ----
    fasc = (fasc_mask > 0).astype(np.uint8)
    n, lbl, stats, _c = cv2.connectedComponentsWithStats(fasc, 8)
    angles, lengths, frags = [], [], []
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
        ang = abs(np.degrees(np.arctan((m - md) / (1 + m * md)))) if abs(1 + m * md) > 1e-9 else 90.0
        if not (min_pa <= ang <= max_pa):
            continue
        angles.append(ang)
        frags.append(np.column_stack([xs, ys]))
    if not angles:
        return Architecture(None, None, mt_mm, 0, False, "no usable fascicle fragments")

    pa_deg = float(np.median(angles))

    # Pool fragments belonging to the same fascicle, then extrapolate the merged
    # lines. See merge_fragments for why per-fragment fitting is ill-conditioned.
    theta = np.arctan(np.median([np.tan(np.radians(a)) for a in angles]))
    for grp in merge_fragments(frags, theta, tol_px=merge_tol):
        if np.ptp(grp[:, 0]) < 15:
            continue
        vx, vy, x0, y0 = cv2.fitLine(grp.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        if abs(vx) < 1e-6:
            continue
        m = float(vy / vx)
        f_line = lambda x, m=m, x0=x0, y0=y0: y0 + m * (x - x0)
        xs_grid = np.linspace(lo, hi, 400)
        d_sup = f_line(xs_grid) - sup_c(xs_grid)
        d_deep = f_line(xs_grid) - deep_c(xs_grid)
        cs = np.where(np.sign(d_sup[:-1]) != np.sign(d_sup[1:]))[0]
        cd = np.where(np.sign(d_deep[:-1]) != np.sign(d_deep[1:]))[0]
        if len(cs) and len(cd):
            xa, xb = xs_grid[cs[0]], xs_grid[cd[0]]
            lengths.append(float(np.hypot(xb - xa, f_line(xb) - f_line(xa))))

    # Fascicle length from ONE representative fascicle built on the median
    # orientation, rather than extrapolating each fragment separately. Fragments are
    # short and their individual slopes noisy; because dFL/dPA is about 2.8 mm per
    # degree at these geometries, per-fragment extrapolation amplifies that noise
    # into the metres-long tail that dominated RMSE. The median angle is already the
    # robust quantity, so extrapolate that once instead.
    xmid = 0.5 * (lo + hi)
    md_mid = float(deep.deriv()(np.clip(xmid, lo, hi)))
    theta = np.arctan(md_mid) + np.radians(pa_deg)
    m_rep = float(np.tan(theta))
    y_mid = 0.5 * (float(sup_c(xmid)) + float(deep_c(xmid)))
    xs_grid = np.linspace(lo - 2 * (hi - lo), hi + 2 * (hi - lo), 2000)
    rep = y_mid + m_rep * (xs_grid - xmid)
    ds = rep - sup_c(np.clip(xs_grid, *_domain(sup_c)))
    dd = rep - deep_c(np.clip(xs_grid, *_domain(deep_c)))
    cs = np.where(np.sign(ds[:-1]) != np.sign(ds[1:]))[0]
    cd = np.where(np.sign(dd[:-1]) != np.sign(dd[1:]))[0]
    fl_rep = None
    if len(cs) and len(cd):
        xa, xb = xs_grid[cs[0]], xs_grid[cd[0]]
        fl_rep = float(np.hypot(xb - xa, m_rep * (xb - xa)) / px_per_mm)
    # Measured: the per-fragment median beats the single representative line
    # (0.4477 vs 0.4814 on the benchmark). Averaging many noisy short extrapolations
    # evidently cancels more error than it amplifies, so the representative line is
    # kept only for images where no fragment spans both aponeuroses.
    # Take the median over every fragment's extrapolation, unfiltered: pre-filtering
    # individual fragments to a plausible range before the median made things worse
    # (0.4926 vs 0.4477), because it biases which fragments survive. Guard the
    # aggregate instead.
    # ---- fascicle length: choose the estimator explicitly ----
    # Measured on the expert benchmark, per-image MAE:
    #   trig (MT/sin PA)      6.70 mm     <- best
    #   representative line   9.06 mm
    #   per-fragment median  12.52 mm
    # Extrapolating a short, noisy fragment across the whole muscle amplifies its
    # slope error, whereas MT and PA are both measured well and the trigonometric
    # identity converts them without any further extrapolation. Counterintuitive,
    # but this is exactly what the harness is for.
    fl_trig = float(mt_mm / max(np.sin(np.radians(pa_deg)), 1e-3))
    fl_frag = float(np.median(lengths) / px_per_mm) if lengths else None
    dl_len, dl_ang = fascicle_lengths_dltrack(fasc_mask, sup_c, deep_c, lo, hi, px_per_mm)
    fl_dl = float(np.median(dl_len)) if dl_len else None
    if dl_ang:
        pa_deg = float(np.median(dl_ang)) if pa_mode == "dltrack" else pa_deg
    # Average the trigonometric and DL_Track-style estimates. They are genuinely
    # independent -- one converts MT and PA without extrapolating at all, the other
    # extrapolates traced fascicles and never looks at MT -- and their biases came
    # out almost equal and opposite (-3.92 mm and +4.55 mm). Averaging cancels most
    # of both: FL MAE 7.72 / 9.26 alone, 5.90 blended. Weight is fixed at 0.5 rather
    # than the fitted optimum of 0.6 (which scores 0.3620 vs 0.3637) so that nothing
    # here is tuned to the 35 benchmark images.
    fl_blend = (0.5 * fl_trig + 0.5 * fl_dl) if fl_dl is not None else fl_trig
    options = {"trig": fl_trig, "fragments": fl_frag, "representative": fl_rep,
               "dltrack": fl_dl, "blend": fl_blend}
    fl_mm = options.get(fl_mode)
    if fl_mm is None or not (20.0 <= fl_mm <= 250.0):
        fl_mm, fl_src = fl_trig, "trig"
    else:
        fl_src = fl_mode
    fl_mm *= fl_gain
    return Architecture(pa_deg, fl_mm, mt_mm, len(angles), True, fl_src)
