//! Exact 1-D Akima cubic Hermite interpolator matching `scipy.interpolate.Akima1DInterpolator`.

/// Piecewise cubic Hermite polynomial interpolator using Akima's local slope weighting.
#[derive(Debug, Clone)]
pub struct Akima1DInterpolator {
    x: Vec<f64>,
    /// Polynomial coefficients `[c3, c2, c1, c0]` per interval `[x[i], x[i+1]]`
    /// evaluated as `((c3 * dx + c2) * dx + c1) * dx + c0` with `dx = xi - x[i]`.
    coeffs: Vec<[f64; 4]>,
}

impl Akima1DInterpolator {
    /// Fits an Akima 1-D interpolator on strictly increasing `x` and matching `y`.
    ///
    /// # Panics
    /// Panics if `x.len() != y.len()`, `x.len() < 2`, or any interval `x[i+1] - x[i] <= 0.0`.
    pub fn new(x: &[f64], y: &[f64]) -> Self {
        assert_eq!(
            x.len(),
            y.len(),
            "Akima1DInterpolator: x and y length mismatch"
        );
        let n = x.len();
        assert!(n >= 2, "Akima1DInterpolator requires at least 2 points");

        let mut dx = Vec::with_capacity(n - 1);
        for i in 0..(n - 1) {
            let h = x[i + 1] - x[i];
            assert!(
                h > 0.0,
                "Akima1DInterpolator: x must be strictly increasing"
            );
            dx.push(h);
        }

        let t: Vec<f64> = if n == 2 {
            let slope = (y[1] - y[0]) / dx[0];
            vec![slope, slope]
        } else {
            // Slopes m has length n + 3: indices 2..n+1 hold interval slopes,
            // indices 0,1 and n+1,n+2 hold extrapolated boundary slopes.
            let mut m = vec![0.0_f64; n + 3];
            for i in 0..(n - 1) {
                m[i + 2] = (y[i + 1] - y[i]) / dx[i];
            }
            m[1] = 2.0 * m[2] - m[3];
            m[0] = 2.0 * m[1] - m[2];
            m[n + 1] = 2.0 * m[n] - m[n - 1];
            m[n + 2] = 2.0 * m[n + 1] - m[n];

            let mut dm = Vec::with_capacity(n + 2);
            for i in 0..(n + 2) {
                dm.push((m[i + 1] - m[i]).abs());
            }

            let mut f1 = Vec::with_capacity(n);
            let mut f2 = Vec::with_capacity(n);
            let mut f12 = Vec::with_capacity(n);
            let mut mmax = f64::NEG_INFINITY;
            for i in 0..n {
                let a = dm[i + 2];
                let b = dm[i];
                let s = a + b;
                if s > mmax {
                    mmax = s;
                }
                f1.push(a);
                f2.push(b);
                f12.push(s);
            }

            let threshold = 1e-9 * mmax;
            let mut slopes = Vec::with_capacity(n);
            for i in 0..n {
                if f12[i] > threshold {
                    slopes.push(m[i + 1] + (f2[i] / f12[i]) * (m[i + 2] - m[i + 1]));
                } else {
                    slopes.push(0.5 * (m[i + 3] + m[i]));
                }
            }
            slopes
        };

        // Construct cubic Hermite coefficients matching CubicHermiteSpline
        let mut coeffs = Vec::with_capacity(n - 1);
        for i in 0..(n - 1) {
            let h = dx[i];
            let slope = (y[i + 1] - y[i]) / h;
            let t0 = t[i];
            let t1 = t[i + 1];
            let c0 = y[i];
            let c1 = t0;
            let c2 = (3.0 * slope - 2.0 * t0 - t1) / h;
            let c3 = (t0 + t1 - 2.0 * slope) / (h * h);
            coeffs.push([c3, c2, c1, c0]);
        }

        Self {
            x: x.to_vec(),
            coeffs,
        }
    }

    /// Evaluates the Akima spline at `xi`. Values outside `[x[0], x[n-1]]` are clamped
    /// if the caller clamps `xi`, or return `f64::NAN` matching SciPy `extrapolate=False`.
    pub fn eval(&self, xi: f64) -> f64 {
        let n = self.x.len();
        if xi.is_nan() || xi < self.x[0] || xi > self.x[n - 1] {
            return f64::NAN;
        }
        let idx = if xi == self.x[n - 1] {
            n - 2
        } else {
            // Find largest i such that x[i] <= xi
            let pos = self.x.partition_point(|&xk| xk <= xi);
            pos.saturating_sub(1).min(n - 2)
        };
        let d = xi - self.x[idx];
        let [c3, c2, c1, c0] = self.coeffs[idx];
        ((c3 * d + c2) * d + c1) * d + c0
    }

    /// Evaluates the Akima spline over a slice, clamping query points to `[x[0], x[last]]`.
    pub fn eval_clamped(&self, xs: &[f64]) -> Vec<f64> {
        let lo = self.x[0];
        let hi = self.x[self.x.len() - 1];
        xs.iter().map(|&xi| self.eval(xi.clamp(lo, hi))).collect()
    }
}
