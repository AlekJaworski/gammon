//! Cheap phase-level wall-clock timers for diagnosing per-iter cost
//! breakdown. Inactive by default — enable with the `profile` feature.
//!
//! Usage:
//! ```no_run
//! # use gamrs::profile;
//! let _guard = profile::scoped("fit_inner_at");
//! // ... work ...
//! drop(_guard);  // accumulates elapsed ns into the named bucket
//! ```
//!
//! At end of bench:
//! ```no_run
//! # use gamrs::profile;
//! profile::dump(&mut std::io::stderr()).unwrap();
//! ```

#[cfg(feature = "profile")]
pub use enabled::*;

#[cfg(not(feature = "profile"))]
pub use disabled::*;

#[cfg(feature = "profile")]
mod enabled {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;
    use std::time::Instant;

    /// Phase buckets. Update when adding new instrumented phases.
    static BUCKETS: Mutex<Vec<(&'static str, AtomicU64, AtomicU64)>> = Mutex::new(Vec::new());

    pub struct ScopedTimer {
        name: &'static str,
        start: Instant,
    }

    impl Drop for ScopedTimer {
        fn drop(&mut self) {
            let ns = self.start.elapsed().as_nanos() as u64;
            let mut bs = BUCKETS.lock().unwrap();
            for (n, count, ns_total) in bs.iter() {
                if *n == self.name {
                    count.fetch_add(1, Ordering::Relaxed);
                    ns_total.fetch_add(ns, Ordering::Relaxed);
                    return;
                }
            }
            bs.push((self.name, AtomicU64::new(1), AtomicU64::new(ns)));
        }
    }

    pub fn scoped(name: &'static str) -> ScopedTimer {
        ScopedTimer {
            name,
            start: Instant::now(),
        }
    }

    pub fn dump<W: std::io::Write>(w: &mut W) -> std::io::Result<()> {
        let bs = BUCKETS.lock().unwrap();
        let mut sorted: Vec<_> = bs
            .iter()
            .map(|(n, c, ns)| {
                (
                    *n,
                    c.load(Ordering::Relaxed),
                    ns.load(Ordering::Relaxed),
                )
            })
            .collect();
        sorted.sort_by_key(|(_, _, ns)| std::cmp::Reverse(*ns));
        writeln!(w, "{:<40}  {:>10}  {:>14}  {:>10}", "phase", "count", "total_ms", "avg_us")?;
        for (name, count, ns) in sorted {
            let ms = ns as f64 / 1.0e6;
            let avg_us = if count == 0 { 0.0 } else { ns as f64 / count as f64 / 1000.0 };
            writeln!(w, "{:<40}  {:>10}  {:>14.2}  {:>10.2}", name, count, ms, avg_us)?;
        }
        Ok(())
    }

    pub fn reset() {
        let mut bs = BUCKETS.lock().unwrap();
        bs.clear();
    }
}

#[cfg(not(feature = "profile"))]
mod disabled {
    pub struct ScopedTimer;
    #[inline(always)]
    pub fn scoped(_name: &'static str) -> ScopedTimer {
        ScopedTimer
    }
    pub fn dump<W: std::io::Write>(_w: &mut W) -> std::io::Result<()> {
        Ok(())
    }
    pub fn reset() {}
}
