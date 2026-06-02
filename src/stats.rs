//! Per-fit diagnostic counters.
//!
//! Shared between the outer solver, the score's value/grad/hess paths,
//! and the inner solver. Always-on; the overhead is a handful of
//! `Cell<usize>` increments per fit (~50 ns total at ms-scale fit times,
//! not measurable in benches).
//!
//! Scope: count *events* (calls, iterations, hits, attempts). No
//! `Instant::now()` timing — wall-clock belongs in the bench script.
//!
//! Wired through `ScoreDerivatives::stats()` (default `None`) so test-only
//! score impls (`QuadScore` etc.) don't need to carry counters.

use std::cell::Cell;

/// Cell-based counters owned by the score. Read after a fit completes
/// via [`FitStats::snapshot`].
#[derive(Debug, Default)]
pub struct FitStats {
    /// Outer Newton iterations actually executed.
    pub(crate) outer_iterations: Cell<usize>,
    /// Line-search trial points evaluated across all outer iters
    /// (Armijo-halving steps, including the accepted one).
    pub(crate) line_search_trials: Cell<usize>,
    /// NoRefresh IFT-shortcut probes attempted (Phase A in
    /// `profile_shape.rs` / `outer.rs` two-phase line search).
    pub(crate) no_refresh_attempts: Cell<usize>,
    /// NoRefresh probes that returned a value (i.e. the family allowed
    /// it and the η/μ guardrails passed). Hits / attempts is the IFT
    /// usability rate.
    pub(crate) no_refresh_hits: Cell<usize>,
    /// Full PIRLS inner solves invoked (excluding NoRefresh's
    /// single-step IRLS).
    pub(crate) inner_pirls_calls: Cell<usize>,
    /// Sum of inner-PIRLS iterations across all `inner_pirls_calls`.
    /// Divide by `inner_pirls_calls` for the mean per-call iter count.
    pub(crate) inner_pirls_iterations_total: Cell<usize>,
}

impl FitStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn bump_outer(&self) {
        self.outer_iterations.set(self.outer_iterations.get() + 1);
    }

    pub(crate) fn bump_line_search_trial(&self) {
        self.line_search_trials
            .set(self.line_search_trials.get() + 1);
    }

    pub(crate) fn bump_no_refresh_attempt(&self) {
        self.no_refresh_attempts
            .set(self.no_refresh_attempts.get() + 1);
    }

    pub(crate) fn bump_no_refresh_hit(&self) {
        self.no_refresh_hits.set(self.no_refresh_hits.get() + 1);
    }

    pub(crate) fn record_pirls_call(&self, iters: usize) {
        self.inner_pirls_calls.set(self.inner_pirls_calls.get() + 1);
        self.inner_pirls_iterations_total
            .set(self.inner_pirls_iterations_total.get() + iters);
    }

    /// Snapshot the current counters as a plain (Cell-free) struct.
    /// Consumers — Python bindings, bench scripts, tests — read this.
    pub fn snapshot(&self) -> FitStatsSnapshot {
        FitStatsSnapshot {
            outer_iterations: self.outer_iterations.get(),
            line_search_trials: self.line_search_trials.get(),
            no_refresh_attempts: self.no_refresh_attempts.get(),
            no_refresh_hits: self.no_refresh_hits.get(),
            inner_pirls_calls: self.inner_pirls_calls.get(),
            inner_pirls_iterations_total: self.inner_pirls_iterations_total.get(),
        }
    }

    /// Zero all counters.
    pub fn reset(&self) {
        self.outer_iterations.set(0);
        self.line_search_trials.set(0);
        self.no_refresh_attempts.set(0);
        self.no_refresh_hits.set(0);
        self.inner_pirls_calls.set(0);
        self.inner_pirls_iterations_total.set(0);
    }
}

/// Plain (Cell-free) snapshot of [`FitStats`]. Cloneable, Sendable —
/// the value you hand to Python bindings or stash in a bench result.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct FitStatsSnapshot {
    pub outer_iterations: usize,
    pub line_search_trials: usize,
    pub no_refresh_attempts: usize,
    pub no_refresh_hits: usize,
    pub inner_pirls_calls: usize,
    pub inner_pirls_iterations_total: usize,
}

impl FitStatsSnapshot {
    /// Mean inner-PIRLS iterations per call; `0.0` when no PIRLS calls
    /// have happened (Gaussian closed-form fits).
    pub fn pirls_iters_per_call(&self) -> f64 {
        if self.inner_pirls_calls == 0 {
            0.0
        } else {
            self.inner_pirls_iterations_total as f64 / self.inner_pirls_calls as f64
        }
    }

    /// Fraction of NoRefresh probes that produced a usable value;
    /// `0.0` if no probes were attempted (families on the skip list).
    pub fn no_refresh_hit_rate(&self) -> f64 {
        if self.no_refresh_attempts == 0 {
            0.0
        } else {
            self.no_refresh_hits as f64 / self.no_refresh_attempts as f64
        }
    }
}

impl std::fmt::Display for FitStatsSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "outer={} ls_trials={} pirls_calls={} pirls_iters={} (≈{:.1}/call) \
             no_refresh={}/{} ({:.0}%)",
            self.outer_iterations,
            self.line_search_trials,
            self.inner_pirls_calls,
            self.inner_pirls_iterations_total,
            self.pirls_iters_per_call(),
            self.no_refresh_hits,
            self.no_refresh_attempts,
            self.no_refresh_hit_rate() * 100.0,
        )
    }
}
