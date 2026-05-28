//! Fit-time persistence — `FittedGam::serialize` / `::deserialize`.
//!
//! Wire format (binary, length-framed bincode):
//!
//! ```text
//! ┌────────────────┬───────────────┬──────────────┬───────────────────┐
//! │ MAGIC (5 B)    │ VERSION (4 B) │ LEN (8 B LE) │ bincode body      │
//! └────────────────┴───────────────┴──────────────┴───────────────────┘
//! ```
//!
//! - `MAGIC = b"GAMRS"` — quick guard against accidentally loading the
//!   wrong file format.
//! - `VERSION` — a `u32` little-endian schema tag. Bumped on breaking
//!   FittedGam / Predictor field changes; downstream consumers can use
//!   it to gate migrations. Current value: `2` (v1 was JSON-bodied).
//! - `LEN` — `u64` little-endian byte length of the bincode body that
//!   follows.
//! - bincode body — bincode 1.3 default-options serialization of
//!   [`FittedGam`]. Bincode's binary float encoding round-trips
//!   `f64` byte-for-byte (no decimal-string lossy reparse), and the
//!   wire size is ~3-5× smaller than the equivalent JSON.
//!
//! For human-debuggable serialization, use [`FittedGam::serialize_json`]
//! / [`FittedGam::deserialize_json`] — those are unframed and keep
//! `serde_json` semantics.
//!
//! Round-trip guarantees:
//!
//! - β, vcov, knots, centring, reparam_v all round-trip to bit-for-bit
//!   equality.
//! - Predictions after a `deserialize` match the in-memory original
//!   exactly (asserted by the `persistence_roundtrip` tests).
//!
//! The `LinearSolver` factorisation is NEVER serialized — we only carry
//! the materialised `vcov` (already a `FittedGam` field). This keeps the
//! wire format independent of which backend produced the fit.

use crate::error::{GamrsError, Result};
use crate::fit::FittedGam;

const MAGIC: &[u8; 5] = b"GAMRS";
const FORMAT_VERSION: u32 = 2;
const HEADER_LEN: usize = 5 + 4 + 8;

impl FittedGam {
    /// Serialize to a compact binary frame (`MAGIC | VERSION | LEN | bincode`).
    ///
    /// Returns `Vec<u8>` ready for `std::fs::write` or
    /// `std::net::TcpStream::write_all`. Round-trip with
    /// [`FittedGam::deserialize`] yields a bit-for-bit equal coefficient
    /// vector and predictor — predictions are FP-identical to the
    /// original fit (bincode encodes f64 as raw little-endian bytes).
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let body = bincode::serialize(self).map_err(|e| {
            GamrsError::InvalidParameter(format!(
                "FittedGam::serialize: bincode encode failed: {e}"
            ))
        })?;
        let len = body.len() as u64;
        let mut out = Vec::with_capacity(HEADER_LEN + body.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&body);
        Ok(out)
    }

    /// Deserialize from a buffer produced by [`FittedGam::serialize`].
    ///
    /// Validates magic, version, and length; returns
    /// `GamrsError::InvalidParameter` with row-aware context on any
    /// malformed input.
    pub fn deserialize(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_LEN {
            return Err(GamrsError::InvalidParameter(format!(
                "FittedGam::deserialize: buffer too short ({} < {} header bytes)",
                bytes.len(),
                HEADER_LEN
            )));
        }
        if &bytes[..5] != MAGIC {
            return Err(GamrsError::InvalidParameter(format!(
                "FittedGam::deserialize: bad magic; expected {:?}, got {:?}",
                MAGIC,
                &bytes[..5]
            )));
        }
        let mut v4 = [0u8; 4];
        v4.copy_from_slice(&bytes[5..9]);
        let version = u32::from_le_bytes(v4);
        if version != FORMAT_VERSION {
            return Err(GamrsError::InvalidParameter(format!(
                "FittedGam::deserialize: unsupported format_version={version} \
                 (this build expects {FORMAT_VERSION}). Re-serialize with a \
                 matching gamrs release, or roll a migration."
            )));
        }
        let mut l8 = [0u8; 8];
        l8.copy_from_slice(&bytes[9..17]);
        let body_len = u64::from_le_bytes(l8) as usize;
        let end = HEADER_LEN.checked_add(body_len).ok_or_else(|| {
            GamrsError::InvalidParameter("FittedGam::deserialize: length overflow".into())
        })?;
        if bytes.len() < end {
            return Err(GamrsError::InvalidParameter(format!(
                "FittedGam::deserialize: truncated body (have {} bytes, header \
                 says {body_len} payload bytes follow the {HEADER_LEN}-byte header)",
                bytes.len() - HEADER_LEN
            )));
        }
        let body = &bytes[HEADER_LEN..end];
        let fitted: FittedGam = bincode::deserialize(body).map_err(|e| {
            GamrsError::InvalidParameter(format!(
                "FittedGam::deserialize: bincode decode failed: {e}"
            ))
        })?;
        Ok(fitted)
    }

    /// Serialize as plain UTF-8 JSON (no framing). Useful for diffing
    /// two fits, hand-inspection, or piping through `jq`. Not
    /// version-tagged — only [`FittedGam::serialize`] carries the
    /// `format_version` byte.
    pub fn serialize_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|e| {
            GamrsError::InvalidParameter(format!("FittedGam::serialize_json: encode failed: {e}"))
        })
    }

    /// Inverse of [`FittedGam::serialize_json`]. Accepts unframed JSON.
    pub fn deserialize_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).map_err(|e| {
            GamrsError::InvalidParameter(format!(
                "FittedGam::deserialize_json: parse failed at line {}, column {}: {e}",
                e.line(),
                e.column()
            ))
        })
    }

    /// Current serialization format version. Bumped on breaking schema
    /// changes; exposed so downstream code can branch on the value
    /// read from a serialized bundle.
    pub const FORMAT_VERSION: u32 = FORMAT_VERSION;
}
