//! Fit-time persistence — `FittedGam::serialize` / `::deserialize`.
//!
//! Wire format (binary-ish, length-framed JSON):
//!
//! ```text
//! ┌────────────────┬───────────────┬──────────────┬──────────────────┐
//! │ MAGIC (6 B)    │ VERSION (4 B) │ LEN (8 B LE) │ JSON body (UTF-8)│
//! └────────────────┴───────────────┴──────────────┴──────────────────┘
//! ```
//!
//! - `MAGIC = b"GAMMON"` — quick guard against accidentally loading the
//!   wrong file format.
//! - `VERSION` — a `u32` little-endian schema tag. Bumped on breaking
//!   FittedGam / Predictor field changes; downstream consumers can use
//!   it to gate migrations. Current value: `1`.
//! - `LEN` — `u64` little-endian byte length of the JSON body that
//!   follows. Lets the reader know how much to slice off without
//!   trusting the input length.
//! - JSON body — serde_json serialization of [`FittedGam`]. JSON over
//!   bincode here because (a) the gammon dependency surface is already
//!   minimal and serde_json was the only binary serializer cached at
//!   crate-publish time; (b) JSON is debuggable — a developer can
//!   pop the body out, inspect it, diff two fits — without a custom
//!   tool. Cost is ~3-5× larger payload than bincode, which for a
//!   single-smooth GAM (10 knots → p=10) is still a few KB.
//!
//! Round-trip guarantees:
//!
//! - β, vcov, knots, centring, reparam_v all round-trip to bit-for-bit
//!   equality (JSON serializes f64 with full-precision text format and
//!   serde_json parses it back to the same bits).
//! - Predictions after a `deserialize` match the in-memory original
//!   exactly (asserted by the `persistence_roundtrip` tests).
//!
//! The `LinearSolver` factorisation is NEVER serialized — we only carry
//! the materialised `vcov` (already a `FittedGam` field). This keeps the
//! wire format independent of which backend produced the fit.

use crate::error::{GammonError, Result};
use crate::fit::FittedGam;

const MAGIC: &[u8; 6] = b"GAMMON";
const FORMAT_VERSION: u32 = 1;
const HEADER_LEN: usize = 6 + 4 + 8;

impl FittedGam {
    /// Serialize to a compact binary frame (`MAGIC | VERSION | LEN | JSON`).
    ///
    /// Returns `Vec<u8>` ready for `std::fs::write` or
    /// `std::net::TcpStream::write_all`. Round-trip with
    /// [`FittedGam::deserialize`] yields a bit-for-bit equal coefficient
    /// vector and predictor — predictions are FP-identical to the
    /// original fit.
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let body = serde_json::to_vec(self).map_err(|e| {
            GammonError::InvalidParameter(format!(
                "FittedGam::serialize: JSON encode failed: {e}"
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
    /// `GammonError::InvalidParameter` with row-aware context on any
    /// malformed input.
    pub fn deserialize(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_LEN {
            return Err(GammonError::InvalidParameter(format!(
                "FittedGam::deserialize: buffer too short ({} < {} header bytes)",
                bytes.len(),
                HEADER_LEN
            )));
        }
        if &bytes[..6] != MAGIC {
            return Err(GammonError::InvalidParameter(format!(
                "FittedGam::deserialize: bad magic; expected {:?}, got {:?}",
                MAGIC,
                &bytes[..4]
            )));
        }
        let mut v4 = [0u8; 4];
        v4.copy_from_slice(&bytes[4..8]);
        let version = u32::from_le_bytes(v4);
        if version != FORMAT_VERSION {
            return Err(GammonError::InvalidParameter(format!(
                "FittedGam::deserialize: unsupported format_version={version} \
                 (this build expects {FORMAT_VERSION}). Re-serialize with a \
                 matching gammon release, or roll a migration."
            )));
        }
        let mut l8 = [0u8; 8];
        l8.copy_from_slice(&bytes[8..16]);
        let body_len = u64::from_le_bytes(l8) as usize;
        let end = HEADER_LEN
            .checked_add(body_len)
            .ok_or_else(|| GammonError::InvalidParameter(
                "FittedGam::deserialize: length overflow".into(),
            ))?;
        if bytes.len() < end {
            return Err(GammonError::InvalidParameter(format!(
                "FittedGam::deserialize: truncated body (have {} bytes, header \
                 says {body_len} payload bytes follow the {HEADER_LEN}-byte header)",
                bytes.len() - HEADER_LEN
            )));
        }
        let body = &bytes[HEADER_LEN..end];
        let fitted: FittedGam = serde_json::from_slice(body).map_err(|e| {
            GammonError::InvalidParameter(format!(
                "FittedGam::deserialize: JSON decode failed at byte {}: {e}",
                e.column()
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
            GammonError::InvalidParameter(format!(
                "FittedGam::serialize_json: encode failed: {e}"
            ))
        })
    }

    /// Inverse of [`FittedGam::serialize_json`]. Accepts unframed JSON.
    pub fn deserialize_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).map_err(|e| {
            GammonError::InvalidParameter(format!(
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
