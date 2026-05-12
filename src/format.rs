// SPDX-License-Identifier: MIT OR Apache-2.0

//! Display formatters shared across binary subcommands.
//!
//! Currently exports [`format_size`] for human-readable byte counts. Future
//! formatters that are shared across more than one CLI subcommand (age,
//! short-`SHA`, parameter counts) belong here too.

/// Formats a byte count as a human-readable string with binary `IEC` units.
///
/// Buckets:
///
/// | Range | Format |
/// |-------|--------|
/// | `< 1024` | `"{N} B"` |
/// | `< 1 MiB` | `"{X.X} KiB"` |
/// | `< 1000 MiB` | `"{X.XX} MiB"` |
/// | `< 1000 GiB` | `"{X.XX} GiB"` |
/// | otherwise | `"{X.XX} TiB"` |
///
/// The `< 1000 MiB` / `< 1000 GiB` thresholds (rather than `< 1024`) match the
/// pre-extraction CLI convention: a 1.0 GiB file prints as `"1024.00 MiB"` only
/// up to 999.99 MiB, then flips to `"1.00 GiB"`. Same for the `GiB → TiB` flip.
#[must_use]
pub fn format_size(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * 1024 * 1024;
    const TIB: u64 = 1024 * GIB;

    if bytes >= 1000 * GIB {
        // CAST: u64 → f64, precision loss acceptable; value is a display-only size scalar
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let val = bytes as f64 / TIB as f64;
        format!("{val:.2} TiB")
    } else if bytes >= 1000 * MIB {
        // CAST: u64 → f64, precision loss acceptable; value is a display-only size scalar
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let val = bytes as f64 / GIB as f64;
        format!("{val:.2} GiB")
    } else if bytes >= MIB {
        // CAST: u64 → f64, precision loss acceptable; value is a display-only size scalar
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let val = bytes as f64 / MIB as f64;
        format!("{val:.2} MiB")
    } else if bytes >= KIB {
        // CAST: u64 → f64, precision loss acceptable; value is a display-only size scalar
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let val = bytes as f64 / KIB as f64;
        format!("{val:.1} KiB")
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::format_size;

    #[test]
    fn bytes_under_kib() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(1), "1 B");
        assert_eq!(format_size(1023), "1023 B");
    }

    #[test]
    fn kib_range() {
        assert_eq!(format_size(1024), "1.0 KiB");
        assert_eq!(format_size(1536), "1.5 KiB");
    }

    #[test]
    fn mib_range() {
        assert_eq!(format_size(1024 * 1024), "1.00 MiB");
        assert_eq!(format_size(10 * 1024 * 1024), "10.00 MiB");
    }

    #[test]
    fn gib_range_kicks_in_at_1000_mib() {
        // 999 MiB still formats as MiB; 1000 MiB flips to GiB.
        assert_eq!(format_size(999 * 1024 * 1024), "999.00 MiB");
        let gib = 1u64 << 30;
        assert_eq!(format_size(gib), "1.00 GiB");
    }

    #[test]
    fn tib_range_kicks_in_at_1000_gib() {
        let gib = 1u64 << 30;
        assert_eq!(format_size(999 * gib), "999.00 GiB");
        let tib = 1024 * gib;
        assert_eq!(format_size(tib), "1.00 TiB");
    }
}
