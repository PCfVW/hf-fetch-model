// SPDX-License-Identifier: MIT OR Apache-2.0

//! Progress reporting for model downloads.
//!
//! [`ProgressEvent`] carries per-file and overall download status.
//! When the `indicatif` feature is enabled, `IndicatifProgress`
//! provides multi-progress bars out of the box.

/// A progress event emitted during download.
///
/// Passed to the `on_progress` callback on [`crate::FetchConfig`].
#[derive(Debug, Clone)]
pub struct ProgressEvent {
    /// The filename currently being downloaded.
    pub filename: String,
    /// Bytes downloaded so far for this file.
    pub bytes_downloaded: u64,
    /// Total size of this file in bytes (0 if unknown).
    pub bytes_total: u64,
    /// Download percentage for this file (0.0–100.0).
    pub percent: f64,
    /// Number of files still remaining (after this one).
    pub files_remaining: usize,
}

/// Creates a [`ProgressEvent`] for a completed file.
#[must_use]
pub(crate) fn completed_event(filename: &str, size: u64, files_remaining: usize) -> ProgressEvent {
    ProgressEvent {
        filename: filename.to_owned(),
        bytes_downloaded: size,
        bytes_total: size,
        percent: 100.0,
        files_remaining,
    }
}

/// Multi-progress bar display using `indicatif`.
///
/// Available only when the `indicatif` feature is enabled.
///
/// # Example
///
/// ```rust,no_run
/// # fn example() -> Result<(), hf_fetch_model::FetchError> {
/// use hf_fetch_model::FetchConfig;
/// # #[cfg(feature = "indicatif")]
/// use hf_fetch_model::progress::IndicatifProgress;
///
/// # #[cfg(feature = "indicatif")]
/// let progress = IndicatifProgress::new();
/// let config = FetchConfig::builder()
///     # ;
///     # #[cfg(feature = "indicatif")]
///     # let config = FetchConfig::builder()
///     .on_progress(move |e| progress.handle(e))
///     .build()?;
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "indicatif")]
pub struct IndicatifProgress {
    multi: indicatif::MultiProgress,
    overall: indicatif::ProgressBar,
}

#[cfg(feature = "indicatif")]
impl IndicatifProgress {
    /// Creates a new multi-progress bar display.
    ///
    /// Call [`IndicatifProgress::set_total_files`] once the file count is known.
    #[must_use]
    pub fn new() -> Self {
        let multi = indicatif::MultiProgress::new();
        let overall = multi.add(indicatif::ProgressBar::new(0));
        overall.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("{msg} [{bar:40.cyan/blue}] {pos}/{len} files")
                .ok()
                .unwrap_or_else(indicatif::ProgressStyle::default_bar)
                .progress_chars("=> "),
        );
        overall.set_message("Overall");
        Self { multi, overall }
    }

    /// Returns a reference to the underlying [`indicatif::MultiProgress`].
    ///
    /// Useful for adding custom progress bars alongside the built-in ones.
    #[must_use]
    pub fn multi(&self) -> &indicatif::MultiProgress {
        &self.multi
    }

    /// Sets the total number of files to download.
    pub fn set_total_files(&self, total: u64) {
        self.overall.set_length(total);
    }

    /// Handles a [`ProgressEvent`], updating progress bars.
    pub fn handle(&self, event: &ProgressEvent) {
        if event.percent >= 100.0 {
            // Derive total: completed so far + this file + remaining
            // EXPLICIT: try_from for usize → u64 (infallible on 64-bit, safe fallback otherwise)
            let remaining = u64::try_from(event.files_remaining).unwrap_or(u64::MAX);
            let total = self.overall.position() + 1 + remaining;
            self.overall.set_length(total);
            self.overall.inc(1);
        }
    }
}

#[cfg(feature = "indicatif")]
impl Default for IndicatifProgress {
    fn default() -> Self {
        Self::new()
    }
}
