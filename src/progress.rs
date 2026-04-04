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

/// Creates a [`ProgressEvent`] for an in-progress file (streaming update).
#[must_use]
pub(crate) fn streaming_event(
    filename: &str,
    bytes_downloaded: u64,
    bytes_total: u64,
    files_remaining: usize,
) -> ProgressEvent {
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    // CAST: u64 → f64, precision loss acceptable; values are display-only percentage scalars
    let percent = if bytes_total > 0 {
        (bytes_downloaded as f64 / bytes_total as f64) * 100.0
    } else {
        0.0
    };
    ProgressEvent {
        filename: filename.to_owned(),
        bytes_downloaded,
        bytes_total,
        percent,
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
    // Multi-progress container for all bars.
    multi: indicatif::MultiProgress,
    // Overall file-count bar (always the last bar in the display).
    overall: indicatif::ProgressBar,
    // Per-file progress bars, keyed by filename.
    file_bars: std::sync::Mutex<std::collections::HashMap<String, indicatif::ProgressBar>>,
    // Filenames already counted as complete (deduplicates chunked + orchestrator events).
    completed_files: std::sync::Mutex<std::collections::HashSet<String>>,
    // Guards against double-finish on drop.
    finished: std::sync::atomic::AtomicBool,
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
        Self {
            multi,
            overall,
            file_bars: std::sync::Mutex::new(std::collections::HashMap::new()),
            completed_files: std::sync::Mutex::new(std::collections::HashSet::new()),
            finished: std::sync::atomic::AtomicBool::new(false),
        }
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
    ///
    /// For in-progress events, creates or updates a per-file progress bar
    /// showing bytes downloaded, throughput, and ETA. On completion, the
    /// per-file bar is finished and the overall file counter is incremented.
    pub fn handle(&self, event: &ProgressEvent) {
        if event.percent >= 100.0 {
            // Remove and finish per-file bar if it exists.
            if let Ok(mut bars) = self.file_bars.lock() {
                if let Some(bar) = bars.remove(&event.filename) {
                    bar.finish_and_clear();
                }
            }
            // Deduplicate: chunked downloads fire a streaming 100% event,
            // then the orchestrator fires a completed_event for the same file.
            let is_new = self
                .completed_files
                .lock()
                .is_ok_and(|mut set| set.insert(event.filename.clone()));
            if is_new {
                // Derive total: completed so far + this file + remaining
                // EXPLICIT: try_from for usize → u64 (infallible on 64-bit, safe fallback otherwise)
                let remaining = u64::try_from(event.files_remaining).unwrap_or(u64::MAX);
                let total = self.overall.position() + 1 + remaining;
                self.overall.set_length(total);
                self.overall.inc(1);
            }
        } else if event.bytes_total > 0 {
            // In-progress streaming update — create or update per-file bar.
            if let Ok(mut bars) = self.file_bars.lock() {
                let bar = bars.entry(event.filename.clone()).or_insert_with(|| {
                    let pb = self.multi.insert_before(
                        &self.overall,
                        indicatif::ProgressBar::new(event.bytes_total),
                    );
                    pb.set_style(
                        indicatif::ProgressStyle::default_bar()
                            .template(
                                "{msg} [{bar:40.green/dim}] {bytes}/{total_bytes} {bytes_per_sec} ({eta})",
                            )
                            .ok()
                            .unwrap_or_else(indicatif::ProgressStyle::default_bar)
                            .progress_chars("=> "),
                    );
                    pb.set_message(event.filename.clone());
                    pb
                });
                bar.set_position(event.bytes_downloaded);
            }
        }
    }

    /// Finishes the progress bar, ensuring the final state is rendered.
    ///
    /// Called automatically on drop, but can be called explicitly for
    /// immediate visual feedback.
    pub fn finish(&self) {
        if !self
            .finished
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            self.overall.finish();
        }
    }
}

#[cfg(feature = "indicatif")]
impl Drop for IndicatifProgress {
    fn drop(&mut self) {
        self.finish();
    }
}

#[cfg(feature = "indicatif")]
impl Default for IndicatifProgress {
    fn default() -> Self {
        Self::new()
    }
}
