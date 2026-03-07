// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retry logic with exponential backoff and jitter.
//!
//! Used internally by the download orchestrator to retry transient failures.

use std::future::Future;
use std::time::Duration;

use crate::error::FetchError;

/// Configuration for retry behavior.
#[derive(Debug, Clone)]
pub(crate) struct RetryPolicy {
    /// Maximum number of retry attempts (0 = no retries).
    pub max_retries: u32,
    /// Base delay between retries.
    pub base_delay: Duration,
    /// Maximum delay cap.
    pub max_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(300),
            max_delay: Duration::from_secs(10),
        }
    }
}

/// Executes an async operation with retry on transient failures.
///
/// Returns the first successful result, or the last error after all retries
/// are exhausted. The `is_retryable` closure determines whether a given
/// error should trigger a retry.
pub(crate) async fn retry_async<F, Fut, T>(
    policy: &RetryPolicy,
    is_retryable: fn(&FetchError) -> bool,
    mut operation: F,
) -> Result<T, FetchError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, FetchError>>,
{
    let mut last_error: Option<FetchError> = None;

    for attempt in 0..=policy.max_retries {
        match operation().await {
            Ok(value) => return Ok(value),
            Err(e) => {
                if attempt == policy.max_retries || !is_retryable(&e) {
                    return Err(e);
                }
                last_error = Some(e);

                let delay = compute_delay(policy, attempt);
                tokio::time::sleep(delay).await;
            }
        }
    }

    // This is unreachable in practice (the loop always returns), but
    // satisfies the compiler without unwrap().
    Err(last_error.unwrap_or_else(|| FetchError::Http("retry exhausted".to_owned())))
}

/// Computes the delay for a given attempt using exponential backoff + jitter.
fn compute_delay(policy: &RetryPolicy, attempt: u32) -> Duration {
    // Exponential backoff: base * 2^attempt
    let exp_delay = policy.base_delay.saturating_mul(1u32.wrapping_shl(attempt));
    let capped = exp_delay.min(policy.max_delay);

    // Jitter: scale to 50–100% of the capped delay using system time nanoseconds.
    let jitter_nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    let jitter_fraction = u64::from(jitter_nanos % 500) + 500; // 500–999 out of 1000
    let capped_millis = u64::try_from(capped.as_millis()).unwrap_or(u64::MAX);
    let jittered_millis = capped_millis.saturating_mul(jitter_fraction) / 1000;

    Duration::from_millis(jittered_millis)
}

/// Returns whether a [`FetchError`] is likely transient and worth retrying.
///
/// HTTP 416 Range Not Satisfiable is deterministic (the server will never
/// support Range for that file), so it is excluded from retries.
#[must_use]
pub(crate) fn is_retryable(error: &FetchError) -> bool {
    match error {
        FetchError::Api(e) => {
            let msg = e.to_string();
            // 416 Range Not Satisfiable is deterministic, not transient.
            !msg.contains("416")
        }
        FetchError::Http(_) | FetchError::Timeout { .. } | FetchError::ChunkedDownload { .. } => {
            true
        }
        FetchError::Io { .. }
        | FetchError::RepoNotFound { .. }
        | FetchError::Auth { .. }
        | FetchError::InvalidPattern { .. }
        | FetchError::Checksum { .. }
        | FetchError::PartialDownload { .. }
        | FetchError::InvalidArgument(_)
        | FetchError::NoFilesMatched { .. } => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_policy() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_retries, 3);
        assert_eq!(policy.base_delay, Duration::from_millis(300));
        assert_eq!(policy.max_delay, Duration::from_secs(10));
    }

    #[test]
    fn test_compute_delay_capped() {
        let policy = RetryPolicy {
            max_retries: 3,
            base_delay: Duration::from_secs(5),
            max_delay: Duration::from_secs(10),
        };
        // Attempt 3: 5 * 2^3 = 40s, capped to 10s, then jittered to 5–10s range
        let delay = compute_delay(&policy, 3);
        assert!(delay <= Duration::from_secs(10));
        assert!(delay >= Duration::from_millis(5000));
    }

    #[test]
    fn test_is_retryable() {
        assert!(is_retryable(&FetchError::Http("timeout".to_owned())));
        assert!(is_retryable(&FetchError::Timeout {
            filename: "f".to_owned(),
            seconds: 30,
        }));
        assert!(!is_retryable(&FetchError::RepoNotFound {
            repo_id: "x".to_owned(),
        }));
        assert!(!is_retryable(&FetchError::Auth {
            reason: "bad token".to_owned(),
        }));
    }
}
