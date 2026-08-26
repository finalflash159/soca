//! Native microphone authorization for the desktop bundle.
//!
//! The engine is a separate bundled Python executable. Asking CoreAudio from
//! that child alone is not a reliable way to make macOS attribute consent to
//! SoCa's GUI bundle, so authorization is deliberately requested here first.

use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MicrophonePermission {
    Authorized,
    Denied,
    Restricted,
}

#[cfg(test)]
impl MicrophonePermission {
    pub fn user_message(self) -> &'static str {
        match self {
            Self::Authorized => "Microphone access is available.",
            Self::Denied => "SoCa needs microphone access. Allow SoCa in System Settings > Privacy & Security > Microphone, then try again.",
            Self::Restricted => "Microphone access is restricted by this Mac. Check the device's privacy restrictions, then try again.",
        }
    }
}

#[cfg(target_os = "macos")]
fn from_native_status(
    status: objc2_av_foundation::AVAuthorizationStatus,
) -> Result<MicrophonePermission, String> {
    use objc2_av_foundation::AVAuthorizationStatus;

    if status == AVAuthorizationStatus::Authorized {
        Ok(MicrophonePermission::Authorized)
    } else if status == AVAuthorizationStatus::Denied {
        Ok(MicrophonePermission::Denied)
    } else if status == AVAuthorizationStatus::Restricted {
        Ok(MicrophonePermission::Restricted)
    } else {
        Err("Microphone permission request did not resolve to a terminal state.".to_string())
    }
}

/// Request microphone permission from the actual macOS application bundle.
///
/// Tauri runs this command away from the WebView event loop, so waiting for the
/// system sheet does not freeze the UI. The wait is bounded and no sidecar is
/// launched unless macOS returns an explicit authorization result.
#[cfg(target_os = "macos")]
pub fn request_microphone_access() -> Result<MicrophonePermission, String> {
    use std::sync::mpsc;
    use std::time::Duration;

    use block2::RcBlock;
    use objc2_av_foundation::{AVAuthorizationStatus, AVCaptureDevice, AVMediaTypeAudio};

    let audio_media_type = unsafe {
        AVMediaTypeAudio
            .ok_or_else(|| "macOS did not provide the audio media type.".to_string())?
    };
    let current = unsafe { AVCaptureDevice::authorizationStatusForMediaType(audio_media_type) };
    if current != AVAuthorizationStatus::NotDetermined {
        return from_native_status(current);
    }

    let (sender, receiver) = mpsc::sync_channel(1);
    let completion = RcBlock::new(move |granted| {
        // The receiver may already have timed out; the sender is intentionally
        // best-effort because the system owns the completion timing.
        let _ = sender.send(bool::from(granted));
    });
    unsafe {
        AVCaptureDevice::requestAccessForMediaType_completionHandler(audio_media_type, &completion);
    }

    match receiver.recv_timeout(Duration::from_secs(60)) {
        Ok(true) => Ok(MicrophonePermission::Authorized),
        Ok(false) => {
            let final_status = unsafe { AVCaptureDevice::authorizationStatusForMediaType(audio_media_type) };
            from_native_status(final_status)
        }
        Err(mpsc::RecvTimeoutError::Timeout) => Err(
            "Microphone permission is still waiting for a response. Finish the macOS prompt, then try again."
                .to_string(),
        ),
        Err(mpsc::RecvTimeoutError::Disconnected) => Err(
            "macOS closed the microphone permission request before it completed. Try again."
                .to_string(),
        ),
    }
}

/// Platforms without macOS TCC leave device authorization to their native
/// audio stack. This preserves a single frontend contract without claiming a
/// macOS-specific permission result elsewhere.
#[cfg(not(target_os = "macos"))]
pub fn request_microphone_access() -> Result<MicrophonePermission, String> {
    Ok(MicrophonePermission::Authorized)
}

#[tauri::command]
pub async fn microphone_request_access() -> Result<MicrophonePermission, String> {
    tauri::async_runtime::spawn_blocking(request_microphone_access)
        .await
        .map_err(|error| format!("Microphone permission task failed: {error}"))?
}

#[cfg(test)]
mod tests {
    use super::MicrophonePermission;

    #[test]
    fn permission_messages_are_actionable() {
        assert!(MicrophonePermission::Denied
            .user_message()
            .contains("System Settings"));
        assert!(MicrophonePermission::Restricted
            .user_message()
            .contains("restricted"));
    }
}
