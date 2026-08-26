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

/// Begin microphone authorization on the native application main thread.
///
/// TCC attributes the request to the GUI bundle which calls AVFoundation. The
/// completion arrives asynchronously, so this function never blocks the main
/// thread while macOS displays its consent sheet.
#[cfg(target_os = "macos")]
fn begin_microphone_access(
    sender: std::sync::mpsc::SyncSender<Result<MicrophonePermission, String>>,
) {
    use block2::RcBlock;
    use objc2_av_foundation::{AVAuthorizationStatus, AVCaptureDevice, AVMediaTypeAudio};

    let audio_media_type = unsafe {
        AVMediaTypeAudio.ok_or_else(|| "macOS did not provide the audio media type.".to_string())
    };
    let Ok(audio_media_type) = audio_media_type else {
        let _ = sender.send(Err(
            "macOS did not provide the audio media type.".to_string()
        ));
        return;
    };
    let current = unsafe { AVCaptureDevice::authorizationStatusForMediaType(audio_media_type) };
    if current != AVAuthorizationStatus::NotDetermined {
        let _ = sender.send(from_native_status(current));
        return;
    }

    let completion = RcBlock::new(move |granted| {
        // The receiver may already have timed out; delivery is deliberately
        // best-effort because the system owns the completion timing.
        let permission = if bool::from(granted) {
            MicrophonePermission::Authorized
        } else {
            MicrophonePermission::Denied
        };
        let _ = sender.send(Ok(permission));
    });
    unsafe {
        AVCaptureDevice::requestAccessForMediaType_completionHandler(audio_media_type, &completion);
    }
}

#[cfg(target_os = "macos")]
fn wait_for_microphone_access(
    receiver: std::sync::mpsc::Receiver<Result<MicrophonePermission, String>>,
) -> Result<MicrophonePermission, String> {
    use std::sync::mpsc;
    use std::time::Duration;

    match receiver.recv_timeout(Duration::from_secs(60)) {
        Ok(result) => result,
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
fn request_microphone_access() -> Result<MicrophonePermission, String> {
    Ok(MicrophonePermission::Authorized)
}

#[tauri::command]
pub async fn microphone_request_access(
    app: tauri::AppHandle,
) -> Result<MicrophonePermission, String> {
    #[cfg(target_os = "macos")]
    {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        app.run_on_main_thread(move || begin_microphone_access(sender))
            .map_err(|error| {
                format!("Could not start the microphone permission request: {error}")
            })?;

        return tauri::async_runtime::spawn_blocking(move || wait_for_microphone_access(receiver))
            .await
            .map_err(|error| format!("Microphone permission task failed: {error}"))?;
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = app;
        request_microphone_access()
    }
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
