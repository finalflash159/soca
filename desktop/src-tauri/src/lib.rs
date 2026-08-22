mod engine;

use tauri::{Manager, RunEvent};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(engine::EngineState::default())
        .invoke_handler(tauri::generate_handler![
            engine::engine_start,
            engine::engine_send,
            engine::engine_stop,
            engine::engine_is_running,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            // Exit is the last point where the child is still reachable. Without
            // this, closing the window leaves `soca engine` holding the
            // microphone and any provider client it opened.
            if matches!(event, RunEvent::ExitRequested { .. } | RunEvent::Exit) {
                engine::shutdown_on_exit(&app_handle.app_handle().clone());
            }
        });
}
