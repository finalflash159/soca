mod engine;

use tauri::{Manager, RunEvent};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let builder = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_process::init())
        .manage(engine::EngineState::default())
        .invoke_handler(tauri::generate_handler![
            engine::engine_start,
            engine::engine_send,
            engine::engine_stop,
            engine::engine_is_running,
            engine::engine_hello,
            engine::engine_model_root,
            engine::engine_set_model_root,
        ]);

    // Unsigned package proofs deliberately have no updater public key or
    // endpoint. Registering the plugin there makes Tauri deserialize a null
    // config and panic before the first window. The signed release workflow
    // compiles this feature only after generating its key-bearing config.
    #[cfg(feature = "release-updater")]
    let builder = builder.plugin(tauri_plugin_updater::Builder::new().build());

    builder
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
