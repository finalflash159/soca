# SoCa desktop release notes

## Before installing

Only install a release whose platform signature and download source you trust.
If macOS Gatekeeper or Windows SmartScreen warns that an app is unsigned or
unrecognized, stop rather than bypass the warning. A signed public release is
published only after its release workflow has completed the signing and updater
checks and the release owner has reviewed and published the resulting draft.

## What stays on this device

SoCa keeps conversation state in memory by default. Choosing local resumable
sessions stores session data on the device in SoCa's per-user application-data
folder, along with configuration and optional vault data. The app stores this
outside the app installation so an update does not erase it.

The updater checks signed release metadata over HTTPS. It does not send session
contents or API keys as part of that check. Local-model weights are not bundled
with the desktop installer; remote use requires the provider configuration that
the user supplies.

## Upgrading and rolling back

An in-app update downloads a signed package, installs it, and relaunches the
app. It preserves local application data. If an update cannot be checked or
installed, SoCa shows the failure instead of claiming success.

To roll back, explicitly install a verified, signed earlier release compatible
with the existing session schema. Do not delete the app-data folder merely to
roll back. Export important sessions first; future schema migrations must state
their backup and compatibility behavior in their release notes.

## Uninstalling and deleting data

Uninstalling SoCa does not promise to erase its sessions, vault, or settings.
To permanently delete local data, export anything needed, close SoCa, then
remove SoCa's OS-managed per-user application-data folder. This cannot be
undone.

## Support information

When reporting an issue, include the application version, operating system and
whether the engine used a local model or a remote provider. Never attach API
keys, raw session databases, vault contents, or private conversation text to a
public issue.
