/** Minimal browser APIs the desktop WebView has but JSDOM deliberately omits. */
if (typeof HTMLElement !== "undefined" && typeof HTMLElement.prototype.getAnimations !== "function") {
  Object.defineProperty(HTMLElement.prototype, "getAnimations", {
    configurable: true,
    value: () => [],
  });
}

if (typeof HTMLCanvasElement !== "undefined") {
  Object.defineProperty(HTMLCanvasElement.prototype, "getContext", {
    configurable: true,
    value: () => null,
  });
}
