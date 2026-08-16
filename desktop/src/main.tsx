import React from "react";
import ReactDOM from "react-dom/client";

import App from "./App";
// The only stylesheet entry point. Without this the whole Tailwind/shadcn layer
// never reaches the bundle and every component renders as unstyled HTML.
import "./index.css";
import { initTheme } from "./theme";

// Before the first paint, not in an effect: the stylesheet defaults to light,
// so a dark-mode user would otherwise get a white flash on every launch.
initTheme();

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
