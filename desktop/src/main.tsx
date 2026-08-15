import React from "react";
import ReactDOM from "react-dom/client";

import App from "./App";
// The only stylesheet entry point. Without this the whole Tailwind/shadcn layer
// never reaches the bundle and every component renders as unstyled HTML.
import "./index.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
