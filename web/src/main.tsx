import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource-variable/public-sans";
import "@fontsource-variable/bricolage-grotesque";
import "./styles.css";
import { App } from "./App";
import { startTheme } from "./theme";

startTheme();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
