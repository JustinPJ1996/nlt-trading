// Light or dark. Follows the computer until the switch in the menu is used,
// then remembers the choice on this browser.

export type Theme = "light" | "dark";
const KEY = "sd-theme";

export function savedTheme(): Theme | null {
  const t = localStorage.getItem(KEY);
  return t === "light" || t === "dark" ? t : null;
}

export function systemTheme(): Theme {
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
}

export function chooseTheme(theme: Theme): void {
  localStorage.setItem(KEY, theme);
  applyTheme(theme);
}

/** Called once before the first render. */
export function startTheme(): void {
  applyTheme(savedTheme() ?? systemTheme());
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (!savedTheme()) applyTheme(systemTheme());
  });
}
