(function () {
  const THEME_KEY = "remedqa_theme";

  function applyTheme(theme) {
    const root = document.documentElement;
    if (theme === "dark") {
      root.setAttribute("data-theme", "dark");
    } else if (theme === "light") {
      root.setAttribute("data-theme", "light");
    } else {
      root.removeAttribute("data-theme");
    }
    const btn = document.querySelector(".theme-toggle");
    if (btn) {
      btn.setAttribute("aria-label", theme === "dark" ? "Switch to light mode" : theme === "light" ? "Switch to system mode" : "Switch to dark mode");
      btn.textContent = theme === "dark" ? "☀" : theme === "light" ? "◐" : "☾";
    }
  }

  function currentTheme() {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored === "dark" || stored === "light") return stored;
    return "system";
  }

  function cycleTheme() {
    const now = currentTheme();
    const next = now === "system" ? "light" : now === "light" ? "dark" : "system";
    if (next === "system") localStorage.removeItem(THEME_KEY);
    else localStorage.setItem(THEME_KEY, next);
    applyTheme(next);
  }

  document.addEventListener("DOMContentLoaded", function () {
    applyTheme(currentTheme());
    const tbtn = document.querySelector(".theme-toggle");
    if (tbtn) tbtn.addEventListener("click", cycleTheme);

    document.querySelectorAll('a[href^="#"]').forEach(function (a) {
      a.addEventListener("click", function (e) {
        const id = this.getAttribute("href").slice(1);
        if (!id) return;
        const el = document.getElementById(id);
        if (el) {
          e.preventDefault();
          el.scrollIntoView({ behavior: "smooth", block: "start" });
          if (history.replaceState) history.replaceState(null, "", "#" + id);
        }
      });
    });
  });
})();
