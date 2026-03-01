/**
 * CardioPredict — Client-side JS
 * Author: Sandeep Kumar
 */
document.addEventListener("DOMContentLoaded", function () {

    // =========================================================================
    //  Mobile nav
    // =========================================================================
    var hamburger = document.getElementById("nav-hamburger");
    var navLinks = document.getElementById("nav-links");
    if (hamburger && navLinks) {
        hamburger.addEventListener("click", function () {
            navLinks.classList.toggle("open");
        });
    }

    // =========================================================================
    //  Display Settings (dark mode, font size, font family)
    // =========================================================================
    var FONT_FAMILIES = {
        "Inter": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "Roboto": "'Roboto', -apple-system, sans-serif",
        "Outfit": "'Outfit', -apple-system, sans-serif",
        "Merriweather": "'Merriweather', Georgia, serif",
        "Source Code Pro": "'Source Code Pro', 'SF Mono', Consolas, monospace",
        "System": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    };

    var DEFAULTS = { theme: "light", fontSize: 100, fontFamily: "Inter" };

    // --- Load saved preferences immediately (before paint) ---
    function getPrefs() {
        try {
            var saved = JSON.parse(localStorage.getItem("cp_display_prefs"));
            if (saved) return {
                theme: saved.theme || DEFAULTS.theme,
                fontSize: saved.fontSize || DEFAULTS.fontSize,
                fontFamily: saved.fontFamily || DEFAULTS.fontFamily
            };
        } catch (e) { /* ignore */ }
        return Object.assign({}, DEFAULTS);
    }

    function savePrefs(prefs) {
        localStorage.setItem("cp_display_prefs", JSON.stringify(prefs));
    }

    var prefs = getPrefs();

    // --- Apply preferences ---
    function applyTheme(theme) {
        document.documentElement.setAttribute("data-theme", theme);
        prefs.theme = theme;
        savePrefs(prefs);
        updateThemeButtons();
    }

    function applyFontSize(size) {
        size = Math.max(80, Math.min(130, size));
        document.documentElement.style.fontSize = size + "%";
        prefs.fontSize = size;
        savePrefs(prefs);
        updateFontSizeUI();
    }

    function applyFontFamily(name) {
        var stack = FONT_FAMILIES[name] || FONT_FAMILIES["Inter"];
        document.documentElement.style.setProperty("--font", stack);
        prefs.fontFamily = name;
        savePrefs(prefs);
        updateFontFamilyButtons();
    }

    // Apply immediately
    applyTheme(prefs.theme);
    applyFontSize(prefs.fontSize);
    applyFontFamily(prefs.fontFamily);

    // --- Settings drawer toggle ---
    var settingsToggle = document.getElementById("settings-toggle");
    var settingsDrawer = document.getElementById("settings-drawer");
    var settingsOverlay = document.getElementById("settings-overlay");
    var settingsClose = document.getElementById("settings-close");

    function openSettings() {
        settingsDrawer.classList.add("open");
        settingsOverlay.classList.add("open");
    }
    function closeSettings() {
        settingsDrawer.classList.remove("open");
        settingsOverlay.classList.remove("open");
    }

    if (settingsToggle) settingsToggle.addEventListener("click", openSettings);
    if (settingsClose) settingsClose.addEventListener("click", closeSettings);
    if (settingsOverlay) settingsOverlay.addEventListener("click", closeSettings);

    // Close on Escape key
    document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") closeSettings();
    });

    // --- Theme toggle ---
    function updateThemeButtons() {
        var btns = document.querySelectorAll(".theme-btn");
        btns.forEach(function (b) {
            b.classList.toggle("active", b.getAttribute("data-theme") === prefs.theme);
        });
    }

    var themeToggle = document.getElementById("theme-toggle");
    if (themeToggle) {
        themeToggle.addEventListener("click", function (e) {
            var btn = e.target.closest(".theme-btn");
            if (!btn) return;
            applyTheme(btn.getAttribute("data-theme"));
        });
    }

    // --- Font size ---
    var fontSlider = document.getElementById("font-size-slider");
    var fontValue = document.getElementById("font-size-value");
    var fontDecrease = document.getElementById("font-decrease");
    var fontIncrease = document.getElementById("font-increase");

    function updateFontSizeUI() {
        if (fontSlider) fontSlider.value = prefs.fontSize;
        if (fontValue) fontValue.textContent = prefs.fontSize + "%";
    }

    if (fontSlider) {
        fontSlider.addEventListener("input", function () {
            applyFontSize(parseInt(fontSlider.value, 10));
        });
    }
    if (fontDecrease) {
        fontDecrease.addEventListener("click", function () {
            applyFontSize(prefs.fontSize - 5);
        });
    }
    if (fontIncrease) {
        fontIncrease.addEventListener("click", function () {
            applyFontSize(prefs.fontSize + 5);
        });
    }

    // --- Font family ---
    function updateFontFamilyButtons() {
        var btns = document.querySelectorAll(".font-family-btn");
        btns.forEach(function (b) {
            b.classList.toggle("active", b.getAttribute("data-font") === prefs.fontFamily);
        });
    }

    var fontFamilyOptions = document.getElementById("font-family-options");
    if (fontFamilyOptions) {
        fontFamilyOptions.addEventListener("click", function (e) {
            var btn = e.target.closest(".font-family-btn");
            if (!btn) return;
            applyFontFamily(btn.getAttribute("data-font"));
        });
    }

    // --- Reset ---
    var resetBtn = document.getElementById("settings-reset");
    if (resetBtn) {
        resetBtn.addEventListener("click", function () {
            applyTheme(DEFAULTS.theme);
            applyFontSize(DEFAULTS.fontSize);
            applyFontFamily(DEFAULTS.fontFamily);
        });
    }

    // Sync UI on load
    updateThemeButtons();
    updateFontSizeUI();
    updateFontFamilyButtons();


    // =========================================================================
    //  Info panels (toggle on ⓘ click)
    // =========================================================================
    var infoButtons = document.querySelectorAll(".info-btn");
    infoButtons.forEach(function (btn) {
        btn.addEventListener("click", function () {
            var field = btn.getAttribute("data-field");
            var panel = document.getElementById("info-" + field);
            if (!panel) return;

            // Close any other open panels
            document.querySelectorAll(".info-panel.open").forEach(function (p) {
                if (p !== panel) {
                    p.classList.remove("open");
                    var otherField = p.id.replace("info-", "");
                    var otherBtn = document.querySelector('.info-btn[data-field="' + otherField + '"]');
                    if (otherBtn) otherBtn.classList.remove("active");
                }
            });

            // Toggle this panel
            panel.classList.toggle("open");
            btn.classList.toggle("active");
        });
    });

    // =========================================================================
    //  Form validation feedback
    // =========================================================================
    var form = document.getElementById("heart-form");
    if (form) {
        form.addEventListener("submit", function (e) {
            var inputs = form.querySelectorAll("[required]");
            var allValid = true;

            inputs.forEach(function (input) {
                if (!input.value || input.value === "") {
                    allValid = false;
                    input.style.borderColor = "#ef4444";
                } else {
                    input.style.borderColor = "";
                }
            });

            if (!allValid) {
                e.preventDefault();
                var firstInvalid = form.querySelector("[required]:invalid");
                if (firstInvalid) {
                    firstInvalid.scrollIntoView({ behavior: "smooth", block: "center" });
                    firstInvalid.focus();
                }
                return;
            }

            var btn = document.getElementById("predict-btn");
            if (btn) {
                btn.textContent = "Processing…";
                btn.disabled = true;
            }
        });

        form.querySelectorAll("input, select").forEach(function (el) {
            el.addEventListener("input", function () {
                el.style.borderColor = "";
            });
            el.addEventListener("change", function () {
                el.style.borderColor = "";
            });
        });
    }
});
