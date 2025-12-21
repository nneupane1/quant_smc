// theme_loader.js
// Central theme controller. Each theme exports a ThemeConfig object.

import * as DarkTheme from "./dark.js";
import * as BloombergTheme from "./bloomberg.js";
import * as MinimalTheme from "./minimal.js";
import * as HighContrastTheme from "./high_contrast.js";

const THEMES = {
    dark: DarkTheme.ThemeConfig,
    bloomberg: BloombergTheme.ThemeConfig,
    minimal: MinimalTheme.ThemeConfig,
    high_contrast: HighContrastTheme.ThemeConfig
};

export function loadTheme(name) {
    const theme = THEMES[name] || THEMES["dark"];
    console.log(`[Theme] Loaded theme: ${name}`);
    return theme;
}
