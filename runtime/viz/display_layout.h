/*
 * viz/display_layout.h — 5.5 寸竖屏等场景的预览窗口布局参数
 */
#pragma once

struct DisplayWindowConfig {
    bool enabled = true;
    int screen_width = 1080;
    int screen_height = 1920;
    float max_screen_ratio = 0.85f;
    bool fullscreen = false;
    int title_bar_reserve_px = 56;  // 顶部留白，避免挡住窗口标题栏按钮
};
