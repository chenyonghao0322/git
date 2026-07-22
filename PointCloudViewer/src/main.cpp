// 程序入口：初始化 GLFW/OpenGL/ImGui，进入 Application 主循环。
#include "app/Application.h"

int main() {
    Application app;
    if (!app.Init()) {
        return 1;
    }
    app.Run();
    app.Shutdown();
    return 0;
}
