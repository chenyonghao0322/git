#pragma once

#include "app/AlgoToolsPanel.h"

class PclPanel {
public:
    void Draw(const AlgoToolsHost& host);
    void SetVisible(bool v) { visible_ = v; }
    bool IsVisible() const { return visible_; }
    void ToggleVisible() { visible_ = !visible_; }

    float& PlaneDistThresh() { return planeDistThresh_; }
    int& PlaneMaxIter() { return planeMaxIter_; }

private:
    bool visible_ = false;
    float planeDistThresh_ = 0.05f;
    int planeMaxIter_ = 1000;
};
