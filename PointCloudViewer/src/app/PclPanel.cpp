#include "app/PclPanel.h"

void PclPanel::Draw(const AlgoToolsHost& host) {
    AlgoToolsHost h = host;
    h.backend = AlgorithmBackend::Native;
    h.planeDistThresh = nullptr;
    h.planeMaxIter = nullptr;
    bool vis = visible_;
    AlgoToolsPanel::Draw(vis, u8"PclPanel（自研算法）", h);
    visible_ = vis;
}
