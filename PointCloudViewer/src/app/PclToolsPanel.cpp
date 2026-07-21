#include "app/PclToolsPanel.h"

void PclToolsPanel::Draw(const AlgoToolsHost& host) {
    AlgoToolsHost h = host;
    h.backend = AlgorithmBackend::PCL;
    h.planeDistThresh = &planeDistThresh_;
    h.planeMaxIter = &planeMaxIter_;
    bool vis = visible_;
    AlgoToolsPanel::Draw(vis, u8"PclTools（PCL 算法）", h);
    visible_ = vis;
}
