#pragma once

// 点云算法后端：自研实现 或 PCL（Point Cloud Library，点云库）
enum class AlgorithmBackend {
    Native = 0,
    PCL = 1,
};

inline const char* AlgorithmBackendLabel(AlgorithmBackend b) {
    switch (b) {
        case AlgorithmBackend::Native:
            return u8"自研算法";
        case AlgorithmBackend::PCL:
            return u8"PCL 算法";
        default:
            return u8"未知";
    }
}
