#pragma once

// UndoHistory — 点云编辑的撤销/重做栈
//
// 设计要点：
// - 以「操作前快照」入栈：Push 保存的是即将被修改的状态，Undo 时恢复该快照。
// - 快照区分主点云 (Main) 与填充点云 (Filled)，避免填充视图下误操作主云掩码。
// - 可选保存主点云坐标 (points)，用于平面投影等会改几何的操作。
// - 新操作会清空 redo 栈；栈深上限 kMaxDepth。

#include "core/MathTypes.h"

#include <cstdint>
#include <string>
#include <vector>

// 快照所对应的目标点云
enum class HistoryCloudTarget {
    Main,    // cloud_：加载的原始点云，ROI 挖孔/滤波/剖切等主要编辑对象
    Filled,  // filledCloud_：执行填充后生成的补点云（仅掩码可编辑）
};

// 单次可恢复的状态
struct CloudSnapshot {
    HistoryCloudTarget target = HistoryCloudTarget::Main;
    std::vector<uint8_t> mask;   // 可见性掩码，与对应点云 points 等长
    std::vector<Vec3> points;    // 非空时表示同时恢复主点云坐标（如平面投影撤销）
    std::string label;           // 操作名称，用于状态栏提示
};

class UndoHistory {
public:
    void Clear();
    // 记录一次操作前的快照；会清空重做栈
    void Push(const CloudSnapshot& snap);
    bool CanUndo() const { return !undo_.empty(); }
    bool CanRedo() const { return !redo_.empty(); }
    // 弹出待恢复的快照（由 Application 负责写回点云并构造 redo 快照）
    bool PopUndo(CloudSnapshot& outRestore);
    void PushRedo(CloudSnapshot snap);
    bool PopRedo(CloudSnapshot& outRestore);
    void PushUndo(CloudSnapshot snap);
    const std::string& LastUndoLabel() const;

private:
    static constexpr std::size_t kMaxDepth = 30;
    std::vector<CloudSnapshot> undo_;
    std::vector<CloudSnapshot> redo_;
};
