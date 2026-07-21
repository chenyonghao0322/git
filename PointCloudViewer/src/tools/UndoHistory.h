#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Snapshot of point visibility for undo/redo.
struct MaskSnapshot {
    std::vector<uint8_t> mask;
    std::string label;
};

class UndoHistory {
public:
    void Clear();
    void Push(const std::vector<uint8_t>& mask, const std::string& label);
    bool CanUndo() const { return !undo_.empty(); }
    bool CanRedo() const { return !redo_.empty(); }
    // Apply undo: currentMask is replaced; returns label of restored state.
    bool Undo(std::vector<uint8_t>& currentMask, std::string& outLabel);
    bool Redo(std::vector<uint8_t>& currentMask, std::string& outLabel);
    const std::string& LastUndoLabel() const;

private:
    static constexpr std::size_t kMaxDepth = 30;
    std::vector<MaskSnapshot> undo_;
    std::vector<MaskSnapshot> redo_;
};
