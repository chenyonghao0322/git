#include "tools/UndoHistory.h"

void UndoHistory::Clear() {
    undo_.clear();
    redo_.clear();
}

void UndoHistory::Push(const std::vector<uint8_t>& mask, const std::string& label) {
    undo_.push_back({mask, label});
    if (undo_.size() > kMaxDepth) {
        undo_.erase(undo_.begin());
    }
    redo_.clear();
}

bool UndoHistory::Undo(std::vector<uint8_t>& currentMask, std::string& outLabel) {
    if (undo_.empty()) return false;
    redo_.push_back({currentMask, "redo-temp"});
    MaskSnapshot snap = std::move(undo_.back());
    undo_.pop_back();
    currentMask = std::move(snap.mask);
    outLabel = snap.label;
    return true;
}

bool UndoHistory::Redo(std::vector<uint8_t>& currentMask, std::string& outLabel) {
    if (redo_.empty()) return false;
    undo_.push_back({currentMask, "undo-temp"});
    MaskSnapshot snap = std::move(redo_.back());
    redo_.pop_back();
    currentMask = std::move(snap.mask);
    outLabel = snap.label;
    return true;
}

const std::string& UndoHistory::LastUndoLabel() const {
    static const std::string kEmpty;
    if (undo_.empty()) return kEmpty;
    return undo_.back().label;
}
