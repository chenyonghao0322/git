#include "tools/UndoHistory.h"

void UndoHistory::Clear() {
    undo_.clear();
    redo_.clear();
}

void UndoHistory::Push(const CloudSnapshot& snap) {
    undo_.push_back(snap);
    if (undo_.size() > kMaxDepth) {
        undo_.erase(undo_.begin());
    }
    redo_.clear();
}

bool UndoHistory::PopUndo(CloudSnapshot& outRestore) {
    if (undo_.empty()) return false;
    outRestore = std::move(undo_.back());
    undo_.pop_back();
    return true;
}

void UndoHistory::PushRedo(CloudSnapshot snap) {
    redo_.push_back(std::move(snap));
}

bool UndoHistory::PopRedo(CloudSnapshot& outRestore) {
    if (redo_.empty()) return false;
    outRestore = std::move(redo_.back());
    redo_.pop_back();
    return true;
}

void UndoHistory::PushUndo(CloudSnapshot snap) {
    undo_.push_back(std::move(snap));
}

const std::string& UndoHistory::LastUndoLabel() const {
    static const std::string kEmpty;
    if (undo_.empty()) return kEmpty;
    return undo_.back().label;
}
