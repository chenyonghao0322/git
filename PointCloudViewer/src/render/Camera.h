#pragma once

#include "core/MathTypes.h"

class Camera {
public:
    void SetTarget(const Vec3& target, float distance);  // 设置观察目标与距离
    void SetOrbitTarget(const Vec3& target);           // 只改旋转中心（双击设中心）
    void Orbit(float dx, float dy);                    // 鼠标拖拽旋转
    void Pan(float dx, float dy, float sensitivity = 1.f);  // 平移
    void Zoom(float delta);                            // 滚轮缩放
    void Reset();                                    // 恢复初始视角
    void SetYawPitch(float yaw, float pitch);
    float Yaw() const { return yaw_; }
    float Pitch() const { return pitch_; }

    Mat4 ViewMatrix() const;
    Mat4 ProjMatrix(float aspect) const;
    Vec3 Eye() const;              // 相机世界坐标
    Vec3 Target() const { return target_; }
    float Distance() const { return distance_; }

private:
    Vec3 target_{0, 0, 0};
    float distance_ = 2.f;
    float yaw_ = 0.6f;
    float pitch_ = 0.5f;
    float fovy_ = 0.8f;
    Vec3 initialTarget_{0, 0, 0};
    float initialDistance_ = 2.f;
};
