#pragma once

#include "core/MathTypes.h"

class Camera {
public:
    void SetTarget(const Vec3& target, float distance);
    // 只改旋转中心，保持距离与视角（用于双击设中心）
    void SetOrbitTarget(const Vec3& target);
    void Orbit(float dx, float dy);
    void Pan(float dx, float dy, float sensitivity = 1.f);
    void Zoom(float delta);
    void Reset();
    void SetYawPitch(float yaw, float pitch);
    float Yaw() const { return yaw_; }
    float Pitch() const { return pitch_; }

    Mat4 ViewMatrix() const;
    Mat4 ProjMatrix(float aspect) const;
    Vec3 Eye() const;
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
