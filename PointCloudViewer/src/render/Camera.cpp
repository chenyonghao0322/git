#include "render/Camera.h"

#include <algorithm>
#include <cmath>

void Camera::SetTarget(const Vec3& target, float distance) {
    target_ = target;
    distance_ = std::max(distance, 1e-3f);
    initialTarget_ = target_;
    initialDistance_ = distance_;
}

void Camera::SetOrbitTarget(const Vec3& target) {
    target_ = target;
}

void Camera::Orbit(float dx, float dy) {
    yaw_ += dx;
    pitch_ = std::clamp(pitch_ + dy, -1.55f, 1.55f);
}

void Camera::Pan(float dx, float dy, float sensitivity) {
    const Vec3 eye = Eye();
    Vec3 forward = (target_ - eye);
    if (forward.LengthSq() < 1e-12f) forward = {0, 0, -1};
    forward = forward.Normalized();

    Vec3 worldUp{0, 1, 0};
    Vec3 right = forward.Cross(worldUp);
    if (right.LengthSq() < 1e-10f) {
        worldUp = {0, 0, 1};
        right = forward.Cross(worldUp);
    }
    right = right.Normalized();
    const Vec3 up = right.Cross(forward).Normalized();
    const float scale = distance_ * 0.0018f * std::max(sensitivity, 0.05f);
    target_ = target_ + right * (-dx * scale) + up * (dy * scale);
}

void Camera::Zoom(float delta) {
    distance_ = std::clamp(distance_ * (1.f - delta * 0.1f), 1e-4f, 1e8f);
}

void Camera::Reset() {
    target_ = initialTarget_;
    distance_ = initialDistance_;
    yaw_ = 0.8f;
    pitch_ = 0.9f;
}

void Camera::SetYawPitch(float yaw, float pitch) {
    yaw_ = yaw;
    pitch_ = std::clamp(pitch, -1.55f, 1.55f);
}

Vec3 Camera::Eye() const {
    const float cp = std::cos(pitch_);
    const float sp = std::sin(pitch_);
    const float cy = std::cos(yaw_);
    const float sy = std::sin(yaw_);
    const Vec3 offset{distance_ * cp * cy, distance_ * sp, distance_ * cp * sy};
    return target_ + offset;
}

Mat4 Camera::ViewMatrix() const {
    Vec3 up{0, 1, 0};
    const Vec3 eye = Eye();
    const Vec3 forward = (target_ - eye).Normalized();
    if (std::fabs(forward.Dot(up)) > 0.999f) {
        up = {0, 0, 1};
    }
    return Mat4::LookAt(eye, target_, up);
}

Mat4 Camera::ProjMatrix(float aspect) const {
    // Wide near/far to avoid clipping large or flat industrial clouds.
    const float zNear = std::max(distance_ * 1e-4f, 1e-4f);
    const float zFar = std::max(distance_ * 5000.f, zNear * 1000.f);
    return Mat4::Perspective(fovy_, std::max(aspect, 1e-3f), zNear, zFar);
}
