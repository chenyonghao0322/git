#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

struct Vec3 {
    float x = 0.f, y = 0.f, z = 0.f;

    Vec3() = default;
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(float s) const { return {x / s, y / s, z / s}; }
    Vec3& operator+=(const Vec3& o) {
        x += o.x;
        y += o.y;
        z += o.z;
        return *this;
    }

    float Dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 Cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    float Length() const { return std::sqrt(Dot(*this)); }
    float LengthSq() const { return Dot(*this); }
    Vec3 Normalized() const {
        const float len = Length();
        if (len < 1e-12f) return {0, 0, 0};
        return (*this) / len;
    }
};

inline Vec3 operator*(float s, const Vec3& v) { return v * s; }

struct Vec2 {
    float x = 0.f, y = 0.f;
    Vec2() = default;
    Vec2(float x_, float y_) : x(x_), y(y_) {}
    Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
    float Length() const { return std::sqrt(x * x + y * y); }
};

struct Vec4 {
    float x = 0.f, y = 0.f, z = 0.f, w = 0.f;
};

struct Mat4 {
    float m[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };

    static Mat4 Identity() { return {}; }

    static Mat4 Perspective(float fovyRad, float aspect, float zNear, float zFar) {
        Mat4 r{};
        const float f = 1.f / std::tan(fovyRad * 0.5f);
        r.m[0] = f / aspect;
        r.m[5] = f;
        r.m[10] = (zFar + zNear) / (zNear - zFar);
        r.m[11] = -1.f;
        r.m[14] = (2.f * zFar * zNear) / (zNear - zFar);
        r.m[15] = 0.f;
        return r;
    }

    static Mat4 LookAt(const Vec3& eye, const Vec3& center, const Vec3& up) {
        const Vec3 f = (center - eye).Normalized();
        const Vec3 s = f.Cross(up).Normalized();
        const Vec3 u = s.Cross(f);
        Mat4 r{};
        r.m[0] = s.x;
        r.m[4] = s.y;
        r.m[8] = s.z;
        r.m[1] = u.x;
        r.m[5] = u.y;
        r.m[9] = u.z;
        r.m[2] = -f.x;
        r.m[6] = -f.y;
        r.m[10] = -f.z;
        r.m[12] = -s.Dot(eye);
        r.m[13] = -u.Dot(eye);
        r.m[14] = f.Dot(eye);
        return r;
    }

    Mat4 operator*(const Mat4& o) const {
        Mat4 r{};
        for (int col = 0; col < 4; ++col) {
            for (int row = 0; row < 4; ++row) {
                r.m[col * 4 + row] =
                    m[0 * 4 + row] * o.m[col * 4 + 0] +
                    m[1 * 4 + row] * o.m[col * 4 + 1] +
                    m[2 * 4 + row] * o.m[col * 4 + 2] +
                    m[3 * 4 + row] * o.m[col * 4 + 3];
            }
        }
        return r;
    }

    Vec4 MulVec4(const Vec4& v) const {
        return {
            m[0] * v.x + m[4] * v.y + m[8] * v.z + m[12] * v.w,
            m[1] * v.x + m[5] * v.y + m[9] * v.z + m[13] * v.w,
            m[2] * v.x + m[6] * v.y + m[10] * v.z + m[14] * v.w,
            m[3] * v.x + m[7] * v.y + m[11] * v.z + m[15] * v.w
        };
    }

    Vec3 TransformPoint(const Vec3& p) const {
        const Vec4 r = MulVec4({p.x, p.y, p.z, 1.f});
        if (std::fabs(r.w) < 1e-12f) return {r.x, r.y, r.z};
        return {r.x / r.w, r.y / r.w, r.z / r.w};
    }

    Vec3 TransformVector(const Vec3& v) const {
        return {
            m[0] * v.x + m[4] * v.y + m[8] * v.z,
            m[1] * v.x + m[5] * v.y + m[9] * v.z,
            m[2] * v.x + m[6] * v.y + m[10] * v.z,
        };
    }
};

// 绕单位轴 axis 旋转 angleRad（Rodrigues）
inline Mat4 RotationAxisAngle(const Vec3& axisUnit, float angleRad) {
    const float x = axisUnit.x;
    const float y = axisUnit.y;
    const float z = axisUnit.z;
    const float c = std::cos(angleRad);
    const float s = std::sin(angleRad);
    const float t = 1.f - c;
    Mat4 r{};
    r.m[0] = t * x * x + c;
    r.m[4] = t * x * y - s * z;
    r.m[8] = t * x * z + s * y;
    r.m[1] = t * x * y + s * z;
    r.m[5] = t * y * y + c;
    r.m[9] = t * y * z - s * x;
    r.m[2] = t * x * z - s * y;
    r.m[6] = t * y * z + s * x;
    r.m[10] = t * z * z + c;
    r.m[15] = 1.f;
    return r;
}

// 将 fromDir 旋转到 toDir 的最小旋转矩阵（输入方向会自动归一化）
inline Mat4 RotationFromTo(Vec3 fromDir, Vec3 toDir) {
    fromDir = fromDir.Normalized();
    toDir = toDir.Normalized();
    float c = fromDir.Dot(toDir);
    c = std::clamp(c, -1.f, 1.f);
    if (c > 0.9999f) return Mat4::Identity();

    Vec3 axis = fromDir.Cross(toDir);
    const float s = axis.Length();
    if (s < 1e-8f) {
        Vec3 perp = (std::fabs(fromDir.x) < 0.9f) ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
        axis = fromDir.Cross(perp).Normalized();
        return RotationAxisAngle(axis, 3.14159265f);
    }
    axis = axis / s;
    const float angle = std::atan2(s, c);
    return RotationAxisAngle(axis, angle);
}

struct Aabb {
    Vec3 min{1e30f, 1e30f, 1e30f};
    Vec3 max{-1e30f, -1e30f, -1e30f};

    void Expand(const Vec3& p) {
        min.x = std::min(min.x, p.x);
        min.y = std::min(min.y, p.y);
        min.z = std::min(min.z, p.z);
        max.x = std::max(max.x, p.x);
        max.y = std::max(max.y, p.y);
        max.z = std::max(max.z, p.z);
    }

    Vec3 Center() const { return (min + max) * 0.5f; }
    Vec3 Extent() const { return max - min; }
    float Diagonal() const { return Extent().Length(); }
    bool Valid() const { return min.x <= max.x; }
};

inline Vec3 HeightToColor(float t) {
    // Simple jet-like ramp: blue -> cyan -> green -> yellow -> red
    t = std::clamp(t, 0.f, 1.f);
    if (t < 0.25f) {
        const float u = t / 0.25f;
        return {0.f, u, 1.f};
    }
    if (t < 0.5f) {
        const float u = (t - 0.25f) / 0.25f;
        return {0.f, 1.f, 1.f - u};
    }
    if (t < 0.75f) {
        const float u = (t - 0.5f) / 0.25f;
        return {u, 1.f, 0.f};
    }
    const float u = (t - 0.75f) / 0.25f;
    return {1.f, 1.f - u, 0.f};
}

// Diverging map for signed scalars, t in [0,1] where 0.5 ≈ 0
inline Vec3 DivergingColor(float t) {
    t = std::clamp(t, 0.f, 1.f);
    if (t < 0.5f) {
        const float u = t / 0.5f;
        return {u, u, 1.f};  // blue -> white
    }
    const float u = (t - 0.5f) / 0.5f;
    return {1.f, 1.f - u, 1.f - u};  // white -> red
}

