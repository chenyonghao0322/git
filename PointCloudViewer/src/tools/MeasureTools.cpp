#include "tools/MeasureTools.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

bool ProjectToScreen(const Vec3& p, const Mat4& mvp, int fbW, int fbH, float& sx, float& sy,
                     float& depth) {
    const Vec4 clip = mvp.MulVec4({p.x, p.y, p.z, 1.f});
    if (std::fabs(clip.w) < 1e-12f) return false;
    const float ndcX = clip.x / clip.w;
    const float ndcY = clip.y / clip.w;
    const float ndcZ = clip.z / clip.w;
    if (ndcZ < -1.f || ndcZ > 1.f) return false;
    sx = (ndcX * 0.5f + 0.5f) * static_cast<float>(fbW);
    sy = (1.f - (ndcY * 0.5f + 0.5f)) * static_cast<float>(fbH);  // top-left origin
    depth = ndcZ;
    return true;
}

// Jacobi eigen-decomposition for 3x3 symmetric matrix. Returns eigenvalues and eigenvectors
// (columns of V). Finds smallest eigenvalue eigenvector as plane normal.
void Jacobi3(const float A[3][3], float eig[3], float V[3][3]) {
    float a[3][3] = {
        {A[0][0], A[0][1], A[0][2]},
        {A[1][0], A[1][1], A[1][2]},
        {A[2][0], A[2][1], A[2][2]}
    };
    V[0][0] = 1;
    V[0][1] = 0;
    V[0][2] = 0;
    V[1][0] = 0;
    V[1][1] = 1;
    V[1][2] = 0;
    V[2][0] = 0;
    V[2][1] = 0;
    V[2][2] = 1;

    for (int iter = 0; iter < 32; ++iter) {
        int p = 0, q = 1;
        float maxAbs = std::fabs(a[0][1]);
        if (std::fabs(a[0][2]) > maxAbs) {
            maxAbs = std::fabs(a[0][2]);
            p = 0;
            q = 2;
        }
        if (std::fabs(a[1][2]) > maxAbs) {
            maxAbs = std::fabs(a[1][2]);
            p = 1;
            q = 2;
        }
        if (maxAbs < 1e-12f) break;

        const float app = a[p][p];
        const float aqq = a[q][q];
        const float apq = a[p][q];
        const float phi = 0.5f * std::atan2(2.f * apq, aqq - app);
        const float c = std::cos(phi);
        const float s = std::sin(phi);

        float Rp[3], Rq[3];
        for (int i = 0; i < 3; ++i) {
            Rp[i] = c * a[i][p] - s * a[i][q];
            Rq[i] = s * a[i][p] + c * a[i][q];
        }
        for (int i = 0; i < 3; ++i) {
            a[i][p] = a[p][i] = Rp[i];
            a[i][q] = a[q][i] = Rq[i];
        }
        a[p][p] = c * c * app - 2.f * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.f * s * c * apq + c * c * aqq;
        a[p][q] = a[q][p] = 0.f;

        for (int i = 0; i < 3; ++i) {
            const float vip = V[i][p];
            const float viq = V[i][q];
            V[i][p] = c * vip - s * viq;
            V[i][q] = s * vip + c * viq;
        }
    }

    eig[0] = a[0][0];
    eig[1] = a[1][1];
    eig[2] = a[2][2];
}

// Solve A x = b for square system (Gaussian elimination with partial pivoting). n <= 4.
bool SolveLinear(double* A, double* b, int n, double* x) {
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double best = std::fabs(A[col * n + col]);
        for (int row = col + 1; row < n; ++row) {
            const double v = std::fabs(A[row * n + col]);
            if (v > best) {
                best = v;
                pivot = row;
            }
        }
        if (best < 1e-14) return false;
        if (pivot != col) {
            for (int j = 0; j < n; ++j) std::swap(A[col * n + j], A[pivot * n + j]);
            std::swap(b[col], b[pivot]);
        }
        const double diag = A[col * n + col];
        for (int row = col + 1; row < n; ++row) {
            const double f = A[row * n + col] / diag;
            b[row] -= f * b[col];
            for (int j = col; j < n; ++j) A[row * n + j] -= f * A[col * n + j];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < n; ++j) s -= A[i * n + j] * x[j];
        const double diag = A[i * n + i];
        if (std::fabs(diag) < 1e-14) return false;
        x[i] = s / diag;
    }
    return true;
}

void OrthonormalBasis(const Vec3& nIn, Vec3& u, Vec3& v) {
    const Vec3 n = nIn.Normalized();
    const Vec3 tmp = (std::fabs(n.x) < 0.9f) ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    u = n.Cross(tmp).Normalized();
    v = n.Cross(u).Normalized();
}

std::vector<std::size_t> CollectIndices(const PointCloud& cloud,
                                        const std::vector<std::size_t>& indices) {
    if (!indices.empty()) return indices;
    std::vector<std::size_t> use;
    use.reserve(cloud.points.size());
    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        if (cloud.mask.empty() || cloud.mask[i]) use.push_back(i);
    }
    return use;
}

bool FitCircle2D(const std::vector<float>& xs, const std::vector<float>& ys, float& cx, float& cy,
                 float& radius, float& rms) {
    const int n = static_cast<int>(xs.size());
    if (n < 3 || ys.size() != xs.size()) return false;

    double ATA[9] = {};
    double ATb[3] = {};
    for (int i = 0; i < n; ++i) {
        const double x = xs[i];
        const double y = ys[i];
        const double rhs = x * x + y * y;
        // columns: x, y, 1
        const double row[3] = {x, y, 1.0};
        for (int r = 0; r < 3; ++r) {
            ATb[r] += row[r] * rhs;
            for (int c = 0; c < 3; ++c) ATA[r * 3 + c] += row[r] * row[c];
        }
    }
    double sol[3] = {};
    if (!SolveLinear(ATA, ATb, 3, sol)) return false;
    cx = static_cast<float>(0.5 * sol[0]);
    cy = static_cast<float>(0.5 * sol[1]);
    const double r2 = sol[2] + static_cast<double>(cx) * cx + static_cast<double>(cy) * cy;
    if (r2 <= 1e-12) return false;
    radius = static_cast<float>(std::sqrt(r2));

    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        const float dx = xs[i] - cx;
        const float dy = ys[i] - cy;
        const float ri = std::sqrt(dx * dx + dy * dy);
        const float e = ri - radius;
        acc += static_cast<double>(e) * e;
    }
    rms = static_cast<float>(std::sqrt(acc / static_cast<double>(n)));
    return true;
}

}  // namespace

namespace MeasureTools {

std::optional<std::size_t> PickNearest(const PointCloud& cloud, const Camera& camera, int fbW,
                                       int fbH, float mouseX, float mouseY, float maxPixelDist,
                                       const std::vector<std::size_t>* onlyIndices) {
    if (cloud.points.empty() || fbW <= 0 || fbH <= 0) return std::nullopt;

    const float aspect = static_cast<float>(fbW) / static_cast<float>(fbH);
    const Mat4 mvp = camera.ProjMatrix(aspect) * camera.ViewMatrix();

    float bestScore = maxPixelDist * maxPixelDist;
    std::optional<std::size_t> best;
    float bestDepth = 1e30f;

    auto consider = [&](std::size_t i) {
        if (!cloud.mask.empty() && !cloud.mask[i]) return;
        float sx, sy, depth;
        if (!ProjectToScreen(cloud.points[i], mvp, fbW, fbH, sx, sy, depth)) return;
        const float dx = sx - mouseX;
        const float dy = sy - mouseY;
        const float d2 = dx * dx + dy * dy;
        if (d2 <= bestScore + 1e-6f) {
            if (d2 < bestScore - 1e-6f || depth < bestDepth) {
                bestScore = d2;
                bestDepth = depth;
                best = i;
            }
        }
    };

    if (onlyIndices && !onlyIndices->empty()) {
        for (std::size_t i : *onlyIndices) {
            if (i < cloud.points.size()) consider(i);
        }
    } else {
        for (std::size_t i = 0; i < cloud.points.size(); ++i) consider(i);
    }
    return best;
}

bool FitPlaneSVD(const PointCloud& cloud, const std::vector<std::size_t>& indices, PlaneModel& out,
                 std::string& error) {
    std::vector<std::size_t> use = indices;
    if (use.empty()) {
        use.reserve(cloud.points.size());
        for (std::size_t i = 0; i < cloud.points.size(); ++i) {
            if (cloud.mask.empty() || cloud.mask[i]) use.push_back(i);
        }
    }
    if (use.size() < 3) {
        error = "至少需要 3 个点才能拟合平面。";
        return false;
    }

    Vec3 c{0, 0, 0};
    for (std::size_t idx : use) c += cloud.points[idx];
    c = c / static_cast<float>(use.size());

    float cov[3][3] = {};
    for (std::size_t idx : use) {
        const Vec3 d = cloud.points[idx] - c;
        cov[0][0] += d.x * d.x;
        cov[0][1] += d.x * d.y;
        cov[0][2] += d.x * d.z;
        cov[1][1] += d.y * d.y;
        cov[1][2] += d.y * d.z;
        cov[2][2] += d.z * d.z;
    }
    cov[1][0] = cov[0][1];
    cov[2][0] = cov[0][2];
    cov[2][1] = cov[1][2];

    float eig[3];
    float V[3][3];
    Jacobi3(cov, eig, V);

    int minAxis = 0;
    if (eig[1] < eig[minAxis]) minAxis = 1;
    if (eig[2] < eig[minAxis]) minAxis = 2;

    Vec3 n{V[0][minAxis], V[1][minAxis], V[2][minAxis]};
    n = n.Normalized();
    if (n.z < 0.f) n = n * -1.f;

    double rmsAcc = 0.0;
    Vec3 tmp = (std::fabs(n.x) < 0.9f) ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    Vec3 uAxis = n.Cross(tmp).Normalized();
    Vec3 vAxis = n.Cross(uAxis).Normalized();
    float maxU = 0.f;
    float maxV = 0.f;
    for (std::size_t idx : use) {
        const Vec3 d = cloud.points[idx] - c;
        const float dist = d.Dot(n);
        rmsAcc += static_cast<double>(dist) * static_cast<double>(dist);
        const Vec3 onPlane = d - n * dist;
        maxU = std::max(maxU, std::fabs(onPlane.Dot(uAxis)));
        maxV = std::max(maxV, std::fabs(onPlane.Dot(vAxis)));
    }

    out.centroid = c;
    out.normal = n;
    out.rms = static_cast<float>(std::sqrt(rmsAcc / static_cast<double>(use.size())));
    out.pointCount = static_cast<int>(use.size());
    // 用点集在平面上的轴对齐包络，避免“外接圆→正方形”比框选区域大一圈
    out.halfExtentU = std::max(maxU * 1.02f, 0.05f);
    out.halfExtentV = std::max(maxV * 1.02f, 0.05f);
    out.halfSize = std::max(out.halfExtentU, out.halfExtentV);
    return true;
}

bool FitSphere(const PointCloud& cloud, const std::vector<std::size_t>& indices, SphereModel& out,
               std::string& error) {
    const std::vector<std::size_t> use = CollectIndices(cloud, indices);
    if (use.size() < 4) {
        error = u8"至少需要 4 个点才能拟合球。";
        return false;
    }

    // Algebraic: x^2+y^2+z^2 = A x + B y + C z + D
    double ATA[16] = {};
    double ATb[4] = {};
    for (std::size_t idx : use) {
        const Vec3& p = cloud.points[idx];
        const double x = p.x, y = p.y, z = p.z;
        const double rhs = x * x + y * y + z * z;
        const double row[4] = {x, y, z, 1.0};
        for (int r = 0; r < 4; ++r) {
            ATb[r] += row[r] * rhs;
            for (int c = 0; c < 4; ++c) ATA[r * 4 + c] += row[r] * row[c];
        }
    }
    double sol[4] = {};
    if (!SolveLinear(ATA, ATb, 4, sol)) {
        error = u8"球面拟合线性方程组求解失败。";
        return false;
    }

    const float cx = static_cast<float>(0.5 * sol[0]);
    const float cy = static_cast<float>(0.5 * sol[1]);
    const float cz = static_cast<float>(0.5 * sol[2]);
    const double r2 = sol[3] + static_cast<double>(cx) * cx + static_cast<double>(cy) * cy +
                      static_cast<double>(cz) * cz;
    if (r2 <= 1e-12) {
        error = u8"拟合得到的球半径无效。";
        return false;
    }
    const float radius = static_cast<float>(std::sqrt(r2));
    const Vec3 center{cx, cy, cz};

    out.center = center;
    out.radius = radius;
    out.pointCount = static_cast<int>(use.size());
    out.inlierIndices.clear();
    out.inlierIndices.reserve(use.size());
    const float band = std::max(radius * 0.02f, 1e-4f);

    double acc = 0.0;
    for (std::size_t idx : use) {
        const float e = (cloud.points[idx] - center).Length() - radius;
        acc += static_cast<double>(e) * static_cast<double>(e);
        if (std::fabs(e) <= band) out.inlierIndices.push_back(idx);
    }

    out.rms = static_cast<float>(std::sqrt(acc / static_cast<double>(use.size())));
    return true;
}

bool FitCircleOnPlane(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                      const PlaneModel& plane, float inlierBand, CircleModel& out,
                      std::string& error) {
    const std::vector<std::size_t> use = CollectIndices(cloud, indices);
    if (use.size() < 3) {
        error = u8"至少需要 3 个点才能拟合圆。";
        return false;
    }

    Vec3 u, v;
    OrthonormalBasis(plane.normal, u, v);

    std::vector<float> xs, ys;
    xs.reserve(use.size());
    ys.reserve(use.size());
    for (std::size_t idx : use) {
        const Vec3 d = cloud.points[idx] - plane.centroid;
        xs.push_back(d.Dot(u));
        ys.push_back(d.Dot(v));
    }

    float cx2 = 0.f, cy2 = 0.f, radiusAlg = 0.f, circRms = 0.f;
    if (!FitCircle2D(xs, ys, cx2, cy2, radiusAlg, circRms)) {
        error = u8"平面内圆拟合失败。";
        return false;
    }

    std::vector<float> radii;
    radii.reserve(use.size());
    for (std::size_t i = 0; i < use.size(); ++i) {
        const float dx = xs[i] - cx2;
        const float dy = ys[i] - cy2;
        radii.push_back(std::sqrt(dx * dx + dy * dy));
    }
    std::sort(radii.begin(), radii.end());
    const std::size_t hiIdx =
        std::min(radii.size() - 1, static_cast<std::size_t>(radii.size() * 98 / 100));
    const float radiusHi = radii[hiIdx];
    float radius = radiusAlg;
    if (radiusHi > radiusAlg * 1.08f) radius = radiusHi;

    const float band = inlierBand > 0.f ? inlierBand : std::max(radius * 0.02f, 1e-4f);
    out.inlierIndices.clear();
    out.inlierIndices.reserve(use.size());
    double circAcc = 0.0;
    for (std::size_t i = 0; i < use.size(); ++i) {
        const float e = radii[i] - radius;
        circAcc += static_cast<double>(e) * static_cast<double>(e);
        if (std::fabs(e) <= band) out.inlierIndices.push_back(use[i]);
    }

    double planeAcc = 0.0;
    for (std::size_t idx : use) {
        const float d = (cloud.points[idx] - plane.centroid).Dot(plane.normal);
        planeAcc += static_cast<double>(d) * static_cast<double>(d);
    }
    const float planeRms =
        static_cast<float>(std::sqrt(planeAcc / static_cast<double>(use.size())));
    const float circRmsFinal =
        static_cast<float>(std::sqrt(circAcc / static_cast<double>(use.size())));

    out.center = plane.centroid + u * cx2 + v * cy2;
    out.normal = plane.normal;
    out.radius = radius;
    out.rms = std::sqrt(circRmsFinal * circRmsFinal + planeRms * planeRms);
    out.pointCount = static_cast<int>(use.size());
    return true;
}

bool FitCircle3D(const PointCloud& cloud, const std::vector<std::size_t>& indices, CircleModel& out,
                 std::string& error) {
    const std::vector<std::size_t> use = CollectIndices(cloud, indices);
    if (use.size() < 3) {
        error = u8"至少需要 3 个点才能拟合圆。";
        return false;
    }

    PlaneModel plane;
    if (!FitPlaneSVD(cloud, use, plane, error)) return false;
    return FitCircleOnPlane(cloud, use, plane, 0.f, out, error);
}

bool FitCylinder(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                 CylinderModel& out, std::string& error) {
    const std::vector<std::size_t> use = CollectIndices(cloud, indices);
    if (use.size() < 6) {
        error = u8"至少需要 6 个点才能拟合圆柱。";
        return false;
    }

    Vec3 c{0, 0, 0};
    for (std::size_t idx : use) c += cloud.points[idx];
    c = c / static_cast<float>(use.size());

    float cov[3][3] = {};
    for (std::size_t idx : use) {
        const Vec3 d = cloud.points[idx] - c;
        cov[0][0] += d.x * d.x;
        cov[0][1] += d.x * d.y;
        cov[0][2] += d.x * d.z;
        cov[1][1] += d.y * d.y;
        cov[1][2] += d.y * d.z;
        cov[2][2] += d.z * d.z;
    }
    cov[1][0] = cov[0][1];
    cov[2][0] = cov[0][2];
    cov[2][1] = cov[1][2];

    float eig[3];
    float V[3][3];
    Jacobi3(cov, eig, V);

    Vec3 axes[3] = {{V[0][0], V[1][0], V[2][0]},
                    {V[0][1], V[1][1], V[2][1]},
                    {V[0][2], V[1][2], V[2][2]}};

    bool found = false;
    CylinderModel best;
    best.rms = 1e30f;

    for (int ai = 0; ai < 3; ++ai) {
        Vec3 axis = axes[ai].Normalized();
        if (axis.Length() < 1e-6f) continue;
        Vec3 u, v;
        OrthonormalBasis(axis, u, v);

        std::vector<float> xs, ys;
        xs.reserve(use.size());
        ys.reserve(use.size());
        float tMin = 1e30f, tMax = -1e30f;
        for (std::size_t idx : use) {
            const Vec3 d = cloud.points[idx] - c;
            xs.push_back(d.Dot(u));
            ys.push_back(d.Dot(v));
            const float t = d.Dot(axis);
            tMin = std::min(tMin, t);
            tMax = std::max(tMax, t);
        }

        float cx2 = 0.f, cy2 = 0.f, radius = 0.f, circRms = 0.f;
        if (!FitCircle2D(xs, ys, cx2, cy2, radius, circRms)) continue;
        if (radius < 1e-6f) continue;

        const Vec3 axisPoint = c + u * cx2 + v * cy2;
        double acc = 0.0;
        for (std::size_t idx : use) {
            const Vec3 d = cloud.points[idx] - axisPoint;
            const Vec3 radial = d - axis * d.Dot(axis);
            const float e = radial.Length() - radius;
            acc += static_cast<double>(e) * e;
        }
        const float rms = static_cast<float>(std::sqrt(acc / static_cast<double>(use.size())));
        if (rms < best.rms) {
            best.axisPoint = axisPoint;
            best.axisDir = axis;
            best.radius = radius;
            best.halfHeight = std::max(0.5f * (tMax - tMin) * 1.05f, radius * 0.25f);
            best.rms = rms;
            best.pointCount = static_cast<int>(use.size());
            found = true;
        }
    }

    if (!found) {
        error = u8"圆柱拟合失败，请检查点是否近似柱面。";
        return false;
    }
    out = best;
    return true;
}

void SelectRoi(const PointCloud& cloud, const Camera& camera, int fbW, int fbH, float x0, float y0,
               float x1, float y1, std::vector<std::size_t>& outIndices) {
    outIndices.clear();
    if (cloud.points.empty() || fbW <= 0 || fbH <= 0) return;

    const float minX = std::min(x0, x1);
    const float maxX = std::max(x0, x1);
    const float minY = std::min(y0, y1);
    const float maxY = std::max(y0, y1);
    if (maxX - minX < 1.f || maxY - minY < 1.f) return;

    const float aspect = static_cast<float>(fbW) / static_cast<float>(fbH);
    const Mat4 mvp = camera.ProjMatrix(aspect) * camera.ViewMatrix();
    const Vec3 eye = camera.Eye();

    constexpr float cell = 1.5f;
    const int gridW = std::max(1, static_cast<int>(std::ceil((maxX - minX) / cell)) + 1);
    const int gridH = std::max(1, static_cast<int>(std::ceil((maxY - minY) / cell)) + 1);

    struct Cand {
        std::size_t idx;
        int ix;
        int iy;
        float depth;
    };
    std::vector<Cand> cands;
    cands.reserve(65536);

    std::vector<float> zbuf(static_cast<std::size_t>(gridW * gridH),
                            std::numeric_limits<float>::max());

    auto cellIndex = [&](float sx, float sy, int& ix, int& iy) -> bool {
        ix = static_cast<int>((sx - minX) / cell);
        iy = static_cast<int>((sy - minY) / cell);
        if (ix < 0 || iy < 0 || ix >= gridW || iy >= gridH) return false;
        return true;
    };

    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        if (!cloud.mask.empty() && !cloud.mask[i]) continue;
        float sx, sy, ndcZ;
        if (!ProjectToScreen(cloud.points[i], mvp, fbW, fbH, sx, sy, ndcZ)) continue;
        if (sx < minX || sx > maxX || sy < minY || sy > maxY) continue;

        int ix = 0, iy = 0;
        if (!cellIndex(sx, sy, ix, iy)) continue;

        const float depth = (cloud.points[i] - eye).Length();
        const std::size_t gi = static_cast<std::size_t>(iy * gridW + ix);
        if (depth < zbuf[gi]) zbuf[gi] = depth;
        cands.push_back({i, ix, iy, depth});
    }

    if (cands.empty()) return;

    float depthTol = camera.Distance() * 0.003f;
    if (cloud.bounds.Valid()) {
        depthTol = std::max(depthTol, cloud.bounds.Diagonal() * 0.0008f);
    }
    depthTol = std::max(depthTol, 1e-4f);

    outIndices.reserve(cands.size() / 2 + 8);
    for (const Cand& c : cands) {
        float front = zbuf[static_cast<std::size_t>(c.iy * gridW + c.ix)];
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int nx = c.ix + dx;
                const int ny = c.iy + dy;
                if (nx < 0 || ny < 0 || nx >= gridW || ny >= gridH) continue;
                front = std::min(front, zbuf[static_cast<std::size_t>(ny * gridW + nx)]);
            }
        }
        if (c.depth <= front + depthTol) outIndices.push_back(c.idx);
    }
}

void SelectRoiCircle(const PointCloud& cloud, const Camera& camera, int fbW, int fbH, float cx,
                     float cy, float radiusPx, std::vector<std::size_t>& outIndices) {
    if (radiusPx < 1.f) return;
    SelectRoi(cloud, camera, fbW, fbH, cx - radiusPx, cy - radiusPx, cx + radiusPx, cy + radiusPx,
              outIndices);
    if (outIndices.empty()) return;
    const float r2 = radiusPx * radiusPx;
    std::vector<std::size_t> filtered;
    filtered.reserve(outIndices.size());
    const float aspect = static_cast<float>(fbW) / static_cast<float>(fbH);
    const Mat4 mvp = camera.ProjMatrix(aspect) * camera.ViewMatrix();
    for (std::size_t idx : outIndices) {
        float sx, sy, ndcZ;
        if (!ProjectToScreen(cloud.points[idx], mvp, fbW, fbH, sx, sy, ndcZ)) continue;
        const float dx = sx - cx;
        const float dy = sy - cy;
        if (dx * dx + dy * dy <= r2 + 1e-3f) filtered.push_back(idx);
    }
    outIndices = std::move(filtered);
}

bool PointInPolygon(float x, float y, const std::vector<float>& polyX,
                    const std::vector<float>& polyY) {
    const std::size_t n = polyX.size();
    if (n < 3 || polyY.size() != n) return false;
    bool inside = false;
    for (std::size_t i = 0, j = n - 1; i < n; j = i++) {
        const float xi = polyX[i], yi = polyY[i];
        const float xj = polyX[j], yj = polyY[j];
        const bool intersect =
            ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12f) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

void SelectRoiPolygon(const PointCloud& cloud, const Camera& camera, int fbW, int fbH,
                      const std::vector<float>& polyX, const std::vector<float>& polyY,
                      std::vector<std::size_t>& outIndices) {
    outIndices.clear();
    if (polyX.size() < 3 || polyY.size() != polyX.size()) return;
    float minX = polyX[0], maxX = polyX[0], minY = polyY[0], maxY = polyY[0];
    for (std::size_t i = 1; i < polyX.size(); ++i) {
        minX = std::min(minX, polyX[i]);
        maxX = std::max(maxX, polyX[i]);
        minY = std::min(minY, polyY[i]);
        maxY = std::max(maxY, polyY[i]);
    }
    SelectRoi(cloud, camera, fbW, fbH, minX, minY, maxX, maxY, outIndices);
    if (outIndices.empty()) return;
    const float aspect = static_cast<float>(fbW) / static_cast<float>(fbH);
    const Mat4 mvp = camera.ProjMatrix(aspect) * camera.ViewMatrix();
    std::vector<std::size_t> filtered;
    filtered.reserve(outIndices.size());
    for (std::size_t idx : outIndices) {
        float sx, sy, ndcZ;
        if (!ProjectToScreen(cloud.points[idx], mvp, fbW, fbH, sx, sy, ndcZ)) continue;
        if (PointInPolygon(sx, sy, polyX, polyY)) filtered.push_back(idx);
    }
    outIndices = std::move(filtered);
}

void SelectRoiWorldCircleXY(const PointCloud& cloud, const Camera& camera, int fbW, int fbH,
                            const Vec3& center, float radius,
                            std::vector<std::size_t>& outIndices) {
    outIndices.clear();
    if (cloud.points.empty() || radius <= 0.f) return;
    const float r2 = radius * radius;
    const float aspect = static_cast<float>(fbW) / static_cast<float>(fbH);
    const Mat4 mvp = camera.ProjMatrix(aspect) * camera.ViewMatrix();
    const Vec3 eye = camera.Eye();

    std::vector<std::size_t> candidates;
    candidates.reserve(cloud.points.size() / 4 + 8);
    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        if (!cloud.mask.empty() && !cloud.mask[i]) continue;
        const Vec3& p = cloud.points[i];
        const float dx = p.x - center.x;
        const float dy = p.y - center.y;
        if (dx * dx + dy * dy > r2) continue;
        float sx, sy, ndcZ;
        if (!ProjectToScreen(p, mvp, fbW, fbH, sx, sy, ndcZ)) continue;
        candidates.push_back(i);
    }
    if (candidates.empty()) return;

    float minX = 1e30f, maxX = -1e30f, minY = 1e30f, maxY = -1e30f;
    for (std::size_t idx : candidates) {
        float sx, sy, ndcZ;
        if (!ProjectToScreen(cloud.points[idx], mvp, fbW, fbH, sx, sy, ndcZ)) continue;
        minX = std::min(minX, sx);
        maxX = std::max(maxX, sx);
        minY = std::min(minY, sy);
        maxY = std::max(maxY, sy);
    }
    SelectRoi(cloud, camera, fbW, fbH, minX, minY, maxX, maxY, outIndices);
    outIndices.erase(std::remove_if(outIndices.begin(), outIndices.end(),
                                    [&](std::size_t idx) {
                                        const Vec3& p = cloud.points[idx];
                                        const float dx = p.x - center.x;
                                        const float dy = p.y - center.y;
                                        return dx * dx + dy * dy > r2;
                                    }),
                     outIndices.end());
    (void)eye;
}

void SelectRoiWorldRectXY(const PointCloud& cloud, const Camera& camera, int fbW, int fbH,
                          const Vec3& center, float halfW, float halfH,
                          std::vector<std::size_t>& outIndices) {
    outIndices.clear();
    if (cloud.points.empty() || halfW <= 0.f || halfH <= 0.f) return;
    const float aspect = static_cast<float>(fbW) / static_cast<float>(fbH);
    const Mat4 mvp = camera.ProjMatrix(aspect) * camera.ViewMatrix();

    std::vector<std::size_t> candidates;
    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        if (!cloud.mask.empty() && !cloud.mask[i]) continue;
        const Vec3& p = cloud.points[i];
        if (std::fabs(p.x - center.x) > halfW || std::fabs(p.y - center.y) > halfH) continue;
        float sx, sy, ndcZ;
        if (!ProjectToScreen(p, mvp, fbW, fbH, sx, sy, ndcZ)) continue;
        candidates.push_back(i);
    }
    if (candidates.empty()) return;

    float minX = 1e30f, maxX = -1e30f, minY = 1e30f, maxY = -1e30f;
    for (std::size_t idx : candidates) {
        float sx, sy, ndcZ;
        if (!ProjectToScreen(cloud.points[idx], mvp, fbW, fbH, sx, sy, ndcZ)) continue;
        minX = std::min(minX, sx);
        maxX = std::max(maxX, sx);
        minY = std::min(minY, sy);
        maxY = std::max(maxY, sy);
    }
    SelectRoi(cloud, camera, fbW, fbH, minX, minY, maxX, maxY, outIndices);
    outIndices.erase(std::remove_if(outIndices.begin(), outIndices.end(),
                                    [&](std::size_t idx) {
                                        const Vec3& p = cloud.points[idx];
                                        return std::fabs(p.x - center.x) > halfW ||
                                               std::fabs(p.y - center.y) > halfH;
                                    }),
                     outIndices.end());
}

bool MeasureHoleRadiusOnPlane(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                              const PlaneModel& plane, HoleMeasureResult& out,
                              std::string& error) {
    out = {};
    const std::vector<std::size_t> use = CollectIndices(cloud, indices);
    if (use.size() < 8) {
        error = u8"至少需要 8 个点才能测量孔径";
        return false;
    }

    Vec3 u, v;
    OrthonormalBasis(plane.normal, u, v);

    std::vector<float> xs, ys;
    xs.reserve(use.size());
    ys.reserve(use.size());
    for (std::size_t idx : use) {
        const Vec3 d = cloud.points[idx] - plane.centroid;
        xs.push_back(d.Dot(u));
        ys.push_back(d.Dot(v));
    }

    float cx2 = 0.f, cy2 = 0.f, rAll = 0.f, rmsAll = 0.f;
    if (!FitCircle2D(xs, ys, cx2, cy2, rAll, rmsAll)) {
        error = u8"平面内圆拟合失败，无法估计孔心";
        return false;
    }

    std::vector<float> radii;
    radii.reserve(use.size());
    for (std::size_t i = 0; i < use.size(); ++i) {
        const float dx = xs[i] - cx2;
        const float dy = ys[i] - cy2;
        radii.push_back(std::sqrt(dx * dx + dy * dy));
    }

    const float rMin = *std::min_element(radii.begin(), radii.end());
    const float rMax = *std::max_element(radii.begin(), radii.end());
    const float span = std::max(rMax - rMin, 1e-4f);
    const float innerBand = std::max(span * 0.18f, rMin * 0.08f + 0.05f);

    std::vector<float> innerXs, innerYs;
    innerXs.reserve(use.size() / 4 + 8);
    innerYs.reserve(use.size() / 4 + 8);
    for (std::size_t i = 0; i < use.size(); ++i) {
        if (radii[i] <= rMin + innerBand) {
            innerXs.push_back(xs[i]);
            innerYs.push_back(ys[i]);
        }
    }

    std::vector<float> sortedRadii = radii;
    std::sort(sortedRadii.begin(), sortedRadii.end());

    float innerR = rMin;
    float innerCx = cx2;
    float innerCy = cy2;
    if (innerXs.size() >= 6) {
        float ir = 0.f, irms = 0.f;
        if (FitCircle2D(innerXs, innerYs, innerCx, innerCy, ir, irms)) {
            innerR = ir;
        }
    } else {
        const std::size_t i8 = std::min(sortedRadii.size() - 1, sortedRadii.size() * 8 / 100);
        innerR = sortedRadii[i8];
    }

    const std::size_t i92 = std::min(sortedRadii.size() - 1, sortedRadii.size() * 92 / 100);
    out.innerRadius = innerR;
    out.outerRadius = sortedRadii[i92];
    out.holeCenter = plane.centroid + u * innerCx + v * innerCy;
    out.plane = plane;
    out.boundaryPointCount = static_cast<int>(use.size());
    out.valid = true;
    return true;
}

bool MeasureHoleRadius(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                       HoleMeasureResult& out, std::string& error) {
    out = {};
    const std::vector<std::size_t> use = CollectIndices(cloud, indices);
    if (use.size() < 8) {
        error = u8"至少需要 8 个点才能测量孔径";
        return false;
    }
    PlaneModel plane;
    if (!FitPlaneSVD(cloud, use, plane, error)) return false;
    return MeasureHoleRadiusOnPlane(cloud, use, plane, out, error);
}

namespace {

void PlaneAxes(int axis, Vec3& normal, Vec3& u, Vec3& v) {
    if (axis == 0) {
        normal = {1, 0, 0};
        u = {0, 1, 0};
        v = {0, 0, 1};
    } else if (axis == 1) {
        normal = {0, 1, 0};
        u = {1, 0, 0};
        v = {0, 0, 1};
    } else {
        normal = {0, 0, 1};
        u = {1, 0, 0};
        v = {0, 1, 0};
    }
}

float EstimateGridStep2D(const std::vector<float>& xs, const std::vector<float>& ys) {
    const std::size_t n = xs.size();
    if (n < 2) return 0.5f;
    float minX = xs[0], maxX = xs[0], minY = ys[0], maxY = ys[0];
    for (std::size_t i = 1; i < n; ++i) {
        minX = std::min(minX, xs[i]);
        maxX = std::max(maxX, xs[i]);
        minY = std::min(minY, ys[i]);
        maxY = std::max(maxY, ys[i]);
    }
    const float diag = std::hypot(maxX - minX, maxY - minY);
    float step = diag / std::sqrt(static_cast<float>(n)) * 0.55f;
    step = std::max(step, diag * 0.01f);
    return std::max(step, 1e-4f);
}

constexpr float kPiFill = 3.14159265358979323846f;

// 极坐标填充圆盘：外圈点精确落在 clipR，避免方格网边界内缩
void FillCirclePolar(float clipUx, float clipUy, float clipR, float step, const Vec3& centroid,
                     const Vec3& u, const Vec3& v, PointCloud& filledOut) {
    filledOut.points.clear();
    if (clipR <= 0.f || step <= 0.f) return;

    const int nBoundary =
        std::max(96, static_cast<int>(std::ceil(2.f * kPiFill * clipR / step)));
    filledOut.points.reserve(static_cast<std::size_t>(nBoundary * 4 + 64));
    for (int k = 0; k < nBoundary; ++k) {
        const float a = 2.f * kPiFill * static_cast<float>(k) / static_cast<float>(nBoundary);
        const float x = clipUx + clipR * std::cos(a);
        const float y = clipUy + clipR * std::sin(a);
        filledOut.points.push_back(centroid + u * x + v * y);
    }
    for (float r = step; r < clipR - step * 0.5f; r += step) {
        const int nAng = std::max(12, static_cast<int>(std::ceil(2.f * kPiFill * r / step)));
        for (int k = 0; k < nAng; ++k) {
            const float a = 2.f * kPiFill * static_cast<float>(k) / static_cast<float>(nAng);
            const float x = clipUx + r * std::cos(a);
            const float y = clipUy + r * std::sin(a);
            filledOut.points.push_back(centroid + u * x + v * y);
        }
    }
}

bool FitCircleOnFilledDisk(const PointCloud& filled, const PlaneModel& plane,
                           const std::vector<float>& sourceXs, const std::vector<float>& sourceYs,
                           float gridStep, CircleModel& out, std::string& error) {
    if (filled.points.size() < 8) {
        error = u8"填充后点数不足";
        return false;
    }

    Vec3 u, v;
    OrthonormalBasis(plane.normal, u, v);

    float cx = 0.f, cy = 0.f, rDummy = 0.f, rms = 0.f;
    if (sourceXs.size() >= 3 &&
        FitCircle2D(sourceXs, sourceYs, cx, cy, rDummy, rms)) {
        // 用原始框选点定圆心，避免填充网格拉偏中心
    } else {
        std::vector<float> fxs, fys;
        fxs.reserve(filled.points.size());
        fys.reserve(filled.points.size());
        for (const Vec3& p : filled.points) {
            const Vec3 d = p - plane.centroid;
            fxs.push_back(d.Dot(u));
            fys.push_back(d.Dot(v));
        }
        if (!FitCircle2D(fxs, fys, cx, cy, rDummy, rms)) {
            error = u8"平面内圆拟合失败。";
            return false;
        }
    }

    std::vector<float> radii;
    radii.reserve(filled.points.size());
    for (const Vec3& p : filled.points) {
        const Vec3 d = p - plane.centroid;
        const float x = d.Dot(u) - cx;
        const float y = d.Dot(v) - cy;
        radii.push_back(std::sqrt(x * x + y * y));
    }
    std::sort(radii.begin(), radii.end());

    const std::size_t i99 =
        std::min(radii.size() - 1, radii.size() * 99 / 100);
    const float radius =
        (gridStep > 0.f) ? (radii[i99] + gridStep * 0.5f) : radii.back();

    const float band = std::max(radius * 0.02f, gridStep > 0.f ? gridStep * 0.6f : 1e-4f);
    out.inlierIndices.clear();
    out.inlierIndices.reserve(filled.points.size());
    double circAcc = 0.0;
    for (std::size_t i = 0; i < filled.points.size(); ++i) {
        const Vec3 d = filled.points[i] - plane.centroid;
        const float x = d.Dot(u) - cx;
        const float y = d.Dot(v) - cy;
        const float ri = std::sqrt(x * x + y * y);
        const float e = ri - radius;
        circAcc += static_cast<double>(e) * static_cast<double>(e);
        if (std::fabs(e) <= band) out.inlierIndices.push_back(i);
    }

    out.center = plane.centroid + u * cx + v * cy;
    out.normal = plane.normal;
    out.radius = radius;
    out.rms = static_cast<float>(std::sqrt(circAcc / static_cast<double>(filled.points.size())));
    out.pointCount = static_cast<int>(filled.points.size());
    return true;
}

bool FillPlaneHolesOnly(const PointCloud& cloud, const Vec3& normal, const Vec3& centroid,
                        const Vec3& u, const Vec3& v, float minX, float maxX, float minY,
                        float maxY, float step, PointCloud& filledOut, std::string& error) {
    filledOut.Clear();
    if (step <= 0.f) {
        error = u8"网格步长无效";
        return false;
    }

    const int gridW = std::max(1, static_cast<int>(std::ceil((maxX - minX) / step)) + 1);
    const int gridH = std::max(1, static_cast<int>(std::ceil((maxY - minY) / step)) + 1);
    if (static_cast<std::int64_t>(gridW) * gridH > 4000000) {
        error = u8"填充网格过大，请增大网格步长";
        return false;
    }

    const float planeD = -centroid.Dot(normal);
    std::vector<uint8_t> surface(static_cast<std::size_t>(gridW) * static_cast<std::size_t>(gridH),
                                 0);
    auto markCell = [&](float x, float y) {
        const int ix = static_cast<int>((x - minX) / step);
        const int iy = static_cast<int>((y - minY) / step);
        if (ix < 0 || iy < 0 || ix >= gridW || iy >= gridH) return;
        surface[static_cast<std::size_t>(iy * gridW + ix)] = 1;
    };

    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        if (!cloud.mask.empty() && !cloud.mask[i]) continue;
        const Vec3& p = cloud.points[i];
        const Vec3 proj = p - normal * (p.Dot(normal) + planeD);
        const Vec3 d = proj - centroid;
        const float x = d.Dot(u);
        const float y = d.Dot(v);
        if (x < minX || x > maxX || y < minY || y > maxY) continue;
        markCell(x, y);
    }

    std::vector<uint8_t> dilated = surface;
    for (int iy = 0; iy < gridH; ++iy) {
        for (int ix = 0; ix < gridW; ++ix) {
            if (!surface[static_cast<std::size_t>(iy * gridW + ix)]) continue;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = ix + dx;
                    const int ny = iy + dy;
                    if (nx < 0 || ny < 0 || nx >= gridW || ny >= gridH) continue;
                    dilated[static_cast<std::size_t>(ny * gridW + nx)] = 1;
                }
            }
        }
    }
    surface = std::move(dilated);

    std::vector<uint8_t> outside(static_cast<std::size_t>(gridW) * static_cast<std::size_t>(gridH),
                                 0);
    std::vector<int> queue;
    queue.reserve(static_cast<std::size_t>(gridW + gridH) * 2);
    auto tryPush = [&](int ix, int iy) {
        if (ix < 0 || iy < 0 || ix >= gridW || iy >= gridH) return;
        const std::size_t gi = static_cast<std::size_t>(iy * gridW + ix);
        if (surface[gi] || outside[gi]) return;
        outside[gi] = 1;
        queue.push_back(iy * gridW + ix);
    };
    for (int ix = 0; ix < gridW; ++ix) {
        tryPush(ix, 0);
        tryPush(ix, gridH - 1);
    }
    for (int iy = 0; iy < gridH; ++iy) {
        tryPush(0, iy);
        tryPush(gridW - 1, iy);
    }
    for (std::size_t qi = 0; qi < queue.size(); ++qi) {
        const int ix = queue[qi] % gridW;
        const int iy = queue[qi] / gridW;
        tryPush(ix - 1, iy);
        tryPush(ix + 1, iy);
        tryPush(ix, iy - 1);
        tryPush(ix, iy + 1);
    }

    filledOut.points.reserve(surface.size() / 8 + 8);
    for (int iy = 0; iy < gridH; ++iy) {
        for (int ix = 0; ix < gridW; ++ix) {
            const std::size_t gi = static_cast<std::size_t>(iy * gridW + ix);
            if (surface[gi] || outside[gi]) continue;
            const float cx = minX + (static_cast<float>(ix) + 0.5f) * step;
            const float cy = minY + (static_cast<float>(iy) + 0.5f) * step;
            filledOut.points.push_back(centroid + u * cx + v * cy);
        }
    }

    if (filledOut.points.empty()) {
        error = u8"未检测到可填充孔洞（请确认已挖孔且矩形框住孔缘）";
        return false;
    }
    return true;
}

void CollectAnnulusIndices(const PointCloud& cloud, const Vec3& center, float holeRadius,
                           std::vector<std::size_t>& out) {
    if (holeRadius <= 0.f) return;
    const float rInner = holeRadius * 1.02f;
    const float rOuter = holeRadius * 4.f;
    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        if (!cloud.mask.empty() && !cloud.mask[i]) continue;
        const Vec3 d = cloud.points[i] - center;
        const float r = std::hypot(d.x, d.y);
        if (r >= rInner && r <= rOuter) out.push_back(i);
    }
}

float MeanCoordZ(const PointCloud& cloud, const std::vector<std::size_t>& idx, float fallback) {
    if (idx.empty()) return fallback;
    double sum = 0.0;
    for (std::size_t i : idx) sum += cloud.points[i].z;
    return static_cast<float>(sum / static_cast<double>(idx.size()));
}

}  // namespace

bool RoiFill(const PointCloud& cloud, const std::vector<std::size_t>& indices, RoiFillMode mode,
             int axis, float gridStepMm, bool clipCircle, const Vec3& clipCenter, float clipRadius,
             const std::vector<std::size_t>* planeFitIndices, PointCloud& filledOut,
             PlaneModel& planeOut, float& outGridStep, std::string& error) {
    filledOut.Clear();
    planeOut = {};
    outGridStep = 0.f;

    const std::vector<std::size_t> use = CollectIndices(cloud, indices);
    const bool circleOnly = clipCircle && clipRadius > 0.f;
    if (mode != RoiFillMode::FitPlane && use.size() < 3 && !circleOnly) {
        error = u8"至少需要 3 个框选点，或设置世界尺寸圆形 ROI";
        return false;
    }
    if (mode == RoiFillMode::ProjectAxis && (axis < 0 || axis > 2)) {
        error = u8"投影轴无效";
        return false;
    }

    Vec3 normal, u, v;
    Vec3 centroid = clipCenter;
    std::vector<float> xs, ys;
    xs.reserve(use.size());
    ys.reserve(use.size());

    if (mode == RoiFillMode::FitPlane) {
        if (!planeFitIndices || planeFitIndices->size() < 3) {
            error = u8"请先用矩形框选区域 A（用于平面拟合）";
            return false;
        }
        const std::vector<std::size_t>& fitIdx = *planeFitIndices;
        if (!FitPlaneSVD(cloud, fitIdx, planeOut, error)) return false;
        normal = planeOut.normal;
        centroid = planeOut.centroid;
        OrthonormalBasis(normal, u, v);
        const float planeD = -centroid.Dot(normal);
        for (std::size_t idx : fitIdx) {
            const Vec3& p = cloud.points[idx];
            const Vec3 proj = p - normal * (p.Dot(normal) + planeD);
            const Vec3 d = proj - centroid;
            xs.push_back(d.Dot(u));
            ys.push_back(d.Dot(v));
        }
    } else if (mode == RoiFillMode::DirectXY) {
        normal = {0, 0, 1};
        u = {1, 0, 0};
        v = {0, 1, 0};
        std::vector<std::size_t> refIdx = use;
        if (refIdx.empty() && circleOnly) {
            CollectAnnulusIndices(cloud, clipCenter, clipRadius, refIdx);
        }
        const float zRef = MeanCoordZ(cloud, refIdx, clipCenter.z);
        if (!use.empty()) {
            centroid = Vec3{0, 0, 0};
            for (std::size_t idx : use) centroid += cloud.points[idx];
            centroid = centroid / static_cast<float>(use.size());
        }
        centroid.z = zRef;
        for (std::size_t idx : use) {
            xs.push_back(cloud.points[idx].x - centroid.x);
            ys.push_back(cloud.points[idx].y - centroid.y);
        }
        planeOut.normal = normal;
        planeOut.centroid = centroid;
    } else {
        PlaneAxes(axis, normal, u, v);
        if (!use.empty()) {
            centroid = Vec3{0, 0, 0};
            for (std::size_t idx : use) centroid += cloud.points[idx];
            centroid = centroid / static_cast<float>(use.size());
        }
        const float planeD = -centroid.Dot(normal);
        for (std::size_t idx : use) {
            const Vec3& p = cloud.points[idx];
            const float t = p.Dot(normal) + planeD;
            const Vec3 proj = p - normal * t;
            const Vec3 d = proj - centroid;
            xs.push_back(d.Dot(u));
            ys.push_back(d.Dot(v));
        }
    }

    float step = gridStepMm;
    float clipUx = 0.f, clipUy = 0.f, clipR = clipRadius;
    if (clipCircle) {
        const Vec3 dc = clipCenter - centroid;
        clipUx = dc.Dot(u);
        clipUy = dc.Dot(v);
        if (clipR <= 0.f && !xs.empty()) {
            clipR = 0.f;
            for (std::size_t i = 0; i < xs.size(); ++i) {
                clipR = std::max(clipR, std::hypot(xs[i] - clipUx, ys[i] - clipUy));
            }
        }
    }
    const bool fullCircleFill =
        mode != RoiFillMode::FitPlane && clipCircle && clipR > 0.f;

    if (step <= 0.f) {
        if (!xs.empty()) {
            step = EstimateGridStep2D(xs, ys);
        } else {
            step = std::max(clipR * 0.05f, 0.1f);
        }
    }
    if (fullCircleFill) {
        const float stepCap = (2.f * clipR) / 120.f;
        if (gridStepMm <= 0.f) step = std::min(step, stepCap);
    }
    outGridStep = step;

    if (clipCircle && clipR <= 0.f && !xs.empty()) clipR += step;

    if (fullCircleFill) {
        FillCirclePolar(clipUx, clipUy, clipR, step, centroid, u, v, filledOut);
    } else if (mode == RoiFillMode::FitPlane) {
        if (xs.empty()) {
            error = u8"框选区域内没有点";
            return false;
        }
        float minX = xs[0], maxX = xs[0], minY = ys[0], maxY = ys[0];
        for (std::size_t i = 1; i < xs.size(); ++i) {
            minX = std::min(minX, xs[i]);
            maxX = std::max(maxX, xs[i]);
            minY = std::min(minY, ys[i]);
            maxY = std::max(maxY, ys[i]);
        }
        const float pad = step * 0.5f;
        minX -= pad;
        maxX += pad;
        minY -= pad;
        maxY += pad;
        if (!FillPlaneHolesOnly(cloud, normal, centroid, u, v, minX, maxX, minY, maxY, step,
                                filledOut, error)) {
            return false;
        }
    } else {
        if (xs.empty()) {
            error = u8"框选区域内没有点";
            return false;
        }
        float minX = xs[0], maxX = xs[0], minY = ys[0], maxY = ys[0];
        for (std::size_t i = 1; i < xs.size(); ++i) {
            minX = std::min(minX, xs[i]);
            maxX = std::max(maxX, xs[i]);
            minY = std::min(minY, ys[i]);
            maxY = std::max(maxY, ys[i]);
        }
        const float pad = step * 2.f;
        minX -= pad;
        maxX += pad;
        minY -= pad;
        maxY += pad;

        const int gridW = std::max(1, static_cast<int>(std::ceil((maxX - minX) / step)) + 1);
        const int gridH = std::max(1, static_cast<int>(std::ceil((maxY - minY) / step)) + 1);
        if (static_cast<std::int64_t>(gridW) * gridH > 4000000) {
            error = u8"填充网格过大，请增大网格步长";
            return false;
        }

        std::vector<uint8_t> occ(static_cast<std::size_t>(gridW * gridH), 0);
        auto markCell = [&](float x, float y) {
            const int ix = static_cast<int>((x - minX) / step);
            const int iy = static_cast<int>((y - minY) / step);
            if (ix < 0 || iy < 0 || ix >= gridW || iy >= gridH) return;
            occ[static_cast<std::size_t>(iy * gridW + ix)] = 1;
        };
        for (std::size_t i = 0; i < xs.size(); ++i) markCell(xs[i], ys[i]);

        for (int pass = 0; pass < 2; ++pass) {
            std::vector<uint8_t> next = occ;
            for (int iy = 0; iy < gridH; ++iy) {
                for (int ix = 0; ix < gridW; ++ix) {
                    if (!occ[static_cast<std::size_t>(iy * gridW + ix)]) continue;
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int nx = ix + dx;
                            const int ny = iy + dy;
                            if (nx < 0 || ny < 0 || nx >= gridW || ny >= gridH) continue;
                            next[static_cast<std::size_t>(ny * gridW + nx)] = 1;
                        }
                    }
                }
            }
            occ = std::move(next);
        }

        std::vector<uint8_t> outside(static_cast<std::size_t>(gridW * gridH), 0);
        std::vector<int> queue;
        queue.reserve(static_cast<std::size_t>(gridW + gridH) * 2);
        auto tryPush = [&](int ix, int iy) {
            if (ix < 0 || iy < 0 || ix >= gridW || iy >= gridH) return;
            const std::size_t gi = static_cast<std::size_t>(iy * gridW + ix);
            if (occ[gi] || outside[gi]) return;
            outside[gi] = 1;
            queue.push_back(iy * gridW + ix);
        };
        for (int ix = 0; ix < gridW; ++ix) {
            tryPush(ix, 0);
            tryPush(ix, gridH - 1);
        }
        for (int iy = 0; iy < gridH; ++iy) {
            tryPush(0, iy);
            tryPush(gridW - 1, iy);
        }
        for (std::size_t qi = 0; qi < queue.size(); ++qi) {
            const int ix = queue[qi] % gridW;
            const int iy = queue[qi] / gridW;
            tryPush(ix - 1, iy);
            tryPush(ix + 1, iy);
            tryPush(ix, iy - 1);
            tryPush(ix, iy + 1);
        }
        for (std::size_t gi = 0; gi < occ.size(); ++gi) {
            if (!occ[gi] && !outside[gi]) occ[gi] = 1;
        }

        filledOut.points.reserve(occ.size() / 4 + 8);
        for (int iy = 0; iy < gridH; ++iy) {
            for (int ix = 0; ix < gridW; ++ix) {
                if (!occ[static_cast<std::size_t>(iy * gridW + ix)]) continue;
                const float cx = minX + (static_cast<float>(ix) + 0.5f) * step;
                const float cy = minY + (static_cast<float>(iy) + 0.5f) * step;
                filledOut.points.push_back(centroid + u * cx + v * cy);
            }
        }
    }

    if (filledOut.points.size() < 8) {
        error = u8"填充后点数不足，请增大框选范围或减小网格步长";
        return false;
    }

    filledOut.ResetMask();
    filledOut.RecomputeBounds();
    if (mode == RoiFillMode::FitPlane) {
        filledOut.sourcePath = u8"<平面拟合填充>";
    } else if (mode == RoiFillMode::DirectXY) {
        filledOut.sourcePath = u8"<直接填充>";
    } else {
        filledOut.sourcePath = u8"<投影填充>";
    }

    if (mode != RoiFillMode::FitPlane) {
        planeOut.normal = normal;
        planeOut.centroid = centroid;
    }
    return true;
}

bool RoiProjectFill(const PointCloud& cloud, const std::vector<std::size_t>& indices, int axis,
                    float gridStepMm, bool clipCircle, const Vec3& clipCenter, float clipRadius,
                    PointCloud& filledOut, PlaneModel& planeOut, float& outGridStep,
                    std::string& error) {
    return RoiFill(cloud, indices, RoiFillMode::ProjectAxis, axis, gridStepMm, clipCircle,
                   clipCenter, clipRadius, nullptr, filledOut, planeOut, outGridStep, error);
}

bool RoiProjectFillAndFitCircle(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                                int axis, float gridStepMm, bool clipCircle,
                                const Vec3& clipCenter, float clipRadius, PointCloud& filledOut,
                                CircleModel& circleOut, PlaneModel& planeOut, float& outGridStep,
                                std::string& error) {
    circleOut = {};
    if (!RoiProjectFill(cloud, indices, axis, gridStepMm, clipCircle, clipCenter, clipRadius,
                         filledOut, planeOut, outGridStep, error)) {
        return false;
    }

    std::vector<float> xs, ys;
    const std::vector<std::size_t> use = CollectIndices(cloud, indices);
    Vec3 u, v;
    OrthonormalBasis(planeOut.normal, u, v);
    for (std::size_t idx : use) {
        const Vec3 d = cloud.points[idx] - planeOut.centroid;
        xs.push_back(d.Dot(u));
        ys.push_back(d.Dot(v));
    }

    const float fitGridStep = (clipCircle && clipRadius > 0.f) ? 0.f : outGridStep;
    return FitCircleOnFilledDisk(filledOut, planeOut, xs, ys, fitGridStep, circleOut, error);
}

void ApplyRoiDelete(PointCloud& cloud, const std::vector<std::size_t>& roiIndices, bool deleteInside) {
    if (cloud.points.empty() || roiIndices.empty()) return;
    if (cloud.mask.size() != cloud.points.size()) cloud.ResetMask();

    std::vector<uint8_t> inRoi(cloud.points.size(), 0);
    for (std::size_t idx : roiIndices) {
        if (idx < inRoi.size()) inRoi[idx] = 1;
    }

    if (deleteInside) {
        for (std::size_t i = 0; i < cloud.points.size(); ++i) {
            if (inRoi[i]) cloud.mask[i] = 0;
        }
    } else {
        for (std::size_t i = 0; i < cloud.points.size(); ++i) {
            if (cloud.mask[i] && !inRoi[i]) cloud.mask[i] = 0;
        }
    }
}

void RestoreAllPoints(PointCloud& cloud) {
    cloud.ResetMask();
}

bool ExtractSection(const PointCloud& cloud, bool cutAlongX, float position, float thickness,
                    SectionData& out, std::string& error, int maxPoints) {
    out = {};
    out.cutAlongX = cutAlongX;
    out.position = position;
    out.thickness = std::max(thickness, 1e-6f);
    const float half = out.thickness * 0.5f;

    std::size_t visible = 0;
    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        if (cloud.mask.empty() || cloud.mask[i]) ++visible;
    }
    if (visible == 0) {
        error = "没有可见点可用于截面。";
        return false;
    }

    // Adaptive stride so extraction stays interactive on huge clouds.
    const int budget = std::max(maxPoints, 1000);
    int stride = 1;
    if (static_cast<int>(visible) > budget * 4) {
        stride = static_cast<int>(visible / static_cast<std::size_t>(budget * 4)) + 1;
    }

    out.points.reserve(static_cast<std::size_t>(budget));
    int seen = 0;
    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        if (!cloud.mask.empty() && !cloud.mask[i]) continue;
        if ((seen++ % stride) != 0) continue;
        const Vec3& p = cloud.points[i];
        const float coord = cutAlongX ? p.x : p.y;
        if (std::fabs(coord - position) > half) continue;

        SectionPoint2D sp;
        sp.p3 = p;
        if (cutAlongX) {
            sp.u = p.y;
            sp.v = p.z;
        } else {
            sp.u = p.x;
            sp.v = p.z;
        }
        out.points.push_back(sp);
        if (static_cast<int>(out.points.size()) >= budget) break;
    }

    if (out.points.empty()) {
        error = "截面厚度内没有点，请增大厚度或调整位置。";
        return false;
    }

    out.uMin = out.uMax = out.points[0].u;
    out.vMin = out.vMax = out.points[0].v;
    for (const auto& sp : out.points) {
        out.uMin = std::min(out.uMin, sp.u);
        out.uMax = std::max(out.uMax, sp.u);
        out.vMin = std::min(out.vMin, sp.v);
        out.vMax = std::max(out.vMax, sp.v);
    }
    if (out.uMax - out.uMin < 1e-6f) {
        out.uMin -= 0.5f;
        out.uMax += 0.5f;
    }
    if (out.vMax - out.vMin < 1e-6f) {
        out.vMin -= 0.5f;
        out.vMax += 0.5f;
    }

    // Sort by u for a cleaner contour polyline.
    std::sort(out.points.begin(), out.points.end(),
              [](const SectionPoint2D& a, const SectionPoint2D& b) { return a.u < b.u; });
    return true;
}

PlaneModel MakeSectionCutPlane(const PointCloud& cloud, bool cutAlongX, float position) {
    PlaneModel plane;
    plane.centroid = cloud.bounds.Center();
    if (cutAlongX) {
        plane.centroid.x = position;
        plane.normal = {1, 0, 0};
        plane.halfSize = std::max({cloud.bounds.Extent().y, cloud.bounds.Extent().z, 0.1f}) * 0.6f;
    } else {
        plane.centroid.y = position;
        plane.normal = {0, 1, 0};
        plane.halfSize = std::max({cloud.bounds.Extent().x, cloud.bounds.Extent().z, 0.1f}) * 0.6f;
    }
    plane.halfExtentU = plane.halfSize;
    plane.halfExtentV = plane.halfSize;
    plane.pointCount = 0;
    plane.rms = 0.f;
    return plane;
}

bool ComputeFlatness(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                     FlatnessResult& out, std::string& error) {
    out = {};
    PlaneModel plane;
    if (!FitPlaneSVD(cloud, indices, plane, error)) return false;

    out.indices = indices.empty() ? std::vector<std::size_t>{} : indices;
    if (out.indices.empty()) {
        out.indices.reserve(cloud.points.size());
        for (std::size_t i = 0; i < cloud.points.size(); ++i) {
            if (cloud.mask.empty() || cloud.mask[i]) out.indices.push_back(i);
        }
    }

    out.signedDist.resize(out.indices.size());
    float minD = std::numeric_limits<float>::max();
    float maxD = std::numeric_limits<float>::lowest();
    double absAcc = 0.0;
    double rmsAcc = 0.0;
    for (std::size_t k = 0; k < out.indices.size(); ++k) {
        const Vec3& p = cloud.points[out.indices[k]];
        const float d = (p - plane.centroid).Dot(plane.normal);
        out.signedDist[k] = d;
        minD = std::min(minD, d);
        maxD = std::max(maxD, d);
        absAcc += std::fabs(static_cast<double>(d));
        rmsAcc += static_cast<double>(d) * static_cast<double>(d);
    }

    out.plane = plane;
    out.minDev = minD;
    out.maxDev = maxD;
    out.peakToValley = maxD - minD;
    out.meanAbs = static_cast<float>(absAcc / static_cast<double>(out.indices.size()));
    out.rms = static_cast<float>(std::sqrt(rmsAcc / static_cast<double>(out.indices.size())));
    out.valid = true;
    return true;
}

void ComputePlaneDistanceProfile(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                                 const PlaneModel& plane, PlaneDistanceProfile& out) {
    out = {};
    std::vector<std::size_t> use = indices;
    if (use.empty()) {
        use.reserve(cloud.points.size());
        for (std::size_t i = 0; i < cloud.points.size(); ++i) {
            if (cloud.mask.empty() || cloud.mask[i]) use.push_back(i);
        }
    }
    if (use.empty()) return;

    out.signedDist.resize(use.size());
    float minD = std::numeric_limits<float>::max();
    float maxD = std::numeric_limits<float>::lowest();
    double rmsAcc = 0.0;
    const Vec3 n = plane.normal.Normalized();
    for (std::size_t k = 0; k < use.size(); ++k) {
        const Vec3& p = cloud.points[use[k]];
        const float d = (p - plane.centroid).Dot(n);
        out.signedDist[k] = d;
        minD = std::min(minD, d);
        maxD = std::max(maxD, d);
        rmsAcc += static_cast<double>(d) * static_cast<double>(d);
    }
    out.minD = minD;
    out.maxD = maxD;
    out.rms = static_cast<float>(std::sqrt(rmsAcc / static_cast<double>(use.size())));
    out.pointCount = static_cast<int>(use.size());
    out.valid = true;
}

bool ComputeStepGapDistances(const PointCloud& cloud, const PlaneModel& planeA,
                             const std::vector<std::size_t>& regionB, StepGapResult& out,
                             std::string& error) {
    // 保留：相对拟合面的竖直高度（仅作对比）；主流程改用 ComputeStepGapZHeight
    if (regionB.empty()) {
        error = u8"区域 B 为空，请先框选";
        return false;
    }
    const Vec3& n = planeA.normal;
    if (std::fabs(n.z) < 1e-6f) {
        error = u8"基准面接近竖直，无法计算 Z 向高度差";
        return false;
    }
    out.regionB = regionB;
    out.signedDistB.resize(regionB.size());
    float minD = std::numeric_limits<float>::max();
    float maxD = std::numeric_limits<float>::lowest();
    double sum = 0.0;
    double absAcc = 0.0;
    double rmsAcc = 0.0;
    for (std::size_t k = 0; k < regionB.size(); ++k) {
        const std::size_t idx = regionB[k];
        if (idx >= cloud.points.size()) {
            out.signedDistB[k] = 0.f;
            continue;
        }
        const float d = (cloud.points[idx] - planeA.centroid).Dot(n) / n.z;
        out.signedDistB[k] = d;
        minD = std::min(minD, d);
        maxD = std::max(maxD, d);
        sum += d;
        absAcc += std::fabs(static_cast<double>(d));
        rmsAcc += static_cast<double>(d) * static_cast<double>(d);
    }
    const double count = static_cast<double>(regionB.size());
    out.planeA = planeA;
    out.hasPlane = true;
    out.hasDistances = true;
    out.mean = static_cast<float>(sum / count);
    out.meanAbs = static_cast<float>(absAcc / count);
    out.minDist = minD;
    out.maxDist = maxD;
    out.rms = static_cast<float>(std::sqrt(rmsAcc / count));
    out.phase = StepGapPhase::Done;
    return true;
}

bool ComputeStepGapZHeight(const PointCloud& cloud, const std::vector<std::size_t>& regionA,
                           const std::vector<std::size_t>& regionB, StepGapResult& out,
                           std::string& error) {
    // 必须先拷贝：调用方常传 ComputeStepGapZHeight(..., sg.regionA, sg.regionB, sg)
    // 若直接 out={} 会把作为引用的 regionA/B 一并清空。
    const std::vector<std::size_t> indicesA = regionA;
    const std::vector<std::size_t> indicesB = regionB;
    const PlaneModel keptPlane = out.planeA;
    const bool keptHasPlane = out.hasPlane;

    if (indicesA.empty()) {
        error = u8"区域 A 为空，请先框选";
        return false;
    }
    if (indicesB.empty()) {
        error = u8"区域 B 为空，请先框选";
        return false;
    }

    double zSum = 0.0;
    int zCount = 0;
    for (std::size_t idx : indicesA) {
        if (idx >= cloud.points.size()) continue;
        if (!cloud.mask.empty() && !cloud.mask[idx]) continue;
        zSum += cloud.points[idx].z;
        ++zCount;
    }
    if (zCount < 1) {
        error = u8"区域 A 无有效点";
        return false;
    }
    const float zRef = static_cast<float>(zSum / static_cast<double>(zCount));

    out = {};
    out.regionA = indicesA;
    out.regionB = indicesB;
    out.planeA = keptPlane;
    out.hasPlane = keptHasPlane;
    out.zRefA = zRef;
    out.signedDistB.resize(indicesB.size());

    float minD = std::numeric_limits<float>::max();
    float maxD = std::numeric_limits<float>::lowest();
    double sum = 0.0;
    double absAcc = 0.0;
    double rmsAcc = 0.0;
    std::vector<float> sorted;
    sorted.reserve(indicesB.size());

    for (std::size_t k = 0; k < indicesB.size(); ++k) {
        const std::size_t idx = indicesB[k];
        if (idx >= cloud.points.size()) {
            out.signedDistB[k] = 0.f;
            continue;
        }
        const float d = cloud.points[idx].z - zRef;
        out.signedDistB[k] = d;
        sorted.push_back(d);
        minD = std::min(minD, d);
        maxD = std::max(maxD, d);
        sum += d;
        absAcc += std::fabs(static_cast<double>(d));
        rmsAcc += static_cast<double>(d) * static_cast<double>(d);
    }

    if (sorted.empty()) {
        error = u8"区域 B 无有效点";
        return false;
    }

    const double count = static_cast<double>(sorted.size());
    out.mean = static_cast<float>(sum / count);
    out.meanAbs = static_cast<float>(absAcc / count);
    out.minDist = minD;
    out.maxDist = maxD;
    out.rms = static_cast<float>(std::sqrt(rmsAcc / count));
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
    out.median = sorted[sorted.size() / 2];
    out.hasDistances = true;
    out.phase = StepGapPhase::Done;
    return true;
}

bool AlignCloudToPlaneNormal(PointCloud& cloud, const PlaneModel& plane, const Vec3& targetNormal,
                             PlaneModel& outPlane, std::string& error) {
    if (cloud.points.empty()) {
        error = u8"点云为空";
        return false;
    }
    Vec3 srcN = plane.normal.Normalized();
    Vec3 dstN = targetNormal.Normalized();
    if (srcN.Length() < 1e-8f || dstN.Length() < 1e-8f) {
        error = u8"平面法向无效";
        return false;
    }

    const float dot = srcN.Dot(dstN);
    if (dot > 0.9999f) {
        error = u8"平面已与目标方向对齐，无需旋转";
        return false;
    }

    const Mat4 rot = RotationFromTo(srcN, dstN);
    const Vec3 pivot = plane.centroid;
    for (Vec3& p : cloud.points) {
        const Vec3 local = p - pivot;
        p = rot.TransformVector(local) + pivot;
    }

    outPlane = plane;
    outPlane.normal = dstN;
    outPlane.centroid = pivot;

    // 旋转平面局部 U/V 轴用于显示
    Vec3 tmp = (std::fabs(dstN.x) < 0.9f) ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    Vec3 uAxis = dstN.Cross(tmp).Normalized();
    Vec3 vAxis = dstN.Cross(uAxis).Normalized();
    Vec3 oldU = srcN.Cross(tmp).Normalized();
    if (oldU.Length() < 1e-8f) oldU = uAxis;
    oldU = rot.TransformVector(oldU).Normalized();
    float maxU = outPlane.halfExtentU;
    float maxV = outPlane.halfExtentV;
    if (std::fabs(oldU.Dot(uAxis)) >= std::fabs(oldU.Dot(vAxis))) {
        outPlane.halfExtentU = maxU;
        outPlane.halfExtentV = maxV;
    } else {
        outPlane.halfExtentU = maxV;
        outPlane.halfExtentV = maxU;
    }
    outPlane.halfSize = std::max(outPlane.halfExtentU, outPlane.halfExtentV);

    cloud.RecomputeBounds();
    return true;
}

}  // namespace MeasureTools
