#include "tools/PclTools.h"

#include "tools/PclAdapter.h"

#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace PclTools {

inline bool Visible(const PointCloud& cloud, std::size_t i) {
    return cloud.mask.empty() || cloud.mask[i];
}

std::vector<std::size_t> CollectIndices(const PointCloud& cloud,
                                        const std::vector<std::size_t>& indices) {
    if (!indices.empty()) return indices;
    std::vector<std::size_t> use;
    use.reserve(cloud.points.size());
    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        if (Visible(cloud, i)) use.push_back(i);
    }
    return use;
}

namespace {
void InitKeepFromVisible(const PointCloud& cloud, std::vector<uint8_t>& keepMask) {
    keepMask.resize(cloud.points.size());
    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        keepMask[i] = Visible(cloud, i) ? 1 : 0;
    }
}

int CountKept(const std::vector<uint8_t>& keepMask) {
    int kept = 0;
    for (uint8_t m : keepMask)
        if (m) ++kept;
    return kept;
}

inline std::int64_t PackKey(int ix, int iy, int iz) {
    constexpr std::int64_t bias = 1 << 20;
    return ((static_cast<std::int64_t>(ix) + bias) << 42) |
           ((static_cast<std::int64_t>(iy) + bias) << 21) |
           (static_cast<std::int64_t>(iz) + bias);
}

bool BuildPlaneModel(const PointCloud& cloud, const std::vector<std::size_t>& use, const Vec3& n,
                     const Vec3& c, PlaneModel& out) {
    if (use.size() < 3) return false;

    Vec3 normal = n.Normalized();
    if (normal.z < 0.f) normal = normal * -1.f;

    double rmsAcc = 0.0;
    Vec3 tmp = (std::fabs(normal.x) < 0.9f) ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    Vec3 uAxis = normal.Cross(tmp).Normalized();
    Vec3 vAxis = normal.Cross(uAxis).Normalized();
    float maxU = 0.f;
    float maxV = 0.f;
    for (std::size_t idx : use) {
        const Vec3 d = cloud.points[idx] - c;
        const float dist = d.Dot(normal);
        rmsAcc += static_cast<double>(dist) * static_cast<double>(dist);
        const Vec3 onPlane = d - normal * dist;
        maxU = std::max(maxU, std::fabs(onPlane.Dot(uAxis)));
        maxV = std::max(maxV, std::fabs(onPlane.Dot(vAxis)));
    }

    out.centroid = c;
    out.normal = normal;
    out.rms = static_cast<float>(std::sqrt(rmsAcc / static_cast<double>(use.size())));
    out.pointCount = static_cast<int>(use.size());
    out.halfExtentU = std::max(maxU * 1.02f, 0.05f);
    out.halfExtentV = std::max(maxV * 1.02f, 0.05f);
    out.halfSize = std::max(out.halfExtentU, out.halfExtentV);
    return true;
}

}  // namespace

const char* VersionString() { return "PCL 1.14.0"; }

bool VoxelDownsample(const PointCloud& cloud, float leafSize, std::vector<uint8_t>& keepMask,
                     std::string& error, int* outKept) {
    if (cloud.points.empty()) {
        error = u8"点云为空";
        return false;
    }
    if (leafSize <= 0.f) {
        error = u8"体素边长必须 > 0";
        return false;
    }

    std::vector<std::size_t> map;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloud =
        PclAdapter::ToPclXYZ(cloud, {}, map);
    if (pclCloud->empty()) {
        error = u8"没有可见点可供滤波";
        return false;
    }

    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(pclCloud);
    vg.setLeafSize(leafSize, leafSize, leafSize);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
    vg.filter(*filtered);

    InitKeepFromVisible(cloud, keepMask);
    for (std::size_t i = 0; i < keepMask.size(); ++i) {
        if (Visible(cloud, i)) keepMask[i] = 0;
    }

    const float inv = 1.f / leafSize;
    std::unordered_set<std::int64_t> pclVoxels;
    pclVoxels.reserve(filtered->size() + 1);
    for (const auto& pt : filtered->points) {
        const int ix = static_cast<int>(std::floor(pt.x * inv));
        const int iy = static_cast<int>(std::floor(pt.y * inv));
        const int iz = static_cast<int>(std::floor(pt.z * inv));
        pclVoxels.insert(PackKey(ix, iy, iz));
    }

    std::unordered_map<std::int64_t, std::size_t> chosen;
    chosen.reserve(map.size() / 4 + 1);
    for (std::size_t pclIdx = 0; pclIdx < map.size(); ++pclIdx) {
        const Vec3& p = cloud.points[map[pclIdx]];
        const int ix = static_cast<int>(std::floor(p.x * inv));
        const int iy = static_cast<int>(std::floor(p.y * inv));
        const int iz = static_cast<int>(std::floor(p.z * inv));
        const std::int64_t key = PackKey(ix, iy, iz);
        if (!pclVoxels.count(key)) continue;
        if (chosen.find(key) == chosen.end()) {
            chosen.emplace(key, pclIdx);
            keepMask[map[pclIdx]] = 1;
        }
    }

    const int kept = CountKept(keepMask);
    if (outKept) *outKept = kept;
    if (kept == 0) {
        error = u8"PCL 体素滤波后无保留点";
        return false;
    }
    return true;
}

bool RadiusOutlier(const PointCloud& cloud, float radius, int minNeighbors,
                   std::vector<uint8_t>& keepMask, std::string& error, int* outKept) {
    if (cloud.points.empty()) {
        error = u8"点云为空";
        return false;
    }
    if (radius <= 0.f) {
        error = u8"搜索半径必须 > 0";
        return false;
    }
    if (minNeighbors < 1) {
        error = u8"最少邻居数必须 >= 1";
        return false;
    }

    std::vector<std::size_t> map;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloud = PclAdapter::ToPclXYZ(cloud, {}, map);
    if (static_cast<int>(pclCloud->size()) <= minNeighbors) {
        error = u8"可见点数不足以做半径滤波";
        return false;
    }

    pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror;
    ror.setInputCloud(pclCloud);
    ror.setRadiusSearch(radius);
    ror.setMinNeighborsInRadius(minNeighbors);
    std::vector<int> kept;
    ror.filter(kept);

    InitKeepFromVisible(cloud, keepMask);
    for (std::size_t i = 0; i < keepMask.size(); ++i) {
        if (Visible(cloud, i)) keepMask[i] = 0;
    }
    PclAdapter::ApplyKeptIndices(cloud, map, kept, keepMask);

    const int keptCount = CountKept(keepMask);
    if (outKept) *outKept = keptCount;
    if (keptCount == 0) {
        error = u8"PCL 半径滤波后无保留点";
        return false;
    }
    return true;
}

bool StatisticalOutlier(const PointCloud& cloud, int meanK, float stdMul,
                        std::vector<uint8_t>& keepMask, std::string& error, int* outKept) {
    if (cloud.points.empty()) {
        error = u8"点云为空";
        return false;
    }
    if (meanK < 2) {
        error = u8"邻域点数 K 必须 >= 2";
        return false;
    }
    if (stdMul <= 0.f) {
        error = u8"标准差倍数必须 > 0";
        return false;
    }

    std::vector<std::size_t> map;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloud = PclAdapter::ToPclXYZ(cloud, {}, map);
    if (static_cast<int>(pclCloud->size()) <= meanK) {
        error = u8"可见点数不足以做统计滤波";
        return false;
    }

    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(pclCloud);
    sor.setMeanK(meanK);
    sor.setStddevMulThresh(stdMul);
    std::vector<int> kept;
    sor.filter(kept);

    InitKeepFromVisible(cloud, keepMask);
    for (std::size_t i = 0; i < keepMask.size(); ++i) {
        if (Visible(cloud, i)) keepMask[i] = 0;
    }
    PclAdapter::ApplyKeptIndices(cloud, map, kept, keepMask);

    const int keptCount = CountKept(keepMask);
    if (outKept) *outKept = keptCount;
    if (keptCount == 0) {
        error = u8"PCL 统计滤波后无保留点";
        return false;
    }
    return true;
}

bool FitPlaneRANSAC(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                    float distanceThreshold, int maxIterations, PlaneModel& out,
                    std::string& error) {
    std::vector<std::size_t> use = indices;
    if (use.empty()) {
        use.reserve(cloud.points.size());
        for (std::size_t i = 0; i < cloud.points.size(); ++i) {
            if (Visible(cloud, i)) use.push_back(i);
        }
    }
    if (use.size() < 3) {
        error = u8"至少需要 3 个点才能拟合平面";
        return false;
    }
    if (distanceThreshold <= 0.f) {
        error = u8"RANSAC 距离阈值必须 > 0";
        return false;
    }

    std::vector<std::size_t> map;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloud = PclAdapter::ToPclXYZ(cloud, use, map);
    if (pclCloud->size() < 3) {
        error = u8"有效点数不足";
        return false;
    }

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(distanceThreshold);
    seg.setMaxIterations(std::max(1, maxIterations));
    seg.setInputCloud(pclCloud);

    pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    seg.segment(*inliers, *coeffs);
    if (coeffs->values.size() < 4 || inliers->indices.empty()) {
        error = u8"PCL 平面 RANSAC 拟合失败";
        return false;
    }

    Vec3 normal{coeffs->values[0], coeffs->values[1], coeffs->values[2]};
    const float d = coeffs->values[3];
    if (normal.Length() < 1e-12f) {
        error = u8"PCL 返回了无效平面法向";
        return false;
    }
    normal = normal.Normalized();

    Vec3 centroid{0, 0, 0};
    std::vector<std::size_t> inlierOrig;
    inlierOrig.reserve(inliers->indices.size());
    for (int idx : inliers->indices) {
        if (idx < 0 || static_cast<std::size_t>(idx) >= map.size()) continue;
        inlierOrig.push_back(map[static_cast<std::size_t>(idx)]);
    }
    if (inlierOrig.size() < 3) {
        error = u8"PCL 平面内点不足";
        return false;
    }
    for (std::size_t idx : inlierOrig) centroid += cloud.points[idx];
    centroid = centroid / static_cast<float>(inlierOrig.size());

    // ax+by+cz+d=0，用内点质心作为显示中心
    (void)d;
    if (!BuildPlaneModel(cloud, inlierOrig, normal, centroid, out)) {
        error = u8"平面模型构建失败";
        return false;
    }
    out.pointCount = static_cast<int>(inlierOrig.size());
    return true;
}

bool FitSphereRANSAC(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                     float distanceThreshold, int maxIterations, SphereModel& out,
                     std::string& error) {
    std::vector<std::size_t> use = indices;
    if (use.empty()) {
        use.reserve(cloud.points.size());
        for (std::size_t i = 0; i < cloud.points.size(); ++i) {
            if (Visible(cloud, i)) use.push_back(i);
        }
    }
    if (use.size() < 4) {
        error = u8"至少需要 4 个点才能拟合球";
        return false;
    }

    std::vector<std::size_t> map;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloud = PclAdapter::ToPclXYZ(cloud, use, map);
    if (pclCloud->size() < 4) {
        error = u8"有效点数不足";
        return false;
    }

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_SPHERE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(distanceThreshold);
    seg.setMaxIterations(std::max(1, maxIterations));
    seg.setInputCloud(pclCloud);

    pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    seg.segment(*inliers, *coeffs);
    if (coeffs->values.size() < 4 || inliers->indices.empty()) {
        error = u8"PCL 球面 RANSAC 拟合失败";
        return false;
    }

    out.center = {coeffs->values[0], coeffs->values[1], coeffs->values[2]};
    out.radius = coeffs->values[3];
    out.pointCount = static_cast<int>(inliers->indices.size());
    out.inlierIndices.clear();
    out.inlierIndices.reserve(inliers->indices.size());

    double rmsAcc = 0.0;
    for (int idx : inliers->indices) {
        if (idx < 0 || static_cast<std::size_t>(idx) >= map.size()) continue;
        out.inlierIndices.push_back(map[static_cast<std::size_t>(idx)]);
        const Vec3& p = cloud.points[map[static_cast<std::size_t>(idx)]];
        const float d = (p - out.center).Length() - out.radius;
        rmsAcc += static_cast<double>(d) * static_cast<double>(d);
    }
    out.rms = static_cast<float>(
        std::sqrt(rmsAcc / static_cast<double>(std::max(1, out.pointCount))));
    return true;
}

bool FitCircleRANSAC(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                     float distanceThreshold, int maxIterations, CircleModel& out,
                     std::string& error) {
    const std::vector<std::size_t> use = CollectIndices(cloud, indices);
    if (use.size() < 3) {
        error = u8"至少需要 3 个点才能拟合圆";
        return false;
    }

    PlaneModel plane;
    if (!FitPlaneRANSAC(cloud, indices, distanceThreshold, maxIterations, plane, error)) {
        return false;
    }
    return MeasureTools::FitCircleOnPlane(cloud, use, plane, distanceThreshold, out, error);
}

bool FitCylinderRANSAC(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                       float distanceThreshold, int maxIterations, CylinderModel& out,
                       std::string& error) {
    const std::vector<std::size_t> use = CollectIndices(cloud, indices);
    if (use.size() < 6) {
        error = u8"至少需要 6 个点才能拟合圆柱";
        return false;
    }
    if (distanceThreshold <= 0.f) {
        error = u8"RANSAC 距离阈值必须 > 0";
        return false;
    }

    std::vector<std::size_t> map;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloud = PclAdapter::ToPclXYZ(cloud, use, map);
    if (pclCloud->size() < 6) {
        error = u8"有效点数不足";
        return false;
    }

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(pclCloud);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(pclCloud);
    ne.setSearchMethod(tree);
    const int kSearch = std::min(20, static_cast<int>(pclCloud->size()) - 1);
    if (kSearch < 3) {
        error = u8"点数不足以做法向估计";
        return false;
    }
    ne.setKSearch(kSearch);
    ne.compute(*normals);

    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_CYLINDER);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setNormalDistanceWeight(0.1);
    seg.setMaxIterations(std::max(1, maxIterations));
    seg.setDistanceThreshold(distanceThreshold);
    seg.setRadiusLimits(0.0, DBL_MAX);
    seg.setInputCloud(pclCloud);
    seg.setInputNormals(normals);

    pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    seg.segment(*inliers, *coeffs);
    if (coeffs->values.size() < 7 || inliers->indices.empty()) {
        error = u8"PCL 圆柱 RANSAC 拟合失败";
        return false;
    }

    out.axisPoint = {coeffs->values[0], coeffs->values[1], coeffs->values[2]};
    out.axisDir = Vec3{coeffs->values[3], coeffs->values[4], coeffs->values[5]}.Normalized();
    out.radius = coeffs->values[6];
    out.pointCount = static_cast<int>(inliers->indices.size());

    float tMin = 1e30f;
    float tMax = -1e30f;
    double rmsAcc = 0.0;
    for (int idx : inliers->indices) {
        if (idx < 0 || static_cast<std::size_t>(idx) >= map.size()) continue;
        const Vec3& p = cloud.points[map[static_cast<std::size_t>(idx)]];
        const Vec3 d = p - out.axisPoint;
        const float t = d.Dot(out.axisDir);
        tMin = std::min(tMin, t);
        tMax = std::max(tMax, t);
        const Vec3 radial = d - out.axisDir * t;
        const float e = radial.Length() - out.radius;
        rmsAcc += static_cast<double>(e) * static_cast<double>(e);
    }
    out.halfHeight = std::max(0.5f * (tMax - tMin) * 1.05f, out.radius * 0.25f);
    out.rms = static_cast<float>(
        std::sqrt(rmsAcc / static_cast<double>(std::max(1, out.pointCount))));
    return true;
}

bool ComputeFlatness(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                     float distanceThreshold, int maxIterations, FlatnessResult& out,
                     std::string& error) {
    out = {};
    PlaneModel plane;
    if (!FitPlaneRANSAC(cloud, indices, distanceThreshold, maxIterations, plane, error)) {
        return false;
    }

    out.indices = indices.empty() ? std::vector<std::size_t>{} : indices;
    if (out.indices.empty()) {
        out.indices.reserve(cloud.points.size());
        for (std::size_t i = 0; i < cloud.points.size(); ++i) {
            if (Visible(cloud, i)) out.indices.push_back(i);
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

bool ComputeStepGapZHeight(const PointCloud& cloud, const std::vector<std::size_t>& regionA,
                           const std::vector<std::size_t>& regionB, StepGapResult& out,
                           std::string& error) {
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
        if (!Visible(cloud, idx)) continue;
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

bool ExtractSection(const PointCloud& cloud, bool cutAlongX, float position, float thickness,
                    SectionData& out, std::string& error, int maxPoints) {
    out = {};
    out.cutAlongX = cutAlongX;
    out.position = position;
    out.thickness = std::max(thickness, 1e-6f);
    const float half = out.thickness * 0.5f;

    std::vector<std::size_t> map;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloud = PclAdapter::ToPclXYZ(cloud, {}, map);
    if (pclCloud->empty()) {
        error = u8"没有可见点可用于截面";
        return false;
    }

    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(pclCloud);
    pass.setFilterFieldName(cutAlongX ? "x" : "y");
    pass.setFilterLimits(position - half, position + half);
    pcl::PointIndices::Ptr kept(new pcl::PointIndices);
    pass.filter(kept->indices);

    if (kept->indices.empty()) {
        error = u8"截面厚度内没有点，请增大厚度或调整位置";
        return false;
    }

    const int budget = std::max(maxPoints, 1000);
    int stride = 1;
    if (static_cast<int>(kept->indices.size()) > budget * 4) {
        stride = static_cast<int>(kept->indices.size() / static_cast<std::size_t>(budget * 4)) + 1;
    }

    out.points.reserve(static_cast<std::size_t>(budget));
    int seen = 0;
    for (int pclIdx : kept->indices) {
        if (pclIdx < 0 || static_cast<std::size_t>(pclIdx) >= map.size()) continue;
        if ((seen++ % stride) != 0) continue;
        const Vec3& p = cloud.points[map[static_cast<std::size_t>(pclIdx)]];
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
        error = u8"截面厚度内没有点，请增大厚度或调整位置";
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

    std::sort(out.points.begin(), out.points.end(),
              [](const SectionPoint2D& a, const SectionPoint2D& b) { return a.u < b.u; });
    return true;
}

void ApplyClipMask(PointCloud& cloud, const Vec3& normal, float d, bool enabled) {
    if (cloud.mask.size() != cloud.points.size()) cloud.ResetMask();
    if (!enabled) return;
    const Vec3 n = normal.Normalized();
    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        if (!cloud.mask[i]) continue;
        const float side = n.Dot(cloud.points[i]) + d;
        cloud.mask[i] = (side >= 0.f) ? 1 : 0;
    }
}

void ApplyRoiDelete(PointCloud& cloud, const std::vector<std::size_t>& roiIndices,
                    bool deleteInside) {
    MeasureTools::ApplyRoiDelete(cloud, roiIndices, deleteInside);
}

void RestoreAllPoints(PointCloud& cloud) { MeasureTools::RestoreAllPoints(cloud); }

std::optional<std::size_t> PickNearest(const PointCloud& cloud, const Camera& camera, int fbW,
                                       int fbH, float mouseX, float mouseY, float maxPixelDist,
                                       const std::vector<std::size_t>* onlyIndices) {
    const auto screenPick = MeasureTools::PickNearest(cloud, camera, fbW, fbH, mouseX, mouseY,
                                                      maxPixelDist, onlyIndices);
    if (!screenPick) return std::nullopt;

    std::vector<std::size_t> map;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloud = PclAdapter::ToPclXYZ(cloud, {}, map);
    if (pclCloud->size() < 2) return screenPick;

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(pclCloud);

    std::size_t pclIdx = 0;
    bool found = false;
    for (std::size_t i = 0; i < map.size(); ++i) {
        if (map[i] == *screenPick) {
            pclIdx = i;
            found = true;
            break;
        }
    }
    if (!found) return screenPick;

    const pcl::PointXYZ& qpt = pclCloud->points[pclIdx];
    std::vector<int> nnIdx(1);
    std::vector<float> nnDist(1);
    if (kdtree.nearestKSearch(qpt, 1, nnIdx, nnDist) > 0 && nnIdx[0] >= 0 &&
        static_cast<std::size_t>(nnIdx[0]) < map.size()) {
        return map[static_cast<std::size_t>(nnIdx[0])];
    }
    return screenPick;
}

void SelectRoi(const PointCloud& cloud, const Camera& camera, int fbW, int fbH, float x0,
               float y0, float x1, float y1, std::vector<std::size_t>& outIndices) {
    MeasureTools::SelectRoi(cloud, camera, fbW, fbH, x0, y0, x1, y1, outIndices);
}

bool ProjectOntoAxis(PointCloud& cloud, const Vec3& axisOrigin, const Vec3& axisDir,
                     std::string& error) {
    if (cloud.points.empty()) {
        error = u8"点云为空";
        return false;
    }
    Vec3 axis = axisDir.Normalized();
    if (axis.Length() < 1e-8f) {
        error = u8"投影轴方向无效";
        return false;
    }
    if (cloud.mask.size() != cloud.points.size()) cloud.ResetMask();
    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        if (!cloud.mask.empty() && !cloud.mask[i]) continue;
        // 正交投影到过 axisOrigin、法向为 axis 的平面（去掉沿轴分量，保留二维形状）
        const Vec3 v = cloud.points[i] - axisOrigin;
        const float t = v.Dot(axis);
        cloud.points[i] = cloud.points[i] - axis * t;
    }
    cloud.RecomputeBounds();
    return true;
}

bool MeasureHoleRadius(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                       float planeDistThresh, int planeMaxIter, HoleMeasureResult& out,
                       std::string& error) {
    std::vector<std::size_t> use = indices;
    if (use.empty()) {
        for (std::size_t i = 0; i < cloud.points.size(); ++i) {
            if (Visible(cloud, i)) use.push_back(i);
        }
    }
    if (use.size() < 8) {
        error = u8"至少需要 8 个点才能测量孔径";
        return false;
    }
    PlaneModel plane;
    if (!FitPlaneRANSAC(cloud, use, planeDistThresh, planeMaxIter, plane, error)) return false;
    return MeasureTools::MeasureHoleRadiusOnPlane(cloud, use, plane, out, error);
}

bool RoiProjectFill(const PointCloud& cloud, const std::vector<std::size_t>& indices, int axis,
                    float gridStepMm, bool clipCircle, const Vec3& clipCenter, float clipRadius,
                    PointCloud& filledOut, PlaneModel& planeOut, float& outGridStep,
                    std::string& error) {
    return MeasureTools::RoiProjectFill(cloud, indices, axis, gridStepMm, clipCircle, clipCenter,
                                        clipRadius, filledOut, planeOut, outGridStep, error);
}

bool RoiProjectFillAndFitCircle(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                                int axis, float gridStepMm, bool clipCircle,
                                const Vec3& clipCenter, float clipRadius, PointCloud& filledOut,
                                CircleModel& circleOut, PlaneModel& planeOut, float& outGridStep,
                                std::string& error) {
    return MeasureTools::RoiProjectFillAndFitCircle(cloud, indices, axis, gridStepMm, clipCircle,
                                                    clipCenter, clipRadius, filledOut, circleOut,
                                                    planeOut, outGridStep, error);
}

}  // namespace PclTools
