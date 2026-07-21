#include "io/PointCloudIO.h"

#include "io/ImageIO.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

namespace {

std::string ToLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string ExtensionOf(const std::string& path) {
    const auto pos = path.find_last_of('.');
    if (pos == std::string::npos) return {};
    return ToLower(path.substr(pos + 1));
}

bool IsFiniteVec(const Vec3& p) {
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}

bool ParseXYZLine(const std::string& line, Vec3& p) {
    if (line.empty() || line[0] == '#') return false;
    std::istringstream ss(line);
    if (!(ss >> p.x >> p.y >> p.z)) return false;
    return IsFiniteVec(p);
}

int TypeSize(const std::string& type) {
    if (type == "float" || type == "float32") return 4;
    if (type == "double" || type == "float64") return 8;
    if (type == "uchar" || type == "uint8" || type == "char" || type == "int8") return 1;
    if (type == "ushort" || type == "uint16" || type == "short" || type == "int16") return 2;
    if (type == "uint" || type == "uint32" || type == "int" || type == "int32") return 4;
    return 0;
}

bool LoadXYZ(const std::string& path, PointCloud& out, std::string& error) {
    std::ifstream ifs(path);
    if (!ifs) {
        error = "无法打开文件: " + path;
        return false;
    }
    out.Clear();
    std::string line;
    Vec3 p;
    while (std::getline(ifs, line)) {
        if (ParseXYZLine(line, p)) out.points.push_back(p);
    }
    if (out.points.empty()) {
        error = "XYZ 文件中没有有效点。";
        return false;
    }
    return true;
}

bool LoadOBJ(const std::string& path, PointCloud& out, std::string& error) {
    std::ifstream ifs(path);
    if (!ifs) {
        error = "无法打开文件: " + path;
        return false;
    }
    out.Clear();
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.size() < 2 || line[0] != 'v' || !std::isspace(static_cast<unsigned char>(line[1]))) {
            continue;
        }
        std::istringstream ss(line.substr(1));
        Vec3 p;
        if (ss >> p.x >> p.y >> p.z && IsFiniteVec(p)) out.points.push_back(p);
    }
    if (out.points.empty()) {
        error = "OBJ 文件中没有顶点。";
        return false;
    }
    return true;
}

struct PlyProp {
    std::string type;
    std::string name;
    int size = 0;
    int offset = 0;
};

bool LoadPLYAscii(std::ifstream& ifs, int vertexCount, PointCloud& out, std::string& error) {
    out.points.reserve(static_cast<std::size_t>(vertexCount));
    std::string line;
    for (int i = 0; i < vertexCount; ++i) {
        if (!std::getline(ifs, line)) {
            error = "PLY 文件提前结束。";
            return false;
        }
        std::istringstream ss(line);
        Vec3 p;
        if (!(ss >> p.x >> p.y >> p.z) || !IsFiniteVec(p)) continue;
        out.points.push_back(p);
    }
    if (out.points.empty()) {
        error = "PLY 中没有有效点。";
        return false;
    }
    return true;
}

bool LoadPLYBinaryLE(std::ifstream& ifs, int vertexCount, const std::vector<PlyProp>& props,
                     PointCloud& out, std::string& error) {
    int stride = 0;
    int xOff = -1, yOff = -1, zOff = -1;
    std::string xType, yType, zType;
    for (const PlyProp& p : props) {
        if (p.name == "x") {
            xOff = p.offset;
            xType = p.type;
        } else if (p.name == "y") {
            yOff = p.offset;
            yType = p.type;
        } else if (p.name == "z") {
            zOff = p.offset;
            zType = p.type;
        }
        stride = std::max(stride, p.offset + p.size);
    }
    if (xOff < 0 || yOff < 0 || zOff < 0 || stride < 12) {
        error = "PLY 缺少 x/y/z 属性。";
        return false;
    }

    auto readCoord = [](const char* base, int off, const std::string& type) -> float {
        if (type == "double" || type == "float64") {
            double v = 0.0;
            std::memcpy(&v, base + off, sizeof(double));
            return static_cast<float>(v);
        }
        float v = 0.f;
        std::memcpy(&v, base + off, sizeof(float));
        return v;
    };

    out.points.reserve(static_cast<std::size_t>(vertexCount));
    std::vector<char> buf(static_cast<std::size_t>(stride));
    for (int i = 0; i < vertexCount; ++i) {
        if (!ifs.read(buf.data(), stride)) {
            error = "读取 PLY 二进制顶点失败。";
            return false;
        }
        Vec3 p{readCoord(buf.data(), xOff, xType), readCoord(buf.data(), yOff, yType),
               readCoord(buf.data(), zOff, zType)};
        if (IsFiniteVec(p)) out.points.push_back(p);
    }
    if (out.points.empty()) {
        error = "PLY 中没有有效点。";
        return false;
    }
    return true;
}

bool LoadPLY(const std::string& path, PointCloud& out, std::string& error) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        error = "无法打开文件: " + path;
        return false;
    }

    std::string line;
    if (!std::getline(ifs, line) || line.find("ply") == std::string::npos) {
        error = "不是有效的 PLY 文件。";
        return false;
    }

    bool binary = false;
    int vertexCount = -1;
    bool inVertexProps = false;
    std::vector<PlyProp> props;
    int runningOffset = 0;

    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "format") {
            std::string fmt;
            ss >> fmt;
            binary = (fmt.find("binary_little_endian") != std::string::npos) ||
                     (fmt == "binary");
            if (fmt.find("binary_big_endian") != std::string::npos) {
                error = "暂不支持 big-endian PLY。";
                return false;
            }
        } else if (tag == "element") {
            std::string name;
            ss >> name;
            inVertexProps = (name == "vertex");
            if (inVertexProps) {
                ss >> vertexCount;
                props.clear();
                runningOffset = 0;
            }
        } else if (tag == "property" && inVertexProps) {
            std::string t1;
            ss >> t1;
            if (t1 == "list") {
                // skip list properties on vertex (rare)
                std::string t2, t3, name;
                ss >> t2 >> t3 >> name;
                continue;
            }
            std::string name;
            ss >> name;
            PlyProp prop;
            prop.type = t1;
            prop.name = name;
            prop.size = TypeSize(t1);
            prop.offset = runningOffset;
            runningOffset += prop.size;
            props.push_back(prop);
        } else if (tag == "end_header") {
            break;
        }
    }

    if (vertexCount <= 0) {
        error = "PLY 头缺少顶点数量。";
        return false;
    }

    out.Clear();
    return binary ? LoadPLYBinaryLE(ifs, vertexCount, props, out, error)
                  : LoadPLYAscii(ifs, vertexCount, out, error);
}

bool LoadPCD(const std::string& path, PointCloud& out, std::string& error) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        error = "无法打开文件: " + path;
        return false;
    }

    std::string line;
    int points = -1;
    bool binary = false;
    bool binaryCompressed = false;
    int width = 0, height = 1;
    std::vector<std::string> fields;
    std::vector<int> sizes;
    std::vector<std::string> types;
    std::vector<int> counts;

    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        if (key == "FIELDS") {
            std::string f;
            while (ss >> f) fields.push_back(f);
        } else if (key == "SIZE") {
            int v;
            while (ss >> v) sizes.push_back(v);
        } else if (key == "TYPE") {
            std::string t;
            while (ss >> t) types.push_back(t);
        } else if (key == "COUNT") {
            int v;
            while (ss >> v) counts.push_back(v);
        } else if (key == "WIDTH") {
            ss >> width;
        } else if (key == "HEIGHT") {
            ss >> height;
        } else if (key == "POINTS") {
            ss >> points;
        } else if (key == "DATA") {
            std::string mode;
            ss >> mode;
            binary = (mode == "binary");
            binaryCompressed = (mode == "binary_compressed");
            break;
        }
    }

    if (binaryCompressed) {
        error = "暂不支持压缩 PCD（binary_compressed）。请另存为 ascii 或 binary。";
        return false;
    }

    if (points < 0 && width > 0) points = width * std::max(height, 1);
    if (points <= 0) {
        error = "PCD 头缺少 POINTS。";
        return false;
    }

    if (counts.empty()) counts.assign(fields.size(), 1);
    if (sizes.size() != fields.size() || types.size() != fields.size()) {
        // fallback: assume xyz float
        fields = {"x", "y", "z"};
        sizes = {4, 4, 4};
        types = {"F", "F", "F"};
        counts = {1, 1, 1};
    }

    int pointStep = 0;
    int xOff = -1, yOff = -1, zOff = -1;
    int xSize = 4, ySize = 4, zSize = 4;
    char xType = 'F', yType = 'F', zType = 'F';
    for (std::size_t i = 0; i < fields.size(); ++i) {
        const int c = (i < counts.size()) ? std::max(counts[i], 1) : 1;
        const int s = sizes[i];
        if (fields[i] == "x") {
            xOff = pointStep;
            xSize = s;
            xType = types[i].empty() ? 'F' : types[i][0];
        } else if (fields[i] == "y") {
            yOff = pointStep;
            ySize = s;
            yType = types[i].empty() ? 'F' : types[i][0];
        } else if (fields[i] == "z") {
            zOff = pointStep;
            zSize = s;
            zType = types[i].empty() ? 'F' : types[i][0];
        }
        pointStep += s * c;
    }

    auto readNum = [](const char* p, int size, char type) -> float {
        if (type == 'F' || type == 'f') {
            if (size == 8) {
                double v = 0.0;
                std::memcpy(&v, p, 8);
                return static_cast<float>(v);
            }
            float v = 0.f;
            std::memcpy(&v, p, 4);
            return v;
        }
        if (type == 'U' || type == 'u') {
            if (size == 1) return static_cast<float>(*reinterpret_cast<const uint8_t*>(p));
            if (size == 2) {
                uint16_t v;
                std::memcpy(&v, p, 2);
                return static_cast<float>(v);
            }
            if (size == 4) {
                uint32_t v;
                std::memcpy(&v, p, 4);
                return static_cast<float>(v);
            }
        }
        if (type == 'I' || type == 'i') {
            if (size == 4) {
                int32_t v;
                std::memcpy(&v, p, 4);
                return static_cast<float>(v);
            }
        }
        float v = 0.f;
        std::memcpy(&v, p, sizeof(float));
        return v;
    };

    out.Clear();
    out.points.reserve(static_cast<std::size_t>(points));

    if (!binary) {
        for (int i = 0; i < points; ++i) {
            if (!std::getline(ifs, line)) break;
            Vec3 p;
            if (ParseXYZLine(line, p)) out.points.push_back(p);
        }
    } else {
        if (xOff < 0 || yOff < 0 || zOff < 0 || pointStep <= 0) {
            error = "PCD 缺少 x/y/z 字段。";
            return false;
        }
        std::vector<char> buf(static_cast<std::size_t>(pointStep));
        for (int i = 0; i < points; ++i) {
            if (!ifs.read(buf.data(), pointStep)) {
                error = "读取 PCD 二进制数据失败（可能字段步长不匹配）。";
                return false;
            }
            Vec3 p{readNum(buf.data() + xOff, xSize, xType),
                   readNum(buf.data() + yOff, ySize, yType),
                   readNum(buf.data() + zOff, zSize, zType)};
            if (IsFiniteVec(p)) out.points.push_back(p);
        }
    }

    if (out.points.empty()) {
        error = "PCD 中没有有效点。";
        return false;
    }
    return true;
}

}  // namespace

namespace {

bool IsPointVisible(const PointCloud& cloud, std::size_t i, bool visibleOnly) {
    if (!visibleOnly) return true;
    if (cloud.mask.empty()) return true;
    if (i >= cloud.mask.size()) return true;
    return cloud.mask[i] != 0;
}

bool SaveXYZ(const std::string& path, const PointCloud& cloud, std::string& error,
             bool visibleOnly) {
    std::ofstream ofs(path);
    if (!ofs) {
        error = "无法写入文件: " + path;
        return false;
    }
    ofs << std::setprecision(9);
    std::size_t written = 0;
    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        if (!IsPointVisible(cloud, i, visibleOnly)) continue;
        const Vec3 w = cloud.ToWorld(cloud.points[i]);
        if (!IsFiniteVec(w)) continue;
        ofs << w.x << ' ' << w.y << ' ' << w.z << '\n';
        ++written;
    }
    if (written == 0) {
        error = "没有可保存的点。";
        return false;
    }
    return true;
}

bool SavePLY(const std::string& path, const PointCloud& cloud, std::string& error,
             bool visibleOnly) {
    const bool withColor = cloud.colors.size() == cloud.points.size();
    std::vector<std::size_t> indices;
    indices.reserve(cloud.points.size());
    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        if (!IsPointVisible(cloud, i, visibleOnly)) continue;
        if (!IsFiniteVec(cloud.ToWorld(cloud.points[i]))) continue;
        indices.push_back(i);
    }
    if (indices.empty()) {
        error = "没有可保存的点。";
        return false;
    }

    std::ofstream ofs(path);
    if (!ofs) {
        error = "无法写入文件: " + path;
        return false;
    }
    ofs << "ply\n"
        << "format ascii 1.0\n"
        << "element vertex " << indices.size() << "\n"
        << "property float x\n"
        << "property float y\n"
        << "property float z\n";
    if (withColor) {
        ofs << "property uchar red\n"
            << "property uchar green\n"
            << "property uchar blue\n";
    }
    ofs << "end_header\n";
    ofs << std::setprecision(9);
    for (std::size_t i : indices) {
        const Vec3 w = cloud.ToWorld(cloud.points[i]);
        ofs << w.x << ' ' << w.y << ' ' << w.z;
        if (withColor) {
            const Vec3& c = cloud.colors[i];
            const int r = static_cast<int>(std::clamp(c.x, 0.f, 1.f) * 255.f + 0.5f);
            const int g = static_cast<int>(std::clamp(c.y, 0.f, 1.f) * 255.f + 0.5f);
            const int b = static_cast<int>(std::clamp(c.z, 0.f, 1.f) * 255.f + 0.5f);
            ofs << ' ' << r << ' ' << g << ' ' << b;
        }
        ofs << '\n';
    }
    return true;
}

}  // namespace

namespace PointCloudIO {

bool Load(const std::string& path, PointCloud& out, std::string& error) {
    const std::string ext = ExtensionOf(path);
    bool ok = false;
    if (ext == "xyz" || ext == "txt") {
        ok = LoadXYZ(path, out, error);
    } else if (ext == "obj") {
        ok = LoadOBJ(path, out, error);
    } else if (ext == "ply") {
        ok = LoadPLY(path, out, error);
    } else if (ext == "pcd") {
        ok = LoadPCD(path, out, error);
    } else {
        error = "不支持的格式: ." + ext + "（请使用 PLY / PCD / XYZ / OBJ）";
        return false;
    }

    if (!ok) return false;

    out.sourcePath = path;
    out.RecomputeBounds();
    if (!out.bounds.Valid()) {
        error = "点云包围盒无效（可能全是 NaN）。";
        return false;
    }
    // Center to origin: avoids only showing a "slice" when coordinates are huge.
    out.CenterToOrigin();
    out.ApplyHeightColors(out.bounds.min.z, out.bounds.max.z);
    out.ResetMask();
    return true;
}

bool Save(const std::string& path, const PointCloud& cloud, std::string& error, bool visibleOnly) {
    if (cloud.points.empty()) {
        error = "当前没有点云可保存。";
        return false;
    }
    const std::string ext = ExtensionOf(path);
    if (ext == "xyz" || ext == "txt") {
        return SaveXYZ(path, cloud, error, visibleOnly);
    }
    if (ext == "ply") {
        return SavePLY(path, cloud, error, visibleOnly);
    }
    error = "不支持的保存格式: ." + ext + "（请使用 .ply / .xyz / .txt）";
    return false;
}

bool LoadDepthMaps(const std::string& depthPath, const std::string& brightnessPath,
                   const DepthMapParams& params, PointCloud& out, std::string& error) {
    if (depthPath.empty()) {
        error = "请指定深度图路径。";
        return false;
    }
    if (!(params.pixelSizeX > 0.f) || !(params.pixelSizeY > 0.f)) {
        error = "像素尺寸必须大于 0。";
        return false;
    }
    if (!(params.depthScale != 0.f) || !std::isfinite(params.depthScale)) {
        error = "深度缩放无效。";
        return false;
    }
    const int step = std::max(params.step, 1);

    ImageIO::GrayImage depth;
    if (!ImageIO::LoadGray(depthPath, depth, error)) return false;

    ImageIO::RgbImage bright;
    const bool hasBright = !brightnessPath.empty();
    if (hasBright) {
        if (!ImageIO::LoadRgb(brightnessPath, bright, error)) return false;
        if (bright.width != depth.width || bright.height != depth.height) {
            error = "深度图与亮度图尺寸不一致（" + std::to_string(depth.width) + "x" +
                    std::to_string(depth.height) + " vs " + std::to_string(bright.width) + "x" +
                    std::to_string(bright.height) + "）。";
            return false;
        }
    }

    out.Clear();
    const std::size_t approx =
        static_cast<std::size_t>((depth.width + step - 1) / step) *
        static_cast<std::size_t>((depth.height + step - 1) / step);
    out.points.reserve(approx);
    if (hasBright) out.colors.reserve(approx);

    for (int row = 0; row < depth.height; row += step) {
        const int yRow = params.flipY ? (depth.height - 1 - row) : row;
        for (int col = 0; col < depth.width; col += step) {
            const std::size_t di =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(depth.width) +
                static_cast<std::size_t>(col);
            const float raw = depth.pixels[di];
            if (params.skipNonFinite && !std::isfinite(raw)) continue;
            if (std::fabs(raw - params.invalidValue) <= params.invalidEps) continue;

            const float z = raw * params.depthScale + params.zOffset;
            if (!std::isfinite(z)) continue;

            Vec3 p;
            p.x = static_cast<float>(col) * params.pixelSizeX;
            p.y = static_cast<float>(yRow) * params.pixelSizeY;
            p.z = z;
            out.points.push_back(p);

            if (hasBright) {
                const std::size_t bi = di * 3u;
                out.colors.push_back(Vec3{bright.rgb[bi] / 255.f, bright.rgb[bi + 1] / 255.f,
                                          bright.rgb[bi + 2] / 255.f});
            }
        }
    }

    if (out.points.empty()) {
        error = "深度图中没有有效点（请检查无效值 / 深度缩放）。";
        return false;
    }

    out.sourcePath = depthPath;
    out.RecomputeBounds();
    if (!out.bounds.Valid()) {
        error = "点云包围盒无效。";
        return false;
    }
    out.CenterToOrigin();
    if (!hasBright) {
        out.ApplyHeightColors(out.bounds.min.z, out.bounds.max.z);
    }
    out.ResetMask();
    return true;
}

}  // namespace PointCloudIO
