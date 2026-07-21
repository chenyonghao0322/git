#include "io/ImageIO.h"

#include <algorithm>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <wincodec.h>

#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "ole32.lib")
#endif

namespace ImageIO {
namespace {

#ifdef _WIN32

struct ComScope {
    bool shouldUninit = false;
    explicit ComScope(std::string& error) {
        const HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
        if (hr == S_OK) {
            shouldUninit = true;
        } else if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
            error = "COM 初始化失败。";
        }
    }
    ~ComScope() {
        if (shouldUninit) CoUninitialize();
    }
};

std::wstring PathToWide(const std::string& path) {
    if (path.empty()) return {};
    // File dialog returns ACP paths on this app; try ACP then UTF-8.
    int n = MultiByteToWideChar(CP_ACP, 0, path.c_str(), -1, nullptr, 0);
    if (n > 0) {
        std::wstring w(static_cast<std::size_t>(n - 1), L'\0');
        MultiByteToWideChar(CP_ACP, 0, path.c_str(), -1, w.data(), n);
        return w;
    }
    n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    if (n <= 0) return {};
    std::wstring w(static_cast<std::size_t>(n - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, w.data(), n);
    return w;
}

bool CreateDecoder(const std::string& path, IWICImagingFactory* factory,
                   IWICBitmapDecoder** decoder, std::string& error) {
    const std::wstring wpath = PathToWide(path);
    if (wpath.empty()) {
        error = "路径编码转换失败: " + path;
        return false;
    }
    const HRESULT hr =
        factory->CreateDecoderFromFilename(wpath.c_str(), nullptr, GENERIC_READ,
                                           WICDecodeMetadataCacheOnLoad, decoder);
    if (FAILED(hr)) {
        error = "无法打开图像: " + path;
        return false;
    }
    return true;
}

bool ConvertTo(IWICImagingFactory* factory, IWICBitmapFrameDecode* frame,
               REFWICPixelFormatGUID fmt, IWICBitmapSource** out, std::string& error) {
    IWICFormatConverter* conv = nullptr;
    HRESULT hr = factory->CreateFormatConverter(&conv);
    if (FAILED(hr) || !conv) {
        error = "无法创建格式转换器。";
        return false;
    }
    hr = conv->Initialize(frame, fmt, WICBitmapDitherTypeNone, nullptr, 0.0,
                          WICBitmapPaletteTypeCustom);
    if (FAILED(hr)) {
        conv->Release();
        error = "图像格式转换失败。";
        return false;
    }
    *out = conv;
    return true;
}

#endif

}  // namespace

bool LoadGray(const std::string& path, GrayImage& out, std::string& error) {
    out = {};
#ifndef _WIN32
    (void)path;
    error = "当前平台暂不支持图像读取。";
    return false;
#else
    ComScope com(error);
    if (!error.empty()) return false;

    IWICImagingFactory* factory = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                  IID_IWICImagingFactory, reinterpret_cast<void**>(&factory));
    if (FAILED(hr) || !factory) {
        error = "无法创建 WIC ImagingFactory。";
        return false;
    }

    IWICBitmapDecoder* decoder = nullptr;
    if (!CreateDecoder(path, factory, &decoder, error)) {
        factory->Release();
        return false;
    }

    IWICBitmapFrameDecode* frame = nullptr;
    hr = decoder->GetFrame(0, &frame);
    if (FAILED(hr) || !frame) {
        decoder->Release();
        factory->Release();
        error = "读取图像帧失败。";
        return false;
    }

    UINT w = 0, h = 0;
    if (FAILED(frame->GetSize(&w, &h)) || w == 0 || h == 0) {
        frame->Release();
        decoder->Release();
        factory->Release();
        error = "无效的图像尺寸。";
        return false;
    }

    WICPixelFormatGUID srcFmt{};
    frame->GetPixelFormat(&srcFmt);

    out.width = static_cast<int>(w);
    out.height = static_cast<int>(h);
    out.pixels.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));

    bool ok = false;
    if (IsEqualGUID(srcFmt, GUID_WICPixelFormat16bppGray)) {
        const UINT stride = w * 2;
        std::vector<uint16_t> buf(out.pixels.size());
        ok = SUCCEEDED(frame->CopyPixels(nullptr, stride,
                                         static_cast<UINT>(buf.size() * sizeof(uint16_t)),
                                         reinterpret_cast<BYTE*>(buf.data())));
        if (ok) {
            for (std::size_t i = 0; i < buf.size(); ++i) out.pixels[i] = static_cast<float>(buf[i]);
        }
    } else if (IsEqualGUID(srcFmt, GUID_WICPixelFormat8bppGray)) {
        const UINT stride = w;
        std::vector<uint8_t> buf(out.pixels.size());
        ok = SUCCEEDED(frame->CopyPixels(nullptr, stride, static_cast<UINT>(buf.size()), buf.data()));
        if (ok) {
            for (std::size_t i = 0; i < buf.size(); ++i) out.pixels[i] = static_cast<float>(buf[i]);
        }
    } else {
        // Prefer 16-bit gray for depth maps (PNG/TIFF 16-bit etc.).
        IWICBitmapSource* converted = nullptr;
        if (ConvertTo(factory, frame, GUID_WICPixelFormat16bppGray, &converted, error)) {
            const UINT stride = w * 2;
            std::vector<uint16_t> buf(out.pixels.size());
            ok = SUCCEEDED(converted->CopyPixels(nullptr, stride,
                                                 static_cast<UINT>(buf.size() * sizeof(uint16_t)),
                                                 reinterpret_cast<BYTE*>(buf.data())));
            converted->Release();
            if (ok) {
                for (std::size_t i = 0; i < buf.size(); ++i)
                    out.pixels[i] = static_cast<float>(buf[i]);
                error.clear();
            } else {
                error = "读取 16 位灰度像素失败。";
            }
        }
        if (!ok) {
            error.clear();
            IWICBitmapSource* converted8 = nullptr;
            if (ConvertTo(factory, frame, GUID_WICPixelFormat8bppGray, &converted8, error)) {
                const UINT stride = w;
                std::vector<uint8_t> buf(out.pixels.size());
                ok = SUCCEEDED(
                    converted8->CopyPixels(nullptr, stride, static_cast<UINT>(buf.size()), buf.data()));
                converted8->Release();
                if (ok) {
                    for (std::size_t i = 0; i < buf.size(); ++i)
                        out.pixels[i] = static_cast<float>(buf[i]);
                    error.clear();
                } else {
                    error = "读取 8 位灰度像素失败。";
                }
            }
        }
    }

    frame->Release();
    decoder->Release();
    factory->Release();
    if (!ok && error.empty()) error = "读取灰度图像失败。";
    return ok;
#endif
}

bool LoadRgb(const std::string& path, RgbImage& out, std::string& error) {
    out = {};
#ifndef _WIN32
    (void)path;
    error = "当前平台暂不支持图像读取。";
    return false;
#else
    ComScope com(error);
    if (!error.empty()) return false;

    IWICImagingFactory* factory = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                  IID_IWICImagingFactory, reinterpret_cast<void**>(&factory));
    if (FAILED(hr) || !factory) {
        error = "无法创建 WIC ImagingFactory。";
        return false;
    }

    IWICBitmapDecoder* decoder = nullptr;
    if (!CreateDecoder(path, factory, &decoder, error)) {
        factory->Release();
        return false;
    }

    IWICBitmapFrameDecode* frame = nullptr;
    hr = decoder->GetFrame(0, &frame);
    if (FAILED(hr) || !frame) {
        decoder->Release();
        factory->Release();
        error = "读取图像帧失败。";
        return false;
    }

    UINT w = 0, h = 0;
    if (FAILED(frame->GetSize(&w, &h)) || w == 0 || h == 0) {
        frame->Release();
        decoder->Release();
        factory->Release();
        error = "无效的图像尺寸。";
        return false;
    }

    IWICBitmapSource* converted = nullptr;
    if (!ConvertTo(factory, frame, GUID_WICPixelFormat24bppRGB, &converted, error)) {
        frame->Release();
        decoder->Release();
        factory->Release();
        return false;
    }

    out.width = static_cast<int>(w);
    out.height = static_cast<int>(h);
    out.rgb.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3u);
    const UINT stride = w * 3;
    const bool ok =
        SUCCEEDED(converted->CopyPixels(nullptr, stride, static_cast<UINT>(out.rgb.size()),
                                        out.rgb.data()));
    converted->Release();
    frame->Release();
    decoder->Release();
    factory->Release();
    if (!ok) {
        error = "读取 RGB 像素失败。";
        out = {};
        return false;
    }
    return true;
#endif
}

}  // namespace ImageIO
