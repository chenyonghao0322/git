#pragma once

#include <string>

class Shader {
public:
    bool Build(const char* vsSrc, const char* fsSrc, std::string& error);
    void Use() const;
    void SetMat4(const char* name, const float* m16) const;
    void SetFloat(const char* name, float v) const;
    void SetVec3(const char* name, float x, float y, float z) const;
    void SetInt(const char* name, int v) const;
    unsigned int Id() const { return program_; }

private:
    unsigned int program_ = 0;
};
