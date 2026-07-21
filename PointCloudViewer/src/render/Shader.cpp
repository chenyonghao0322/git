#include "render/Shader.h"

#include <glad/gl.h>

#include <algorithm>
#include <vector>

bool Shader::Build(const char* vsSrc, const char* fsSrc, std::string& error) {
    auto compile = [&](unsigned int type, const char* src) -> unsigned int {
        const unsigned int shader = glCreateShader(type);
        glShaderSource(shader, 1, &src, nullptr);
        glCompileShader(shader);
        int ok = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            int len = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
            std::vector<char> log(static_cast<std::size_t>(std::max(len, 1)));
            glGetShaderInfoLog(shader, len, nullptr, log.data());
            error = log.data();
            glDeleteShader(shader);
            return 0;
        }
        return shader;
    };

    const unsigned int vs = compile(GL_VERTEX_SHADER, vsSrc);
    if (!vs) return false;
    const unsigned int fs = compile(GL_FRAGMENT_SHADER, fsSrc);
    if (!fs) {
        glDeleteShader(vs);
        return false;
    }

    program_ = glCreateProgram();
    glAttachShader(program_, vs);
    glAttachShader(program_, fs);
    glLinkProgram(program_);
    glDeleteShader(vs);
    glDeleteShader(fs);

    int ok = 0;
    glGetProgramiv(program_, GL_LINK_STATUS, &ok);
    if (!ok) {
        int len = 0;
        glGetProgramiv(program_, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(static_cast<std::size_t>(std::max(len, 1)));
        glGetProgramInfoLog(program_, len, nullptr, log.data());
        error = log.data();
        glDeleteProgram(program_);
        program_ = 0;
        return false;
    }
    return true;
}

void Shader::Use() const { glUseProgram(program_); }

void Shader::SetMat4(const char* name, const float* m16) const {
    glUniformMatrix4fv(glGetUniformLocation(program_, name), 1, GL_FALSE, m16);
}

void Shader::SetFloat(const char* name, float v) const {
    glUniform1f(glGetUniformLocation(program_, name), v);
}

void Shader::SetVec3(const char* name, float x, float y, float z) const {
    glUniform3f(glGetUniformLocation(program_, name), x, y, z);
}

void Shader::SetInt(const char* name, int v) const {
    glUniform1i(glGetUniformLocation(program_, name), v);
}
