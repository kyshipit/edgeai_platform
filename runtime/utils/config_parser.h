/*
utils/config_parser.h（header-only）配置解析器

*/

#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <cctype>
#include <algorithm>
#include <stdexcept>

// 配置节点类型
enum class ConfigType {
    Null,
    String,
    Number,
    Array,
    Object
};

// 配置节点类（header-only 实现）
class ConfigParser {
private:
    ConfigType type_ = ConfigType::Null;
    std::string str_val_;
    double num_val_ = 0.0;
    std::vector<ConfigParser> arr_val_;
    std::map<std::string, ConfigParser> children_;

    // -------------- 工具函数 --------------
    // 去除字符串首尾空格
    static std::string trim(const std::string& s) {
        auto start = s.begin();
        while (start != s.end() && std::isspace(static_cast<unsigned char>(*start))) start++;
        auto end = s.end();
        do { end--; } while (std::distance(start, end) > 0 && std::isspace(static_cast<unsigned char>(*end)));
        return std::string(start, end + 1);
    }

    // 按字符分割字符串
    static std::vector<std::string> split(const std::string& s, char delim) {
        std::vector<std::string> res;
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, delim)) {
            item = trim(item);
            if (!item.empty()) res.push_back(item);
        }
        return res;
    }

    // 判断是否为数字
    static bool is_number(const std::string& s) {
        if (s.empty()) return false;
        char* end = nullptr;
        strtod(s.c_str(), &end);
        return end == s.c_str() + s.size();
    }

    // 解析值（字符串/数字/数组）
    void parse_value(const std::string& val_str) {
        std::string val = trim(val_str);
        if (val.empty()) {
            type_ = ConfigType::Null;
            return;
        }

        // 解析数组 [1,2,3]
        if (val.front() == '[' && val.back() == ']') {
            type_ = ConfigType::Array;
            std::string content = val.substr(1, val.size() - 2);
            auto elements = split(content, ',');
            for (auto& elem : elements) {
                ConfigParser node;
                node.parse_value(elem);
                arr_val_.push_back(node);
            }
            return;
        }

        // 解析数字
        if (is_number(val)) {
            type_ = ConfigType::Number;
            num_val_ = std::stod(val);
            return;
        }

        // 解析字符串
        type_ = ConfigType::String;
        str_val_ = val;
    }

    // -------------- 核心：按 . 分割键（如 model.type → [model, type]） --------------
    std::vector<std::string> split_dot_key(const std::string& key) {
        return split(key, '.');
    }

    // -------------- 根据键路径查找节点 --------------
    ConfigParser* find_node(const std::string& key_path) {
        auto keys = split_dot_key(key_path);
        ConfigParser* node = this;
        for (auto& k : keys) {
            auto it = node->children_.find(k);
            if (it == node->children_.end()) return nullptr; // 任意一层不存在，返回空
            node = &(it->second);
        }
        return node;
    }

public:
    // -------------- 你要求的核心接口 --------------
    // 1. 从文件加载配置
    bool LoadFromFile(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "[Error] 无法打开文件: " << path << std::endl;
            return false;
        }

        std::vector<ConfigParser*> stack;
        stack.push_back(this);
        std::string line;

        while (std::getline(file, line)) {
            std::string trim_line = trim(line);
            if (trim_line.empty() || trim_line[0] == '#') continue;

            // 计算缩进
            int indent = 0;
            while (indent < line.size() && std::isspace(static_cast<unsigned char>(line[indent]))) indent++;
            std::string content = line.substr(indent);

            // 分割 key: value
            size_t colon = content.find(':');
            if (colon == std::string::npos) continue;

            std::string key = trim(content.substr(0, colon));
            std::string val = trim(content.substr(colon + 1));

            // 调整嵌套层级
            while (stack.size() > 1 && indent <= (int)(stack.size() - 2) * 4) {
                stack.pop_back();
            }

            ConfigParser& child = stack.back()->children_[key];
            if (!val.empty()) {
                child.parse_value(val);
            } else {
                stack.push_back(&child);
            }
        }

        file.close();
        return true;
    }

    // 2. 获取字符串（支持 . 嵌套，带默认值）
    std::string GetString(const std::string& key, const std::string& default_val = "") {
        ConfigParser* node = find_node(key);
        if (!node || node->type_ != ConfigType::String) return default_val;
        return node->str_val_;
    }

    // 3. 获取整数（支持 . 嵌套，带默认值）
    int GetInt(const std::string& key, int default_val = 0) {
        ConfigParser* node = find_node(key);
        if (!node || node->type_ != ConfigType::Number) return default_val;
        return static_cast<int>(node->num_val_);
    }

    // -------------- 扩展：获取数组（保留原有功能） --------------
    std::vector<ConfigParser> GetArray(const std::string& key) {
        ConfigParser* node = find_node(key);
        if (!node || node->type_ != ConfigType::Array) return {};
        return node->arr_val_;
    }

    // 辅助：获取数组int值
    std::vector<int> GetIntArray(const std::string& key) {
        std::vector<int> res;
        auto arr = GetArray(key);
        for (auto& n : arr) res.push_back(n.as_int());
        return res;
    }

    // 内部值获取
    int as_int() const { return static_cast<int>(num_val_); }
};