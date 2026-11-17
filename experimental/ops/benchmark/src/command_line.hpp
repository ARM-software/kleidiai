//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>
#include <string>

// Reads benchmark options without platform-specific getopt APIs. A colon in
// short_options marks a required argument; all long options require arguments.
class command_line {
public:
    struct long_option {
        const char *name;
        int code;
    };

    command_line(int argc, char **argv, const char *short_options, const long_option *long_options)
        : argc_(argc), argv_(argv), short_options_(short_options), long_options_(long_options) {}

    // Returns an option code, '?' on error, or -1 after all arguments are read.
    int next() {
        argument_ = nullptr;
        for (;;) {
            if (pending_ && *pending_) {
                const char option = *pending_++;
                const char *spec = std::strchr(short_options_, option);
                if (!spec || option == ':') {
                    return fail(std::string("Unknown option: -") + option);
                }
                if (spec[1] == ':') {
                    if (*pending_) {
                        argument_ = pending_;
                    } else if (index_ < argc_) {
                        argument_ = argv_[index_++];
                    } else {
                        return fail(std::string("Missing argument for -") + option);
                    }
                    pending_ = nullptr;
                }
                return option;
            }

            if (index_ == argc_) {
                return -1;
            }
            char *token = argv_[index_++];
            if (!options_ended_ && std::strcmp(token, "--") == 0) {
                options_ended_ = true;
            } else if (!options_ended_ && token[0] == '-' && token[1]) {
                if (token[1] == '-') {
                    return read_long_option(token + 2);
                }
                pending_ = token + 1;
            } else if (!kernel_) {
                // Continue scanning so options may also follow the kernel name.
                kernel_ = token;
            }
        }
    }

    char *argument() const { return argument_; }
    const char *kernel() const { return kernel_; }
    const char *error() const { return error_.c_str(); }

private:
    int read_long_option(char *name) {
        char *equals = std::strchr(name, '=');
        const size_t length = equals ? static_cast<size_t>(equals - name) : std::strlen(name);
        const long_option *selected = nullptr;
        bool ambiguous = false;
        for (const long_option *option = long_options_; option->name; ++option) {
            if (std::strncmp(option->name, name, length) != 0) {
                continue;
            }
            if (std::strlen(option->name) == length) {
                selected = option;
                ambiguous = false;
                break;
            }
            ambiguous = selected != nullptr;
            selected = option;
        }
        if (!selected || length == 0) {
            return fail(std::string("Unknown option: --") + name);
        }
        if (ambiguous) {
            return fail(std::string("Ambiguous option: --") + name);
        }
        if (equals) {
            argument_ = equals + 1;
        } else if (index_ < argc_) {
            argument_ = argv_[index_++];
        } else {
            return fail(std::string("Missing argument for --") + name);
        }
        return selected->code;
    }

    int fail(const std::string &message) {
        error_ = message;
        return '?';
    }

    int argc_;
    char **argv_;
    const char *short_options_;
    const long_option *long_options_;
    int index_ = 1;
    char *pending_ = nullptr;
    char *argument_ = nullptr;
    const char *kernel_ = nullptr;
    bool options_ended_ = false;
    std::string error_;
};
