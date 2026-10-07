//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cctype>
#include <charconv>
#include <cstddef>
#include <optional>
#include <string_view>
#include <vector>

namespace kai::test {

/// Removes leading and trailing whitespace.
///
/// @param[in] text Text to trim.
/// @return A view of `text` without leading or trailing whitespace.
inline std::string_view trimmed(std::string_view text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())) != 0) {
        text.remove_prefix(1);
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0) {
        text.remove_suffix(1);
    }
    return text;
}

/// Splits text using a delimiter.
///
/// Empty parts are preserved. An empty delimiter returns `text` as the only part.
///
/// @param[in] text Text to split.
/// @param[in] delimiter Delimiter separating the parts.
/// @return Non-owning views of the parts in `text`.
inline std::vector<std::string_view> split(std::string_view text, std::string_view delimiter) {
    std::vector<std::string_view> parts;
    if (delimiter.empty()) {
        parts.emplace_back(text);
        return parts;
    }

    size_t start = 0;
    while (true) {
        const size_t pos = text.find(delimiter, start);
        const size_t length = (pos == std::string_view::npos) ? std::string_view::npos : pos - start;
        parts.emplace_back(text.substr(start, length));
        if (pos == std::string_view::npos) {
            break;
        }
        start = pos + delimiter.size();
    }
    return parts;
}

/// Parses a number from text.
template <typename T>
std::optional<T> parse_num(std::string_view text) {
    if (text.empty()) {
        return std::nullopt;
    }

    T parsed{};
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto [ptr, ec] = std::from_chars(begin, end, parsed);

    if (ec != std::errc() || ptr != end) {
        return std::nullopt;
    }
    return parsed;
}

}  // namespace kai::test
