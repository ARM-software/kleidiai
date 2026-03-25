//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>
#include <variant>

#include "test/common/assert.hpp"

template <typename OkT, typename ErrT>
struct Expected {
private:
    std::variant<OkT, ErrT> m_data;

public:
    Expected(const OkT& v) : m_data(v) {
    }
    Expected(OkT&& v) : m_data(std::move(v)) {
    }

    Expected(const ErrT& e) : m_data(e) {
    }
    Expected(ErrT&& e) : m_data(std::move(e)) {
    }

    bool has_value() const {
        return std::holds_alternative<OkT>(m_data);
    }

    OkT& value() & {
        KAI_ASSUME(has_value());
        return std::get<OkT>(m_data);
    }

    const OkT& value() const& {
        KAI_ASSUME(has_value());
        return std::get<OkT>(m_data);
    }

    OkT&& value() && {
        KAI_ASSUME(has_value());
        return std::get<OkT>(std::move(m_data));
    }

    ErrT& error() & {
        KAI_ASSUME(!has_value());
        return std::get<ErrT>(m_data);
    }

    const ErrT& error() const& {
        KAI_ASSUME(!has_value());
        return std::get<ErrT>(m_data);
    }

    ErrT&& error() && {
        KAI_ASSUME(!has_value());
        return std::get<ErrT>(std::move(m_data));
    }
};
