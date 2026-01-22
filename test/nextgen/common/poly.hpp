//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace kai::test {

namespace internal {

class PolyControl {
public:
    virtual ~PolyControl() = default;

    template <typename Derived>
    explicit PolyControl(std::in_place_type_t<Derived> type);

    template <typename Base>
    [[nodiscard]] const Base& data() const;

    template <typename Base>
    [[nodiscard]] Base& data();

    [[nodiscard]] std::unique_ptr<PolyControl> clone() const;

private:
    std::unique_ptr<PolyControl> (*m_clone_fn)(const PolyControl& inner);
    const void* (*m_get_constant_fn)(const PolyControl& inner) = nullptr;
    void* (*m_get_mutable_fn)(PolyControl& inner) = nullptr;
};

template <typename Derived>
struct PolyInner : public PolyControl {
    template <typename... Args>
    explicit PolyInner(Args&&... args);

    Derived data;
};

template <typename Derived>
inline PolyControl::PolyControl([[maybe_unused]] std::in_place_type_t<Derived> type) :
    m_clone_fn([](const PolyControl& inner) -> std::unique_ptr<PolyControl> {
        auto& self = static_cast<const PolyInner<Derived>&>(inner);
        return std::make_unique<PolyInner<Derived>>(self.data);
    }),
    m_get_constant_fn(+[](const PolyControl& inner) -> const void* {
        auto& self = static_cast<const PolyInner<Derived>&>(inner);
        return &self.data;
    }),
    m_get_mutable_fn(+[](PolyControl& inner) -> void* {
        auto& self = static_cast<PolyInner<Derived>&>(inner);
        return &self.data;
    }) {
}

template <typename Base>
inline const Base& PolyControl::data() const {
    return *static_cast<const Base*>(m_get_constant_fn(*this));
}

template <typename Base>
[[nodiscard]] inline Base& PolyControl::data() {
    return *static_cast<Base*>(m_get_mutable_fn(*this));
}

[[nodiscard]] inline std::unique_ptr<PolyControl> PolyControl::clone() const {
    return m_clone_fn(*this);
}

template <typename Derived>
template <typename... Args>
inline PolyInner<Derived>::PolyInner(Args&&... args) :
    PolyControl(std::in_place_type<Derived>), data(std::forward<Args>(args)...) {
}

}  // namespace internal

/// Copyable unique pointer.
///
/// This class is similar to @ref std::polymorphic from C++23.
template <typename Base>
class Poly {
    template <typename Derived>
    friend class Poly;

public:
    Poly() = default;

    template <typename Derived, typename... Args>
    explicit Poly([[maybe_unused]] std::in_place_type_t<Derived> type, Args&&... args) :
        m_wrapper(std::make_unique<internal::PolyInner<Derived>>(std::forward<Args>(args)...)) {
    }

    Poly(const Poly& other) : m_wrapper(other.m_wrapper->clone()) {
    }

    Poly& operator=(const Poly& other) {
        if (this != &other) {
            m_wrapper = other.m_wrapper->clone();
        }

        return *this;
    }

    template <
        typename Derived,
        std::enable_if_t<std::is_base_of_v<Base, Derived> && std::is_convertible_v<const Derived*, const Base*>, bool> =
            true>
    Poly(Poly<Derived>&& other) : m_wrapper(std::move(other.m_wrapper)) {
    }

    template <
        typename Derived,
        std::enable_if_t<std::is_base_of_v<Base, Derived> && std::is_convertible_v<const Derived*, const Base*>, bool> =
            true>
    Poly& operator=(Poly<Derived>&& other) {
        m_wrapper = std::move(other.m_wrapper);
        return *this;
    }

    [[nodiscard]] const Base* operator->() const {
        return &m_wrapper->template data<Base>();
    }

    [[nodiscard]] Base* operator->() {
        return &m_wrapper->template data<Base>();
    }

    [[nodiscard]] const Base& operator*() const {
        return m_wrapper->template data<Base>();
    }

    [[nodiscard]] Base& operator*() {
        return m_wrapper->template data<Base>();
    }

private:
    std::unique_ptr<internal::PolyControl> m_wrapper;
};

template <typename T, typename... Args>
[[nodiscard]] Poly<T> make_poly(Args&&... args) {
    return Poly<T>(std::in_place_type<T>, std::forward<Args>(args)...);
}

}  // namespace kai::test
