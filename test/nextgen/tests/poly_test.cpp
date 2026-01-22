//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/common/poly.hpp"

#include <gtest/gtest.h>

#include "kai/kai_common.h"
#include "test/nextgen/common/test_registry.hpp"

namespace kai::test {
namespace {

enum class PolyTestId { BASE, A, B };
inline constexpr size_t pad_size = 8;

class PolyTestBase {
public:
    virtual ~PolyTestBase() = default;
    virtual PolyTestId id() const {
        return PolyTestId::BASE;
    }
    virtual void assign_all() = 0;
    virtual bool verify_match() const = 0;
};

class PolyTestDeriveA : public PolyTestBase {
public:
    static inline int constructed = 0;
    static inline int destroyed = 0;
    std::array<uint8_t, pad_size> padding{};
    int m_key;

    PolyTestDeriveA(int key) : m_key{key} {
        constructed++;
    }

    PolyTestId id() const override {
        return PolyTestId::A;
    }

    void assign_all() override {
        std::fill(std::begin(padding), std::end(padding), 0x01);
    }
    bool verify_match() const override {
        return std::all_of(
            std::begin(padding),  //
            std::end(padding),    //
            [](uint8_t v) { return v == 0x01; });
    }

    ~PolyTestDeriveA() override {
        destroyed++;
    }
};

class PolyTestDeriveB final : public PolyTestBase {
public:
    static inline int constructed = 0;
    static inline int destroyed = 0;
    std::array<uint8_t, pad_size * 2> padding{};
    int m_key;

    PolyTestDeriveB(int key) : m_key{key} {
        constructed++;
    }

    PolyTestId id() const override {
        return PolyTestId::B;
    }

    void assign_all() override {
        std::fill(std::begin(padding), std::end(padding), 0x02);
    }
    bool verify_match() const override {
        return std::all_of(
            std::begin(padding),  //
            std::end(padding),    //
            [](uint8_t v) { return v == 0x02; });
    }

    ~PolyTestDeriveB() override {
        destroyed++;
    }
};

class PolyFixture : public ::testing::Test {
public:
    PolyFixture() = default;
};

class PolyDestructorTest : public PolyFixture {
public:
    void TestBody() override {
        PolyTestDeriveA::destroyed = 0;
        PolyTestDeriveB::destroyed = 0;
        const int key_a = 12;
        const int key_b = 21;

        // Scope lifetime objects under test
        {
            Poly<PolyTestBase> a = kai::test::make_poly<PolyTestDeriveA>(key_a);
            Poly<PolyTestBase> b = kai::test::make_poly<PolyTestDeriveB>(key_b);

            // Cast check
            auto& ref_a = dynamic_cast<PolyTestDeriveA&>(*a);
            auto& ref_b = dynamic_cast<PolyTestDeriveB&>(*b);

            // Check derived type using different accessors
            ASSERT_EQ(a->id(), PolyTestId::A);
            ASSERT_EQ(b->id(), PolyTestId::B);
            ASSERT_EQ(ref_a.id(), PolyTestId::A);
            ASSERT_EQ(ref_b.id(), PolyTestId::B);

            // Check assigned key values
            ASSERT_EQ(ref_a.m_key, key_a);
            ASSERT_EQ(ref_b.m_key, key_b);

            // Assign values to different sized derived types
            a->assign_all();
            b->assign_all();

            // Check different sized derived types populated correctly
            KAI_ASSERT(ref_a.verify_match());
            KAI_ASSERT(ref_b.verify_match());
        }

        // Out of scope
        // Check constructor calls
        ASSERT_EQ(PolyTestDeriveA::constructed, 1);
        ASSERT_EQ(PolyTestDeriveB::constructed, 1);

        // Check destructor calls
        ASSERT_EQ(PolyTestDeriveA::destroyed, 1);
        ASSERT_EQ(PolyTestDeriveB::destroyed, 1);
    }
};

const auto poly_tests_setup = TestRegistry::register_setup([]() {
    const char* test_suite_name = "Poly";
    const std::string test_name = "Poly_test";

    KAI_REGISTER_TEST(PolyFixture, PolyDestructorTest, test_suite_name, test_name.c_str());
});

}  // namespace
}  // namespace kai::test
