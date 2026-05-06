//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include <inttypes.h>

#include "utils.hpp"

#include "gemm_lib.hpp"

enum class ConvImpl {
    DEFAULT,
    IM2ROW,
    IM2COL,
    WINOGRAD,
    CONV2D
};

struct impl_names {
    std::string name;
    ConvImpl    impl;
};

impl_names theimpls[] = { { "im2row",      ConvImpl::IM2ROW },
                          { "im2col",      ConvImpl::IM2COL },
                          { "winograd",    ConvImpl::WINOGRAD },
                          { "conv2d",      ConvImpl::CONV2D } };

const char *impl2str(ConvImpl impl) {
    for (auto &&i : theimpls) {
        if (i.impl == impl) {
            return i.name.c_str();
        }
    }

    return "UNKNOWN";
}

ConvImpl str2impl(std::string name) {
    for (auto &&i : theimpls) {
        if (i.name == name) {
            return i.impl;
        }
    }

    // 'DEFAULT' means error in this context.
    return ConvImpl::DEFAULT;
}

class test_suite {
public:
    std::string	    shortname;

    test_suite(std::string sn) : shortname(sn) { }

    virtual bool get_config(GemmProblem *p, unsigned int ID, std::string implname, bool winograd_ok, bool conv_ok) = 0;
    virtual void print_info(bool) = 0;

    virtual ~test_suite() { }
};

#define RANDOM_FLAG_WINOGRAD   0x1
#define RANDOM_FLAG_DEPTHWISE  0x2

class random_suite : public test_suite {
public:
    random_suite(const random_suite &)           = delete;
    random_suite operator=(const random_suite &) = delete;

    uint32_t flags;

    random_suite(std::string name, uint32_t flags) : test_suite(name), flags(flags) { }

    bool get_config(GemmProblem *p, unsigned int ID, std::string, bool, bool) override {
        std::mt19937 eng(ID);

        // Transcribed from the python used to generate the original random suite
        // Start off by generating everything at random.
        p->multis = 1;
        p->batches = (eng() % 10) + 1;

        p->kernel_height = (eng() % 5) * 2 + 1; // [1, 3, 5, 7, 9]
        p->kernel_width = (eng() % 5) * 2 + 1;

        p->out_stride_h = eng() % 3 + 1;
        p->out_stride_w = eng() % 3 + 1;

        p->in_stride_h = eng() % 3 + 1;
        p->in_stride_w = eng() % 3 + 1;

        p->input_channels = ((eng() % 256) + 1);
        p->output_channels = ((eng() % 256) + 1);

        const bool same_padding=eng()%2;

        p->output_height = eng() % 50 + 1;
        p->output_width = eng() % 50 + 1;

        // Constrain for depthwise if needed - this should be updated as the capabilities of our depthwise kernels
        // increases.
        if (flags & RANDOM_FLAG_DEPTHWISE) {
            // Make an above-average number of cases hit the optimized kernels.
            // Remove any dilation set as it gets readded in case 6 below.
            p->in_stride_h = p->in_stride_w = 1;

            switch(eng() % 7) {
                // 3x3s2 - versions without (0),  and with (1), multiplier
                case 0:
                    p->output_channels = p->input_channels;
                    /* fall through */
                case 1:
                    p->kernel_height = p->kernel_width = 3;
                    p->out_stride_h = p->out_stride_w = 2;
                    break;
                // 3x3s1 - no multipler only
                case 2:
                    p->kernel_height = p->kernel_width = 3;
                    p->out_stride_h = p->out_stride_w = 1;
                    p->output_channels = p->input_channels;
                    break;
                // 5x5s1 - versions without (3), and with (4), multiplier
                case 3:
                    p->output_channels = p->input_channels;
                    /* fall through */
                case 4:
                    p->kernel_height = p->kernel_width = 5;
                    p->out_stride_h = p->out_stride_w = 1;
                    break;
                // 5x5s2 - no multiplier only
                case 5:
                    p->kernel_height = p->kernel_width = 5;
                    p->out_stride_h = p->out_stride_w = 2;
                    break;
                // Random and with dilation
                case 6:
                    p->in_stride_h = eng() % 5 + 1;
                    p->in_stride_w = eng() % 5 + 1;
                    break;

                // Let the odd random case through for the generic kernels.
                default:
                    break;
            }

            int64_t multiplier = iceildiv(p->output_channels, p->input_channels);

            p->output_channels = p->input_channels * std::max<int64_t>(1, multiplier);
            p->groups = p->input_channels;
        }

        // Constrain for Winograd
        if (flags & RANDOM_FLAG_WINOGRAD) {
            // Winograd never supports a stride or dilation.
            p->out_stride_h = p->out_stride_w = 1;
            p->in_stride_h = p->in_stride_w = 1;

            // Kernel height/width has to match a supported Winograd configuration.
            switch(eng() % 8) {
                // 3x3
                case 0:
                    p->kernel_height = p->kernel_width = 3;
                    break;
                // 3x1
                case 1:
                    p->kernel_height = 3;
                    p->kernel_width = 1;
                    break;
                // 1x3
                case 2:
                    p->kernel_height = 1;
                    p->kernel_width = 3;
                    break;
                // 5x5
                case 3:
                    p->kernel_height = p->kernel_width = 5;
                    break;
                // 5x1
                case 4:
                    p->kernel_height = 5;
                    p->kernel_width = 1;
                    break;
                // 1x5
                case 5:
                    p->kernel_height = 1;
                    p->kernel_width = 5;
                    break;
                // 7x1
                case 6:
                    p->kernel_height = 7;
                    p->kernel_width = 1;
                    break;
                // 1x7
                case 7:
                    p->kernel_height = 1;
                    p->kernel_width = 7;
                    break;
            }
        }

        // Compute input dimensions and padding.
        p->input_height = ((p->output_height - 1) * p->out_stride_h) + 1;
        p->input_width = ((p->output_width - 1) * p->out_stride_w) + 1;

        auto effective_kernel_height = (p->kernel_height - 1)*p->in_stride_h + 1;
        auto effective_kernel_width = (p->kernel_width - 1)*p->in_stride_w + 1;

        if (same_padding) {
            p->padding_top = (effective_kernel_height - 1) / 2;
            p->padding_left = (effective_kernel_width - 1) / 2;
        } else {
            p->input_height += effective_kernel_height - 1;
            p->input_width += effective_kernel_width - 1;
        }

#ifndef SILENT
        printf ("===============================================================================\n");
        printf ("Test configuration:\n");
        printf (" Suite:       %s\n", this->shortname.c_str());
        printf (" Test ID:     %d\n", ID);
        printf (" Convolution: filter %" PRId64 "x%" PRId64 ", stride %" PRId64 "x%" PRId64 ", ",
                p->kernel_height, p->kernel_width, p->out_stride_h, p->out_stride_w);
        if (p->in_stride_h != 1 || p->in_stride_w != 1)
        {
            printf ("dilation %" PRId64 "x%" PRId64 ", ", p->in_stride_h, p->in_stride_w);
        }
        printf ("output %" PRId64 "x%" PRId64 " (%" PRId64 " -> %" PRId64 " channels)\n",
                p->output_height, p->output_width, p->input_channels, p->output_channels);
        printf ("===============================================================================\n");
#endif

        return true;
    }

    void print_info(bool first) override {
        // We actually support UINT_MAX configurations, but scripts might try and parse a number here so pick a more reasonable number.
        printf("\t%s (1000000 configurations)%s\n", this->shortname.c_str(), first ? ", default" : "");
    }
};

std::vector<std::shared_ptr<test_suite> > thesuites = {
                                                           std::make_shared<random_suite>   ("random_conv2d",    0                     ),
                                                           std::make_shared<random_suite>   ("random_winograd",  RANDOM_FLAG_WINOGRAD  ),
                                                           std::make_shared<random_suite>   ("random_depthwise", RANDOM_FLAG_DEPTHWISE ),
                                                      };

void print_test_suites() {
    printf ("\n");
    printf ("List of avaliable test suites:\n");

    bool first=true;
    for (auto &&i : thesuites) {
        i->print_info(first);
        first=false;
    }

    printf ("\n");
    printf ("List of available convolution implementations:\n");
    for (auto &&i : theimpls) {
        printf("\t%s\n", i.name.c_str());
    }
}

bool get_test_config(GemmProblem *p, std::string suitename, unsigned int ID, std::string implname, bool wg_ok, bool conv_ok) {
    test_suite *thesuite = thesuites[0].get();

    if (suitename != "") {
        thesuite = nullptr;

        for (auto &&i : thesuites) {
            if (suitename == i->shortname) {
                thesuite = i.get();
                break;
            }
        }

        if (thesuite == nullptr) {
            printf("Unknown test suite: %s\n", suitename.c_str());
            return false;
        }
    }

    return thesuite->get_config(p, ID, implname, wg_ok, conv_ok);
}
