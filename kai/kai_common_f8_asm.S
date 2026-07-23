//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if defined(_MSC_VER)
    #define KAI_ASM_GLOBAL(name) GLOBAL name
    #define KAI_ASM_FUNCTION_TYPE(name)
    #define KAI_ASM_FUNCTION_LABEL(name) name PROC
    #define KAI_ASM_FUNCTION_END(name) ENDP
    #define KAI_ASM_FUNCTION_USE(name) name

    #define KAI_ASM_CODE(name) AREA name, CODE, READONLY
    #define KAI_ASM_ALIGN
    #define KAI_ASM_LABEL(name) name
    #define KAI_ASM_INST(hex) DCD hex
    #define KAI_ASM_END END
#else
    #if defined(__APPLE__)
        #define KAI_ASM_GLOBAL(name) .globl _##name
        #define KAI_ASM_FUNCTION_TYPE(name)
        #define KAI_ASM_FUNCTION_LABEL(name) _##name:
        #define KAI_ASM_FUNCTION_END(name)
        #define KAI_ASM_FUNCTION_USE(name) _##name
    #else
        #define KAI_ASM_GLOBAL(name) .global name
        #define KAI_ASM_FUNCTION_TYPE(name) .type name, %function
        #define KAI_ASM_FUNCTION_LABEL(name) name:
        #define KAI_ASM_FUNCTION_END(name) .size name, .-name
        #define KAI_ASM_FUNCTION_USE(name) name
    #endif

    #define KAI_ASM_CODE(name) .text
    #define KAI_ASM_ALIGN .p2align 4,,11
    #define KAI_ASM_LABEL(name) name:
    #define KAI_ASM_INST(hex) .inst hex
    #define KAI_ASM_END
#endif


    KAI_ASM_CODE(kai_common)
    KAI_ASM_ALIGN

    // Declare functions
    KAI_ASM_GLOBAL(kai_write_fpmr_raw)
    KAI_ASM_GLOBAL(kai_read_fpmr_raw)
    KAI_ASM_GLOBAL(kai_convert_f8_f32x4x2_neon)
    KAI_ASM_GLOBAL(kai_convert_f32_f8_neon)
    KAI_ASM_GLOBAL(kai_convert_f8_f32_neon)

// void kai_write_fpmr_raw(uint64_t v);
KAI_ASM_FUNCTION_TYPE(kai_write_fpmr_raw)
KAI_ASM_FUNCTION_LABEL(kai_write_fpmr_raw)
    // x0 = v
    KAI_ASM_INST(0xd51b4440) //	msr	fpmr, x0
    ret
KAI_ASM_FUNCTION_END(kai_write_fpmr_raw)


// uint64_t kai_read_fpmr_raw(void);
KAI_ASM_FUNCTION_TYPE(kai_read_fpmr_raw)
KAI_ASM_FUNCTION_LABEL(kai_read_fpmr_raw)
    KAI_ASM_INST(0xd53b4440) // 	mrs	x0, fpmr
    ret
KAI_ASM_FUNCTION_END(kai_read_fpmr_raw)


// uint8x8_t kai_convert_f8_f32x4x2_neon(float32x4_t a, float32x4_t b);
KAI_ASM_FUNCTION_TYPE(kai_convert_f8_f32x4x2_neon)
KAI_ASM_FUNCTION_LABEL(kai_convert_f8_f32x4x2_neon)
    KAI_ASM_INST(0x0e01f400) // fcvtn v0.8b, v0.4s, v1.4s
    ret
KAI_ASM_FUNCTION_END(kai_convert_f8_f32x4x2_neon)


// void kai_convert_f32_f8_neon(uint8_t* input, float* output, size_t length);
KAI_ASM_FUNCTION_TYPE(kai_convert_f32_f8_neon)
KAI_ASM_FUNCTION_LABEL(kai_convert_f32_f8_neon)
    // if (length == 0) return;
    cbz	x2, fp32_fp8_neon__ret

    // x3 = remaining
    mov	x3, x2

fp32_fp8_neon__loop8:
    // while (remaining >= 8)
    cmp	x3, #8
    blt	fp32_fp8_neon__tail_check

    // y = vld1_u8(&input[i]);
    // input += 8;
    ld1	{v0.8b}, [x0], #8

    // x = f1cvtl(y)
    KAI_ASM_INST(0x2e217801) // f1cvtl	v1.8h, v0.8b

    // out_lo = fcvtl(x.low)
    // out_hi = fcvtl2(x.high)
    fcvtl	v2.4s, v1.4h
    fcvtl2	v3.4s, v1.8h

    // vst1q_f32(&output[i], out_lo); vst1q_f32(&output[i+4], out_hi);
    st1	{v2.4s}, [x1], #16
    st1	{v3.4s}, [x1], #16

    sub	x3, x3, #8
    b	fp32_fp8_neon__loop8

fp32_fp8_neon__tail_check:
    cbz	x3, fp32_fp8_neon__ret

fp32_fp8_neon__tail_loop:
    // y = vdup_n_u8(input[i]);
    // input += 1;
    ldrb	w4, [x0], #1
    dup	v0.8b, w4

    // x = f1cvtl(y)
    KAI_ASM_INST(0x2e217801) // f1cvtl	v1.8h, v0.8b

    // out = fcvtl(x.low); output[i] = lane0(out)
    fcvtl	v2.4s, v1.4h
    str	s2, [x1], #4

    subs	x3, x3, #1
    b.ne	fp32_fp8_neon__tail_loop

fp32_fp8_neon__ret:
    ret
    KAI_ASM_FUNCTION_END(kai_convert_f32_f8_neon)


// void kai_convert_f8_f32_neon(const float* ip, uint8_t* output, uint64_t fpm, size_t length, float scale);
KAI_ASM_FUNCTION_TYPE(kai_convert_f8_f32_neon)
KAI_ASM_FUNCTION_LABEL(kai_convert_f8_f32_neon)
    // use fpmr as provided, no restore in this function
    KAI_ASM_INST(0xd51b4442) // msr	fpmr, x2

    cbz	x3, kai_fp8_fp32_neon__ret

    // x4 = remaining
    mov	x4, x3

    // vscale = vdupq_n_f32(scale)
    dup	v4.4s, v0.s[0]

kai_fp8_fp32_neon__loop8:
    // while (remaining >= 8)
    cmp	x4, #8
    blt	kai_fp8_fp32_neon__tail_check

    // y0 = vmulq_f32(vld1q_f32(&ip[i]), vscale);
    // y1 = vmulq_f32(vld1q_f32(&ip[i+4]), vscale);
    // input += 8;
    ld1	{v1.4s}, [x0], #16
    ld1	{v2.4s}, [x0], #16
    fmul	v1.4s, v1.4s, v4.4s
    fmul	v2.4s, v2.4s, v4.4s

    // packed = fcvtn(y0, y1)
    KAI_ASM_INST(0x0e02f422) // fcvtn	v2.8b, v1.4s, v2.4s

    // vst1_u8(&output[i], packed);
    st1	{v2.8b}, [x1], #8

    sub	x4, x4, #8
    b	kai_fp8_fp32_neon__loop8

kai_fp8_fp32_neon__tail_check:
    cbz	x4, kai_fp8_fp32_neon__ret

kai_fp8_fp32_neon__tail_loop:
    // y = vdupq_n_f32(ip[i] * scale);
    ldr	s1, [x0], #4
    fmul	s1, s1, s0
    dup	v1.4s, v1.s[0]

    // x = fcvtn(y, y); output[i] = byte0(x)
    KAI_ASM_INST(0x0e01f422) // fcvtn	v2.8b, v1.4s, v1.4s
    umov	w5, v2.b[0]
    strb	w5, [x1], #1

    subs	x4, x4, #1
    b.ne	kai_fp8_fp32_neon__tail_loop

kai_fp8_fp32_neon__ret:
    ret
    KAI_ASM_FUNCTION_END(kai_convert_f8_f32_neon)

KAI_ASM_END
