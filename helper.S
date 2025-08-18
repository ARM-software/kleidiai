
.macro ld1b_4, reg1, reg2, reg3, reg4, pred, addr
ld1b \reg1, \pred/Z, [\addr]
ld1b \reg2, \pred/Z, [\addr, #1, MUL VL]
ld1b \reg3, \pred/Z, [\addr, #2, MUL VL]
ld1b \reg4, \pred/Z, [\addr, #3, MUL VL]
addvl \addr, \addr, #4
.endm

.macro ld1b_2, reg1, reg2, pred, addr
ld1b \reg1, \pred/Z, [\addr]
ld1b \reg2, \pred/Z, [\addr, #1, MUL VL]
addvl \addr, \addr, #2
.endm

.macro ld1b_4p, reg1, reg2, reg3, reg4, pred1, pred2, pred3, pred4,  addr
ld1b \reg1, \pred1/Z, [\addr]
ld1b \reg2, \pred2/Z, [\addr, #1, MUL VL]
ld1b \reg3, \pred3/Z, [\addr, #2, MUL VL]
ld1b \reg4, \pred4/Z, [\addr, #3, MUL VL]
.endm

.macro ld1w_4p, reg1, reg2, reg3, reg4, pred1, pred2, pred3, pred4,  addr
ld1w \reg1, \pred1/Z, [\addr]
ld1w \reg2, \pred2/Z, [\addr, #1, MUL VL]
ld1w \reg3, \pred3/Z, [\addr, #2, MUL VL]
ld1w \reg4, \pred4/Z, [\addr, #3, MUL VL]
.endm


.macro ld1w_4, reg1, reg2, reg3, reg4, pred, addr
ld1w \reg1, \pred/Z, [\addr]
ld1w \reg2, \pred/Z, [\addr, #1, MUL VL]
ld1w \reg3, \pred/Z, [\addr, #2, MUL VL]
ld1w \reg4, \pred/Z, [\addr, #3, MUL VL]
addvl \addr, \addr, #4
.endm

.macro ld1w_2, reg1, reg2, pred, addr
ld1w \reg1, \pred/Z, [\addr]
ld1w \reg2, \pred/Z, [\addr, #1, MUL VL]
addvl \addr, \addr, #2
.endm

.macro ld1w_2d, reg1, reg2, pred, addr
ld1w \reg1, \pred/Z, [\addr]
ld1w \reg2, \pred/Z, [\addr, #1, MUL VL]
.endm

.macro ld1w_2p, reg1, reg2, pred1, pred2, addr
ld1w \reg1, \pred1/Z, [\addr]
ld1w \reg2, \pred2/Z, [\addr, #1, MUL VL]
.endm

.macro ldnt1b_2, reg1, reg2, pred, addr
ldnt1b \reg1, \pred/Z, [\addr]
ldnt1b \reg2, \pred/Z, [\addr, #1, MUL VL]
addvl \addr, \addr, #2
.endm

.macro ldnt1w_4, reg1, reg2, reg3, reg4, pred, addr
ldnt1w \reg1, \pred/Z, [\addr]
ldnt1w \reg2, \pred/Z, [\addr, #1, MUL VL]
ldnt1w \reg3, \pred/Z, [\addr, #2, MUL VL]
ldnt1w \reg4, \pred/Z, [\addr, #3, MUL VL]
addvl \addr, \addr, #4
.endm

.macro ldnt1w_2, reg1, reg2,  pred, addr
ldnt1w \reg1, \pred/Z, [\addr]
ldnt1w \reg2, \pred/Z, [\addr, #1, MUL VL]
addvl \addr, \addr, #2
.endm

.macro ldnt1h_2, reg1, reg2, pred, addr
ldnt1h \reg1, \pred/Z, [\addr]
ldnt1h \reg2, \pred/Z, [\addr, #1, MUL VL]
addvl \addr, \addr, #2
.endm

.macro ld1h_4, reg1, reg2, reg3, reg4, pred, addr
ld1h \reg1, \pred/Z, [\addr]
ld1h \reg2, \pred/Z, [\addr, #1, MUL VL]
ld1h \reg3, \pred/Z, [\addr, #2, MUL VL]
ld1h \reg4, \pred/Z, [\addr, #3, MUL VL]
addvl \addr, \addr, #4
.endm

.macro ld1h_2, reg1, reg2, pred, addr
ld1h \reg1, \pred/Z, [\addr]
ld1h \reg2, \pred/Z, [\addr, #1, MUL VL]
addvl \addr, \addr, #2
.endm

.macro ld1h_2p, reg1, reg2, pred1,pred2, addr
ld1h \reg1, \pred1/Z, [\addr]
ld1h \reg2, \pred2/Z, [\addr, #1, MUL VL]
addvl \addr, \addr, #2
.endm

.macro ld1h_2d, reg1, reg2, pred, addr
ld1h \reg1, \pred/Z, [\addr]
ld1h \reg2, \pred/Z, [\addr, #1, MUL VL]
.endm

.macro st1w_4 reg1, reg2, reg3, reg4, pred, addr
st1w \reg1, \pred, [\addr]
st1w \reg2, \pred, [\addr, #1, MUL VL]
st1w \reg3, \pred, [\addr, #2, MUL VL]
st1w \reg4, \pred, [\addr, #3, MUL VL]
addvl \addr, \addr, #4
.endm

.macro st1w_2 reg1, reg2, pred, addr
st1w \reg1, \pred, [\addr]
st1w \reg2, \pred, [\addr, #1, MUL VL]
.endm

.macro st1w_4p reg1, reg2, reg3, reg4, pred1,pred2,pred3,pred4, addr
st1w \reg1, \pred1, [\addr]
st1w \reg2, \pred2, [\addr, #1, MUL VL]
st1w \reg3, \pred3, [\addr, #2, MUL VL]
st1w \reg4, \pred4, [\addr, #3, MUL VL]
addvl \addr, \addr, #4
.endm


.macro st1w_2p, reg1, reg2,  pred1, pred2, addr
st1w \reg1, \pred1, [\addr]
st1w \reg2, \pred2, [\addr, #1, MUL VL]
.endm

.macro st1w_4pd reg1, reg2, reg3, reg4, pred1,pred2,pred3,pred4, addr
st1w \reg1, \pred1, [\addr]
st1w \reg2, \pred2, [\addr, #1, MUL VL]
st1w \reg3, \pred3, [\addr, #2, MUL VL]
st1w \reg4, \pred4, [\addr, #3, MUL VL]
.endm



.macro st1h_2p, reg1, reg2,  pred1, pred2, addr
st1h \reg1, \pred1, [\addr]
st1h \reg2, \pred2, [\addr, #1, MUL VL]
.endm

.macro st1h_2, reg1, reg2,  pred, addr
st1h \reg1, \pred, [\addr]
st1h \reg2, \pred, [\addr, #1, MUL VL]
addvl \addr, \addr, #2
.endm

.macro move_tile_vector,  reg1, reg2, reg3, reg4,tile, addr, pred
mova \reg1, \pred/M, \tile[\addr, 0] 
mova \reg2, \pred/M, \tile[\addr, 1] 
mova \reg3, \pred/M, \tile[\addr, 2] 
mova \reg4, \pred/M, \tile[\addr, 3] 
.endm

.macro mov_4reg,  regx1, regx2, regx3, regx4,  reg1, reg2, reg3, reg4,
mov  \regx1, \reg1
mov  \regx2, \reg2
mov  \regx3, \reg3
mov  \regx4, \reg4
.endm

.macro fmla_4, regx1, regx2, regx3, regx4,  reg1, reg2, reg3, reg4, reg5
fmla \regx1,  \reg1,  \reg5
fmla \regx2,  \reg2,  \reg5
fmla \regx3,  \reg3,  \reg5
fmla \regx4,  \reg4,  \reg5
.endm

.macro sdot_4, reg1x, reg2x, reg3x, reg4x, reg1, reg2, reg3, reg4, reg5
sdot \reg1x,  \reg1,  \reg5
sdot \reg2x,  \reg2,  \reg5
sdot \reg3x,  \reg3,  \reg5
sdot \reg4x,  \reg4,  \reg5
.endm

.macro fdot_4, reg1x, reg2x, reg3x, reg4x, reg1, reg2, reg3, reg4, reg5, reg6
fmlalb \reg1x,  \reg1,  \reg5
fmlalb \reg2x,  \reg2,  \reg5
fmlalb \reg3x,  \reg3,  \reg5
fmlalb \reg4x,  \reg4,  \reg5
fmlalt \reg1x,  \reg1,  \reg6
fmlalt \reg2x,  \reg2,  \reg6
fmlalt \reg3x,  \reg3,  \reg6
fmlalt \reg4x,  \reg4,  \reg6
.endm

.macro clamp_float_4 reg1, reg2, reg3, reg4,min_val, max_val,  pred
FMIN \reg1, \pred/M, \reg1, \max_val
FMIN \reg2, \pred/M, \reg2, \max_val
FMIN \reg3, \pred/M, \reg3, \max_val
FMIN \reg4, \pred/M, \reg4, \max_val
FMAX \reg1, \pred/M, \reg1, \min_val
FMAX \reg2, \pred/M, \reg2, \min_val
FMAX \reg3, \pred/M, \reg3, \min_val
FMAX \reg4, \pred/M, \reg4, \min_val
.endm

.macro clamp_float_2  , reg1, reg2, min_val ,  max_val , pred
FMIN \reg1, \pred/M, \reg1, \max_val
FMAX \reg1, \pred/M, \reg1, \min_val
FMIN \reg2, \pred/M, \reg2, \max_val
FMAX \reg2, \pred/M, \reg2, \min_val
.endm

.macro clamp_float_1 reg, min_val, max_val, pred
FMAX \reg, \pred/M, \reg, \min_val
FMIN \reg, \pred/M, \reg, \max_val
.endm

.macro sclamp_convert, reg1, reg2, reg3, reg4, min_val, max_val, pred
SCLAMP \reg1,  \min_val, \max_val
SCLAMP \reg2,  \min_val, \max_val
SCLAMP \reg3,  \min_val, \max_val
SCLAMP \reg4,  \min_val, \max_val
.endm

.macro scvtf_convert, reg1, reg2, reg3, reg4, pred
scvtf \reg1, \pred/M, \reg1
scvtf \reg2, \pred/M, \reg2
scvtf \reg3, \pred/M, \reg3
scvtf \reg4, \pred/M, \reg4
.endm

.macro frintn_convert, reg1, reg2, reg3, reg4, pred
frintn \reg1, \pred/M, \reg1
frintn \reg2, \pred/M, \reg2
frintn \reg3, \pred/M, \reg3
frintn \reg4, \pred/M, \reg4
.endm

.macro fcvtzs_convert, reg1, reg2, reg3, reg4, pred
fcvtzs \reg1, \pred/M, \reg1
fcvtzs \reg2, \pred/M, \reg2
fcvtzs \reg3, \pred/M, \reg3
fcvtzs \reg4, \pred/M, \reg4
.endm

.macro add____convert, reg1, reg2, reg3, reg4, reg5
add \reg1, \reg1, \reg5
add \reg2, \reg2, \reg5
add \reg3, \reg3, \reg5
add \reg4, \reg4, \reg5
.endm

.macro fcvt_H2S, reg1, reg2, reg3, reg4, reg1x, reg2x, reg3x, reg4x, pred
fcvt \reg1, \pred/M, \reg1x
fcvt \reg2, \pred/M, \reg2x
fcvt \reg3, \pred/M, \reg3x
fcvt \reg4, \pred/M, \reg4x
.endm

.macro fcvt_2S1H, pred,  regt, regf, reg1, reg2
fcvt \regt, \pred/M, \reg1
fcvt \regf, \pred/M, \reg2
uzp1 \regf, \regt,  \regf
.endm

.macro ld1w_2s, reg1, reg2, pred, addr1, addr2, reg_tmp
mov \reg_tmp, \addr2
ld1w \reg1, \pred/z, [\addr1, \reg_tmp, lsl #2]
incw \reg_tmp, all
ld1w \reg2, \pred/z, [\addr1, \reg_tmp, lsl #2]
.endm

.macro convert_int4_to_int8, reg1, reg2, reg3, reg4_tmp
lsr \reg4_tmp, \reg3, #4
and \reg3, \reg3, #0x0F
zip1 \reg1, \reg3, \reg4_tmp
zip2 \reg2, \reg3, \reg4_tmp
sub \reg1, \reg1, #8
sub \reg2, \reg2, #8
.endm


.macro convert2_int4_to_int8, reg1, reg2, reg3, reg4_tmp, pred
mov \reg4_tmp, \pred/m, \reg3
asr \reg4_tmp, \reg4_tmp, #4
lsl \reg3, \reg3, #4
asr \reg3, \reg3, #4
zip1 \reg1, \reg3, \reg4_tmp
zip2 \reg2, \reg3, \reg4_tmp
.endm


.macro zip1_zip2, reg1, reg2, reg3, reg4, reg_tmp
zip1 \reg_tmp, \reg1, \reg2 
zip2 \reg2, \reg1, \reg2
orr \reg1, \reg_tmp, \reg_tmp
.endm

.macro bfcvt_convert, dest, reg1, reg2, pred, temp
BFCVT \temp, \pred/M, \reg1
BFCVT \dest, \pred/M, \reg2
UZP1 \dest, \temp, \dest
.endm

