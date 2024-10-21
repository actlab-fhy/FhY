"""Code selector extension for FhY documentation."""

import os
from docutils import nodes
from docutils.parsers.rst import Directive
from pygments import highlight
from pygments.formatters import HtmlFormatter
from fhy_pygments_lexer import FhYLexer


MATMUL_CODE = """proc matmul<T>(input T[M, K] A, input T[K, N] B, output T[M, N] C) {
    temp index[1:M] i;
    temp index[1:N] j;
    temp index[1:K] k;
    C[i, j] = sum[k](A[i, k] * B[k, j]);
}
"""


RESNET18_CODE = """op SGEMM_1<T>(
    input T alpha,
    input T[M, K] A,
    input T[K, N] B,
    input T beta,
    input T[M, N] C
) -> output T[M, N] {
    temp T[M, N] out;

    temp index[1:M] m;
    temp index[1:N] n;
    temp index[1:K] k;

    out[m, n] = alpha * sum[k](A[m, k] * B[k, n])[m, n]
        + beta * C[m, n];

    return out;
}

op _exp(input float32 x) -> output float32 {
    param float32 e = 2.71828;

    temp float32 y = e ** x;

    return y;
}

op _sqrt(input float32 x) -> output float32 {
    param float32 n = 0.5;

    temp float32 y = x ** n;

    return y;
}

op _expectation(input float32[N, C, H, W] X) -> output float32[N] {
    temp float32[N] E;

    temp index[1:N] n;
    temp index[1:C] c;
    temp index[1:H] h;
    temp index[1:W] w;

    E[n] = sum[c, h, w](X[n, c, h, w]) / (C * H * W);

    return E;
}

op _variance(input float32[N, C, H, W] X) -> output float32[N] {
    temp float32[N] V;

    temp index[1:N] n;
    temp index[1:C] c;
    temp index[1:H] h;
    temp index[1:W] w;

    temp float32[N] E = _expectation(X);
    V[n] = sum[c, h, w]((X[n, c, h, w] - E[n]) ** 2)[n]
        / (C * H * W - 1);

    return V;
}

op _pad(
    input float32 [N, C, H, W] X,
    param tuple[int64, int64] PADDING
) -> output float32[N, C, H + 2 * PADDING.0, W + 2 * PADDING.1] {
    temp float32[
        N,
        C,
        H + 2 * PADDING.0,
        W + 2 * PADDING.1
    ] X_padded;

    temp index[1:N] n;
    temp index[1:C] c;
    temp index[1:H] h;
    temp index[1:W] w;
    temp index[1:PADDING.1] padding_x_lower;
    temp index[W + 1:W + PADDING.1] padding_x_upper;
    temp index[1:PADDING.0] padding_y_lower;
    temp index[H + 1:H + PADDING.0] padding_y_upper;

    X_padded[n, c, h + PADDING.0, w + PADDING.1] = X[n, c, h, w];
    X_padded[n, c, padding_y_lower, padding_x_lower] = 0.0;
    X_padded[n, c, padding_y_upper, padding_x_upper] = 0.0;

    return X_padded;
}

op add(
    input float32[N, C, H, W] A,
    input float32[N, C, H, W] B
) -> output float32[N, C, H, W] {
    temp float32[N, C, H, W] Y;

    temp index[1:N] n;
    temp index[1:C] c;
    temp index[1:H] h;
    temp index[1:W] w;

    Y[n, c, h, w] = A[n, c, h, w] + B[n, c, h, w];

    return Y;
}

op avgpool(
    input float32[N, C, H, W] X,
    param tuple[int64, int64] STRIDES,
    param tuple[int64, int64] PADDING,
    param tuple[int64, int64] DILATION,
    param tuple[int64, int64] KERNEL_SIZE
) -> output float32[
    N,
    C,
    (H + 2 * PADDING.0 - DILATION.0 * (KERNEL_SIZE.0 - 1) - 1)
        // STRIDES.0 + 1,
    (W + 2 * PADDING.1 - DILATION.1 * (KERNEL_SIZE.1 - 1) - 1)
        // STRIDES.1 + 1
] {
    temp float32[
        N,
        C,
        (H + 2 * PADDING.0 - DILATION.0 * (KERNEL_SIZE.0 - 1) - 1)
            // STRIDES.0 + 1,
        (W + 2 * PADDING.1 - DILATION.1 * (KERNEL_SIZE.1 - 1) - 1)
            // STRIDES.1 + 1
    ] Y;

    temp index[1:N] n;
    temp index[1:C] c;
    temp index[1:KERNEL_SIZE.0] kh;
    temp index[1:KERNEL_SIZE.1] kw;
    temp index[
        1
        :
        (H + 2 * PADDING.0 - DILATION.0 * (KERNEL_SIZE.0 - 1) - 1)
            // STRIDES.0 + 1:STRIDES.0
    ] oh;
    temp index[
        1
        :
        (W + 2 * PADDING.1 - DILATION.1 * (KERNEL_SIZE.1 - 1) - 1)
            // STRIDES.1 + 1:STRIDES.1
    ] ow;

    temp float32[
        N,
        C,
        H + 2 * PADDING.0,
        W + 2 * PADDING.1
    ] X_padded = _pad(X, PADDING);
    Y[n, c, oh, ow] = sum[kh, kw](X_padded[n, c, oh + kh, ow + kw]);
    Y[n, c, oh, ow] = Y[n, c, oh, ow]
        / (KERNEL_SIZE.0 * KERNEL_SIZE.1);

    return Y;
}

op batchnorm(
    input float32[N, C, H, W] X,
    param float32 eps,
    param float32 gamma,
    param float32 beta
) -> output float32[N, C, H, W] {
    temp float32[N, C, H, W] Y;

    temp index[1:N] n;
    temp index[1:C] c;
    temp index[1:H] h;
    temp index[1:W] w;

    temp float32[N] E = _expectation(X);
    temp float32[N] V = _variance(X);
    Y[n, c, h, w] = gamma * (X[n, c, h, w] - E[n])
        / _sqrt(V[n] + eps) + beta;

    return Y;
}

op conv(
    param tuple[int64, int64] STRIDES,
    param tuple[int64, int64] PADDING,
    param tuple[int64, int64] DILATION,
    param int64 groups,
    input float32[N, IC, H, W] X,
    param float32[OC, IC // groups, KH, KW] W
) -> output float32[
    N,
    OC,
    (H + 2 * PADDING.0 - DILATION.0 * (KH - 1) - 1)
        // STRIDES.0 + 1,
    (W + 2 * PADDING.1 - DILATION.1 * (KW - 1) - 1)
        // STRIDES.1 + 1
] {
    temp float32[
        N,
        OC,
        (H + 2 * PADDING.0 - DILATION.0 * (KH - 1) - 1)
            // STRIDES.0 + 1,
        (W + 2 * PADDING.1 - DILATION.1 * (KW - 1) - 1)
            // STRIDES.1 + 1
    ] Y;

    temp index[1:N] n;
    temp index[1:IC] ic;
    temp index[1:OC] oc;
    temp index[1:KH] kh;
    temp index[1:KW] kw;
    temp index[
        1
        :
        (H + 2 * PADDING.0 - DILATION.0 * (KH - 1) - 1)
            // STRIDES.0 + 1
        :
        STRIDES.0
    ] oh;
    temp index[
        1
        :
        (W + 2 * PADDING.1 - DILATION.1 * (KW - 1) - 1)
            // STRIDES.1 + 1
        :
        STRIDES.1
    ] ow;

    temp float32[
        N,
        IC,
        H + 2 * PADDING.0, W + 2 * PADDING.1
    ] X_padded = _pad(PADDING, X);
    Y[n, oc, oh, ow] = sum[ic, kh, kw](
        X_padded[n, ic, oh + kh, ow + kw] * W[oc, ic, kh, kw]
    );

    return Y;
}

op fc(
    input float32[N, I] X,
    param float32[I, O] W,
    param float32[O] B
) -> output float32[N, O] {
    temp float32[N, O] Y;

    temp index[1:N] n;
    temp index[1:O] o;

    temp float32[N, O] _B;
    _B[n, o] = B[o];
    Y = SGEMM_1<float32>(1.0, X, W, 1.0, _B);

    return Y;
}

op flatten(
    input float32[N, C, H, W] I
) -> output float32[N, C * H * W] {
    temp float32[N, C * H * W] O;

    temp index[1:N] n;
    temp index[1:C] c;
    temp index[1:H] h;
    temp index[1:W] w;

    O[N, (H * W) * (c - 1) + (W) * (h - 1) + w] = I[n, c, h, w];

    return O;
}

op maxpool(
    param tuple[int64, int64] STRIDES,
    param tuple[int64, int64] PADDING,
    param tuple[int64, int64] DILATION,
    param tuple[int64, int64] KERNEL_SIZE,
    input float32[N, C, H, W] X
) -> output float32[
    N,
    C,
    (H + 2 * PADDING.0 - DILATION.0 * (KERNEL_SIZE.0 - 1) - 1)
        // STRIDES.0 + 1,
    (W + 2 * PADDING.1 - DILATION.1 * (KERNEL_SIZE.1 - 1) - 1)
        // STRIDES.1 + 1
] {
    temp float32[
        N,
        C,
        (H + 2 * PADDING.0 - DILATION.0 * (KERNEL_SIZE.0 - 1) - 1)
            // STRIDES.0 + 1,
        (W + 2 * PADDING.1 - DILATION.1 * (KERNEL_SIZE.1 - 1) - 1)
            // STRIDES.1 + 1
    ] Y;

    temp index[1:N] n;
    temp index[1:C] c;
    temp index[1:KERNEL_SIZE.0] kh;
    temp index[1:KERNEL_SIZE.1] kw;
    temp index[
        1
        :
        (H + 2 * PADDING.0 - DILATION.0 * (KERNEL_SIZE.0 - 1) - 1)
            // STRIDES.0 + 1
        :
        STRIDES.0
    ] oh;
    temp index[
        1
        :
        (W + 2 * PADDING.1 - DILATION.1 * (KERNEL_SIZE.1 - 1) - 1)
        // STRIDES.1 + 1
        :
        STRIDES.1
    ] ow;

    temp float32[
        N,
        C,
        H + 2 * PADDING.0,
        W + 2 * PADDING.1
    ] X_padded = _pad(PADDING, X);
    Y[n, c, oh, ow] = max[kh, kw](X_padded[n, c, oh + kh, ow + kw]);

    return Y;
}

op relu(
    input float32[N, C, H, W] X
) -> output float32[N, C, H, W] {
    temp float32[N, C, H, W] Y;

    temp index[1:N] n;
    temp index[1:C] c;
    temp index[1:H] h;
    temp index[1:W] w;

    # Y[n, c, h, w] = X[n, c, h, w] > 0.0 ? X[n, c, h, w] : 0.0;
    Y[n, c, h, w] = X[n, c, h, w];

    return Y;
}

op softmax(
    input float32[N, M] X
) -> output float32[N, M] {
    temp float32[N, M] Y;

    temp index[1:N] n;
    temp index[1:M] m;

    Y[n, m] = (_exp(X[n, m])) / sum[m](_exp(X[n, m]));

    return Y;
}

proc _conv3x3(
    param tuple[int64, int64] STRIDES,
    param tuple[int64, int64] DILATION,
    param int64 GROUPS,
    input float32[N, IC, H, W] X,
    param float32[OC, IC // GROUPS, 3, 3] W,
    output float32[
        N,
        OC,
        (H + 2 * DILATION.0 - 3) // STRIDES.0 + 1,
        (W + 2 * DILATION.1 - 3) // STRIDES.1 + 1
    ] Y
) {
    Y = conv(STRIDES, DILATION, DILATION, GROUPS, X, W);
}

proc _conv1x1(
    param tuple[int64, int64] STRIDES,
    param int64 GROUPS,
    input float32[N, IC, H, W] X,
    param float32[OC, IC // GROUPS, 1, 1] W,
    output float32[N, OC, (H - 1) // STRIDES.0 + 1, (W - 1)
        // STRIDES.1 + 1] Y
) {
    Y = conv(STRIDES, (0, 0), (1, 1), GROUPS, X, W);
}

proc basic_block_stride_eq_1(
    input float32[N, C, H, W] X,
    param float32[C, C, 3, 3] conv1_W,
    param float32 eps1,
    param float32 gamma1,
    param float32 beta1,
    param float32[C, C, 3, 3] conv2_W,
    param float32 eps2,
    param float32 gamma2,
    param float32 beta2,
    output float32[N, C, H, W] Y
) {
    temp float32[N, C, H, W] conv1_Y;
    temp float32[N, C, H, W] bn1_Y;
    temp float32[N, C, H, W] relu1_Y;
    temp float32[N, C, H, W] conv2_Y;
    temp float32[N, C, H, W] bn2_Y;
    temp float32[N, C, H, W] add_Y;

    _conv3x3((1, 1), (1, 1), 1, X, conv1_W, conv1_Y);
    bn1_Y = batchnorm(conv1_Y, eps1, gamma1, beta1);
    relu1_Y = relu(bn1_Y);
    _conv3x3((1, 1), (1, 1), 1, relu1_Y, conv2_W, conv2_Y);
    bn2_Y = batchnorm(conv2_Y, eps2, gamma2, beta2);
    add_Y = add(bn2_Y, X);
    Y = relu(bn2_Y);
}

proc basic_block_stride_neq_1(
    input float32[N, IC, H, W] X,
    param float32[C, IC, 3, 3] conv1_W,
    param float32 eps1,
    param float32 gamma1,
    param float32 beta1,
    param float32[C, C, 3, 3] conv2_W,
    param float32 eps2,
    param float32 gamma2,
    param float32 beta2,
    param float32[C, IC, 1, 1] conv3_W,
    param float32 eps3,
    param float32 gamma3,
    param float32 beta3,
    param int64 STRIDE,
    output float32[
        N,
        C,
        (H - 1) // STRIDE + 1,
        (W - 1) // STRIDE + 1
    ] Y
) {
    param int64 CONV1_OH = (H - 1) // STRIDE + 1;
    param int64 CONV1_OW = (W - 1) // STRIDE + 1;
    param int64 CONV2_OH = CONV1_OH;
    param int64 CONV2_OW = CONV1_OW;

    temp float32[N, C, CONV1_OH, CONV1_OW] conv1_Y;
    temp float32[N, C, CONV1_OH, CONV1_OW] bn1_Y;
    temp float32[N, C, CONV1_OH, CONV1_OW] relu1_Y;
    temp float32[N, C, CONV2_OH, CONV2_OW] conv2_Y;
    temp float32[N, C, CONV2_OH, CONV2_OW] bn2_Y;
    temp float32[N, C, CONV2_OH, CONV2_OW] conv3_Y;
    temp float32[N, C, CONV2_OH, CONV2_OW] bn3_Y;
    temp float32[N, C, CONV2_OH, CONV2_OW] add_Y;

    _conv3x3((STRIDE, STRIDE), (1, 1), 1, X, conv1_W, conv1_Y);
    bn1_Y = batchnorm(eps1, gamma1, beta1, conv1_Y);
    relu1_Y = relu(bn1_Y);

    _conv3x3((1, 1), (1, 1), 1, relu1_Y, conv2_W, conv2_Y);
    bn2_Y = batchnorm(eps2, gamma2, beta2, conv2_Y);

    _conv1x1((STRIDE, STRIDE), 1, X, conv3_W, conv3_Y);
    bn3_Y = batchnorm(eps3, gamma3, beta3, conv3_Y);

    add_Y = add(bn2_Y, bn3_Y);
    Y = relu(add_Y);
}

proc main(
    input float32[N, 3, 224, 224] X,
    param float32[64, 3, 7, 7] conv1_W,
    param tuple[float32, float32, float32] bn1_params,

    param float32[64, 64, 3, 3] bb1_conv1_W,
    param tuple[float32, float32, float32] bb1_bn1_params,
    param float32[64, 64, 3, 3] bb1_conv2_W,
    param tuple[float32, float32, float32] bb1_bn2_params,

    param float32[64, 64, 3, 3] bb2_conv1_W,
    param tuple[float32, float32, float32] bb2_bn1_params,
    param float32[64, 64, 3, 3] bb2_conv2_W,
    param tuple[float32, float32, float32] bb2_bn2_params,

    param float32[128, 64, 3, 3] bb3_conv1_W,
    param tuple[float32, float32, float32] bb3_bn1_params,
    param float32[128, 128, 3, 3] bb3_conv2_W,
    param tuple[float32, float32, float32] bb3_bn2_params,
    param float32[128, 64, 1, 1] bb3_conv3_W,
    param tuple[float32, float32, float32] bb3_bn3_params,

    param float32[128, 128, 3, 3] bb4_conv1_W,
    param tuple[float32, float32, float32] bb4_bn1_params,
    param float32[128, 128, 3, 3] bb4_conv2_W,
    param tuple[float32, float32, float32] bb4_bn2_params,

    param float32[256, 128, 3, 3] bb5_conv1_W,
    param tuple[float32, float32, float32] bb5_bn1_params,
    param float32[256, 256, 3, 3] bb5_conv2_W,
    param tuple[float32, float32, float32] bb5_bn2_params,
    param float32[256, 128, 1, 1] bb5_conv3_W,
    param tuple[float32, float32, float32] bb5_bn3_params,

    param float32[256, 256, 3, 3] bb6_conv1_W,
    param tuple[float32, float32, float32] bb6_bn1_params,
    param float32[256, 256, 3, 3] bb6_conv2_W,
    param tuple[float32, float32, float32] bb6_bn2_params,

    param float32[512, 256, 3, 3] bb7_conv1_W,
    param tuple[float32, float32, float32] bb7_bn1_params,
    param float32[512, 512, 3, 3] bb7_conv2_W,
    param tuple[float32, float32, float32] bb7_bn2_params,
    param float32[512, 256, 1, 1] bb7_conv3_W,
    param tuple[float32, float32, float32] bb7_bn3_params,

    param float32[512, 512, 3, 3] bb8_conv1_W,
    param tuple[float32, float32, float32] bb8_bn1_params,
    param float32[512, 512, 3, 3] bb8_conv2_W,
    param tuple[float32, float32, float32] bb8_bn2_params,

    param float32[512, 1000] fc_W,
    param float32[1000] fc_B,

    output float32[N, 1000] Y
) {
    temp float32[N, 64, 112, 112] conv1_Y;
    temp float32[N, 64, 112, 112] bn1_Y;
    temp float32[N, 64, 112, 112] relu1_Y;
    temp float32[N, 64, 56, 56] maxpool1_Y;
    temp float32[N, 64, 56, 56] bb1_Y;
    temp float32[N, 64, 56, 56] bb2_Y;
    temp float32[N, 128, 28, 28] bb3_Y;
    temp float32[N, 128, 28, 28] bb4_Y;
    temp float32[N, 256, 14, 14] bb5_Y;
    temp float32[N, 256, 14, 14] bb6_Y;
    temp float32[N, 512, 7, 7] bb7_Y;
    temp float32[N, 512, 7, 7] bb8_Y;
    temp float32[N, 512, 1, 1] global_avg_pool_Y;
    temp float32[N, 512] flatten_Y;

    # Initial operation sequence
    conv1_Y = conv((2, 2), (3, 3), (1, 1), 1, X, conv1_W);
    bn1_Y = batchnorm(conv1_Y, bn1_params.0,
                      bn1_params.1, bn1_params.2);
    relu1_Y = relu(bn1_Y);
    maxpool1_Y = maxpool((2, 2), (1, 1), (1, 1), (3, 3), relu1_Y);

    # First set of basic blocks
    basic_block_stride_eq_1(
        maxpool1_Y,
        bb1_conv1_W,
        bb1_bn1_params.0,
        bb1_bn1_params.1,
        bb1_bn1_params.2,
        bb1_conv2_W,
        bb1_bn2_params.0,
        bb1_bn2_params.1,
        bb1_bn2_params.2,
        bb1_Y
    );
    basic_block_stride_eq_1(
        bb1_Y,
        bb2_conv1_W,
        bb2_bn1_params.0,
        bb2_bn1_params.1,
        bb2_bn1_params.2,
        bb2_conv2_W,
        bb2_bn2_params.0,
        bb2_bn2_params.1,
        bb2_bn2_params.2,
        bb2_Y
    );

    # Second set of basic blocks
    basic_block_stride_neq_1(
        bb2_Y,
        bb3_conv1_W,
        bb3_bn1_params.0,
        bb3_bn1_params.1,
        bb3_bn1_params.2,
        bb3_conv2_W,
        bb3_bn2_params.0,
        bb3_bn2_params.1,
        bb3_bn2_params.2,
        bb3_conv3_W,
        bb3_bn3_params.0,
        bb3_bn3_params.1,
        bb3_bn3_params.2,
        2,
        bb3_Y
    );
    basic_block_stride_eq_1(
        bb3_Y,
        bb4_conv1_W,
        bb4_bn1_params.0,
        bb4_bn1_params.1,
        bb4_bn1_params.2,
        bb4_conv2_W,
        bb4_bn2_params.0,
        bb4_bn2_params.1,
        bb4_bn2_params.2,
        bb4_Y
    );

    # Third set of basic blocks
    basic_block_stride_neq_1(
        bb4_Y,
        bb5_conv1_W,
        bb5_bn1_params.0,
        bb5_bn1_params.1,
        bb5_bn1_params.2,
        bb5_conv2_W,
        bb5_bn2_params.0,
        bb5_bn2_params.1,
        bb5_bn2_params.2,
        bb5_conv3_W,
        bb5_bn3_params.0,
        bb5_bn3_params.1,
        bb5_bn3_params.2,
        2,
        bb5_Y
    );
        basic_block_stride_eq_1(
        bb5_Y,
        bb6_conv1_W,
        bb6_bn1_params.0,
        bb6_bn1_params.1,
        bb6_bn1_params.2,
        bb6_conv2_W,
        bb6_bn2_params.0,
        bb6_bn2_params.1,
        bb6_bn2_params.2,
        bb6_Y
    );

    # Fourth set of basic blocks
    basic_block_stride_neq_1(
        bb6_Y,
        bb7_conv1_W,
        bb7_bn1_params.0,
        bb7_bn1_params.1,
        bb7_bn1_params.2,
        bb7_conv2_W,
        bb7_bn2_params.0,
        bb7_bn2_params.1,
        bb7_bn2_params.2,
        bb7_conv3_W,
        bb7_bn3_params.0,
        bb7_bn3_params.1,
        bb7_bn3_params.2,
        2,
        bb7_Y
    );
        basic_block_stride_eq_1(
        bb7_Y,
        bb8_conv1_W,
        bb8_bn1_params.0,
        bb8_bn1_params.1,
        bb8_bn1_params.2,
        bb8_conv2_W,
        bb8_bn2_params.0,
        bb8_bn2_params.1,
        bb8_bn2_params.2,
        bb8_Y
    );

    # Final operation sequence
    global_avg_pool_Y = avgpool(bb8_Y, (1, 1),
                                (0, 0), (1, 1), (7, 7));
    flatten_Y = flatten(global_avg_pool_Y);
    Y = fc(flatten_Y, fc_W, fc_B);
}
"""


MPC_CODE = """proc mvmul(
    input float32[M, N] A,
    input float32[N] B,
    output float32[M] C
) {
    temp index[1:M] i;
    temp index[1:N] j;
    C[i] = sum[j](A[i, j] * B[j]);
}

proc process_pos(
    input float32[3] pos,
    output float32[3] processed_pos
) {
    temp index[1:3] i;
    processed_pos[i] = pos[i] * 2.0;
}

proc predict_trajectory(
    input float32[A] pos,
    input float32[B] ctrl_mdl,
    param float32[C, A] P,
    param float32[C, B] H,
    output float32[C] pred
) {
    temp index[1:C] i;
    temp float32[C] pred_comp_1;
    temp float32[C] pred_comp_2;
    mvmul(P, pos, pred_comp_1);
    mvmul(H, ctrl_mdl, pred_comp_2);
    pred[i] = pred_comp_1[i] + pred_comp_2[i];
}

proc update_ctrl_model(
    input float32[B] ctrl_prev,
    input float32[B] g,
    output float32[B] ctrl_mdl,
    output float32[S] ctrl_sgnl,
    param int32 h
) {
    temp index[1:B - 1] i;
    temp index[1:S] j;
    ctrl_sgnl[j] = ctrl_prev[h * j];
    ctrl_mdl[(h - 1) * j] = 0.0;
    ctrl_mdl[i] = ctrl_prev[(i + 1) * h] - g[(i + 1) * h];
}

proc compute_ctrl_grad(
    input float32[C] pos_pred,
    input float32[B] ctrl_mdl,
    input float32[C] pos_ref,
    param float32[B, C] HQ_g,
    param float32[B, B] R_g,
    output float32[B] g
) {
    temp index[1:B] i;
    temp index[1:C] j;
    temp float32[B] P_g;
    temp float32[B] H_g;
    temp float32[C] err;
    err[j] = pos_ref[j] - pos_pred[j];
    mvmul(HQ_g, err, P_g);
    mvmul(R_g, ctrl_mdl, H_g);
    g[i] = P_g[i] + H_g[i];
}

proc main(
    input float32[3] pos,
    state float32[20] ctrl_mdl,
    param float32[30] pos_ref,
    param float32[30, 3] P,
    param float32[20, 30] HQ_g,
    param float32[30, 20] H,
    param float32[20, 20] R_g,
    output float32[2] ctrl_sgnl
) {
    temp float32[30] pos_pred;
    temp float32[20] g;
    temp float32[3] processed_pos;
    process_pos(pos, processed_pos);
    predict_trajectory(processed_pos, ctrl_mdl, P, H, pos_pred);
    compute_ctrl_grad(pos_pred, ctrl_mdl, pos_ref, HQ_g, R_g, g);
    update_ctrl_model(ctrl_mdl, g, ctrl_mdl, ctrl_sgnl, 10);
}
"""


def id_to_title(example_id: str) -> str:
    """Convert example ID to example title."""
    return example_id.replace("_", " ").title()


class FhYExample:
    """Dataclass for FhY example."""
    _example_id_registry = set()

    name: str
    _example_id: str
    fhy_code: str
    html_code: str

    def __init__(self, name: str, example_id: str, code: str) -> None:
        self.name = name
        self._example_id = example_id
        self.fhy_code = code
        self.html_code = self._convert_fhy_code_to_html()

        if example_id in FhYExample._example_id_registry:
            raise ValueError(f"Example ID '{example_id}' is not unique.")
        FhYExample._example_id_registry.add(example_id)

    @property
    def example_id(self) -> str:
        return self._example_id

    def _convert_fhy_code_to_html(self) -> str:
        """Convert FhY code to HTML."""
        lexer = FhYLexer()
        formatter = HtmlFormatter()
        return highlight(self.fhy_code, lexer, formatter)


EXAMPLES = {
    "matrix_multiplication": FhYExample("Matrix Multiplication", "matrix_multiplication", MATMUL_CODE),
    "mpc": FhYExample("Model Predictive Control", "mpc", MPC_CODE),
    "resnet18": FhYExample("Resnet18", "resnet18", RESNET18_CODE),
}


class CodeSelectorDirective(Directive):
    """Custom directive for code selector."""
    has_content = False

    def run(self):
        fhy_examples = EXAMPLES

        example_buttons = []
        example_html_code = []
        for i, example in enumerate(fhy_examples.values()):
            if i == 0:
                button_display_style = "active"
                button_selected = "true"
                tab_display_style = "show active"
            else:
                button_display_style = ""
                button_selected = "false"
                tab_display_style = ""
            example_buttons.append(
                f'<button class="nav-link {button_display_style}" id="v-pills-{example.example_id}-tab" data-bs-toggle="pill" data-bs-target="#v-pills-{example.example_id}" type="button" role="tab" aria-controls="v-pills-{example.example_id}" aria-selected="{button_selected}">{example.name}</button>'
            )
            example_html_code.append(
                f'<div class="tab-pane fade {tab_display_style}" id="v-pills-{example.example_id}" role="tabpanel" aria-labelledby="v-pills-{example.example_id}-tab">{example.html_code}</div>'
            )

        custom_html = f"""
<link rel="stylesheet" type="text/css" href="_static/css/code_selector.css">
<div class="card">
    <div class="card-body">
        <div class="d-flex align-items-start">
            <div class="nav flex-column nav-pills me-3" id="v-pills-tab" role="tablist" aria-orientation="vertical">
                {"".join(example_buttons)}
            </div>
            <div class="tab-content" id="v-pills-tabContent">
                {"".join(example_html_code)}
            </div>
        </div>
    </div>
</div>
"""
        raw_node = nodes.raw("", custom_html, format='html')
        return [raw_node]



def setup(app):
    """Setup function for the FhY code selector extension."""
    app.add_directive('code_selector', CodeSelectorDirective)
