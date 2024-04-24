"""Tests the conversion from ANTLR parse tree to FhY AST and the AST printer"""

import pytest

from fhy.lang.printer import pprint_ast

from ..utils import construct_ast, lexer, parser


def test_matmul_proc(parser):
    source_file_content = """
proc matmul(input int32[m, n] A, input int32[n, p] B, output int32[m, p] C) {
   temp index[1:m] i;
   temp index[1:p] j;
   temp index[1:n] k;
   C[i, j] = sum[k](A[i, k] * B[k, j]);
}
"""
    _ast = construct_ast(parser, source_file_content)
    pprinted_ast = pprint_ast(_ast, indent_char="  ")

    expected_pprinted_ast = """proc matmul(input int32[m, n] A, input int32[n, p] B, output int32[m, p] C) {
  temp index[1:m:1] i;
  temp index[1:p:1] j;
  temp index[1:n:1] k;
  C[i, j] = sum<>[k]((A[i, k] * B[k, j]));
}"""
    assert (
        pprinted_ast == expected_pprinted_ast
    ), f"Expected:\n{expected_pprinted_ast}\nGot:\n{pprinted_ast}"


def test_fully_connected_nn(parser):
    source_file_content = """
op sigmoid(input float32[m] x) -> output float32[m] {
   temp index[1:m] i;
   return 1 / (1 + exp(-x[i]));
}

op forward(input float32[n] x, param float32[m, n] W, param float32[m] b) -> output float32[m] {
   temp index[1:m] i;
   temp index[1:n] j;
   temp float32[m] FC_out;

   FC_out[i] = sum[j](W[i, j] * x[j]) + b[i];
   return sigmoid(FC_out);
}

proc main(input float32[examples, n] X, param float32[m, n] W, param float32[m] b, output float32[examples, m] Y) {
    temp index[1:examples] e;
    temp index[1:n] i;
    temp index[1:m] j;

    temp float32[n] x;
    temp float32[m] y;

    forall (e) {
        x[i] = X[e, i];
        y = forward(x, W, b);
        Y[e, j] = y[j];
    }
}
"""
    _ast = construct_ast(parser, source_file_content)
    pprinted_ast = pprint_ast(_ast, indent_char="   ")

    expected_pprinted_ast = """op sigmoid(input float32[m] x) -> output float32[m] {
   temp index[1:m:1] i;
   return (1 / (1 + exp<>[](-(x[i]))));
}
op forward(input float32[n] x, param float32[m, n] W, param float32[m] b) -> output float32[m] {
   temp index[1:m:1] i;
   temp index[1:n:1] j;
   temp float32[m] FC_out;
   FC_out[i] = (sum<>[j]((W[i, j] * x[j])) + b[i]);
   return sigmoid<>[](FC_out);
}
proc main(input float32[examples, n] X, param float32[m, n] W, param float32[m] b, output float32[examples, m] Y) {
   temp index[1:examples:1] e;
   temp index[1:n:1] i;
   temp index[1:m:1] j;
   temp float32[n] x;
   temp float32[m] y;
   forall (e) {
      x[i] = X[e, i];
      y = forward<>[](x, W, b);
      Y[e, j] = y[j];
   }
}"""
    assert (
        pprinted_ast == expected_pprinted_ast
    ), f"Expected:\n{expected_pprinted_ast}\nGot:\n{pprinted_ast}"
