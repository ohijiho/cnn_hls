############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2020 Xilinx, Inc. All Rights Reserved.
############################################################
open_project cnn_hls
add_files cnn_hls/cnn_functions.h
add_files cnn_hls/config.h
add_files cnn_hls/matmul.h
add_files cnn_hls/memory.h
add_files cnn_hls/operators.h
add_files cnn_hls/systolic_array.h
add_files cnn_hls/types.h
add_files cnn_hls/utils.h
open_solution "solution1" -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
#source "./cnn_hls/solution1/directives.tcl"
#csim_design
csynth_design
#cosim_design
export_design -format ip_catalog
