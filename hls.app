<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="com.autoesl.autopilot.project" name="cnn_hls" top="lenet1">
  <files>
    <file name="cnn_hls/cnn_functions.cpp" sc="0" tb="false" cflags="-Icnn_hls/external/hlslib/include -fexceptions" blackbox="false"/>
    <file name="cnn_hls/lenet.cpp" sc="0" tb="false" cflags="-Icnn_hls/external/hlslib/include -fexceptions" blackbox="false"/>
    <file name="cnn_hls/naive_matmul_top.cpp" sc="0" tb="false" cflags="-Icnn_hls/external/hlslib/include -fexceptions" blackbox="false"/>
    <file name="cnn_hls/test1.cpp" sc="0" tb="false" cflags="-Icnn_hls/external/hlslib/include -fexceptions" blackbox="false"/>
  </files>
  <solutions>
    <solution name="solution_Conv2d" status="inactive"/>
    <solution name="solution_test" status="inactive"/>
    <solution name="solution_lenet" status="active"/>
  </solutions>
  <includePaths/>
  <libraryPaths/>
  <Simulation>
    <SimFlow name="csim" csimMode="0" lastCsimMode="0"/>
  </Simulation>
</project>
