<project xmlns="com.autoesl.autopilot.project" name="cnn_hls" top="lenet1">
    <includePaths/>
    <libraryPaths/>
    <Simulation>
        <SimFlow askAgain="false" name="csim" csimMode="0" lastCsimMode="0"/>
    </Simulation>
    <files xmlns="">
        <file name="../test_lenet1.cpp" sc="0" tb="1" cflags=" -I../external/hlslib/include  -fexceptions -Wno-unknown-pragmas" csimflags=" -Wno-unknown-pragmas" blackbox="false"/>
        <file name="cnn_hls/test1.cpp" sc="0" tb="false" cflags="-Icnn_hls/external/hlslib/include -fexceptions" csimflags="" blackbox="false"/>
        <file name="cnn_hls/naive_matmul_top.cpp" sc="0" tb="false" cflags="-Icnn_hls/external/hlslib/include -fexceptions" csimflags="" blackbox="false"/>
        <file name="cnn_hls/lenet.cpp" sc="0" tb="false" cflags="-Icnn_hls/external/hlslib/include -fexceptions" csimflags="" blackbox="false"/>
        <file name="cnn_hls/cnn_functions.cpp" sc="0" tb="false" cflags="-Icnn_hls/external/hlslib/include -fexceptions" csimflags="" blackbox="false"/>
    </files>
    <solutions xmlns="">
        <solution name="solution_Conv2d" status="inactive"/>
        <solution name="solution_test" status="inactive"/>
        <solution name="solution_lenet" status="active"/>
    </solutions>
</project>

