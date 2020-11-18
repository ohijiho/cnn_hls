<project xmlns="com.autoesl.autopilot.project" name="cnn_hls" top="lenet1">
    <includePaths/>
    <libraryPaths/>
    <Simulation>
        <SimFlow askAgain="false" name="csim" csimMode="0" lastCsimMode="0"/>
    </Simulation>
    <files xmlns="">
        <file name="../test_lenet1.cpp" sc="0" tb="1" cflags=" -I../external/hlslib/include  -fexceptions -Wno-unknown-pragmas" csimflags=" -Wno-unknown-pragmas" blackbox="false"/>
        <file name="../debug.h" sc="0" tb="1" cflags=" -Wno-unknown-pragmas" csimflags=" -Wno-unknown-pragmas" blackbox="false"/>
        <file name="cnn_hls/utils.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="cnn_hls/types.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="cnn_hls/systolic_array.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="cnn_hls/operators.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="cnn_hls/memory.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="cnn_hls/matrix.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="cnn_hls/matmul.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="cnn_hls/lenet.cpp" sc="0" tb="false" cflags="-Icnn_hls/external/hlslib/include -fexceptions" csimflags="" blackbox="false"/>
        <file name="cnn_hls/im2col.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="cnn_hls/config.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="cnn_hls/cnn_functions.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
    </files>
    <solutions xmlns="">
        <solution name="solution_Conv2d" status="inactive"/>
        <solution name="solution_test" status="inactive"/>
        <solution name="solution_lenet" status="active"/>
    </solutions>
</project>

