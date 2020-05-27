#
# Copyright 2003-2018 Intel Corporation.
# 
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were
# provided to you (License). Unless the License provides otherwise, you may
# not use, modify, copy, publish, distribute, disclose or transmit this
# software or the related documents without Intel's prior written permission.
# 
# This software and the related documents are provided as is, with no express
# or implied warranties, other than those that are expressly stated in the
# License.
# ?
#
setenv I_MPI_ROOT /bb/scinet/course/ss2018/2_ds/4_parallelpython/.conda/envs/hpcpy3

if !($?PATH) then
    setenv PATH ${I_MPI_ROOT}/bin
else
    setenv PATH ${I_MPI_ROOT}/bin:${PATH}
endif

if !($?CLASSPATH) then
    setenv CLASSPATH ${I_MPI_ROOT}/lib/mpi.jar
else
    set noglob
    setenv CLASSPATH ${I_MPI_ROOT}/lib/mpi.jar:${CLASSPATH}
    unset noglob
endif

if !($?LD_LIBRARY_PATH) then
    setenv LD_LIBRARY_PATH ${I_MPI_ROOT}/lib:${I_MPI_ROOT}/mic/lib
else
    setenv LD_LIBRARY_PATH ${I_MPI_ROOT}/lib:${I_MPI_ROOT}/mic/lib:${LD_LIBRARY_PATH}
endif

if !($?MANPATH) then
    if ( `uname -m` == "k1om" ) then
        setenv MANPATH ${I_MPI_ROOT}/man
    else
        setenv MANPATH ${I_MPI_ROOT}/man:`manpath`
    endif
else
    setenv MANPATH ${I_MPI_ROOT}/man:${MANPATH}
endif

set library_kind=""
if ($#argv > 0) then
    set library_kind=$argv[1]
else if ($?I_MPI_LIBRARY_KIND) then
    set library_kind=$I_MPI_LIBRARY_KIND
endif

switch ($library_kind)
    case debug:
    case debug_mt:
    case release:
    case release_mt:
        setenv LD_LIBRARY_PATH ${I_MPI_ROOT}/lib/${library_kind}:${I_MPI_ROOT}/mic/lib/${library_kind}:${LD_LIBRARY_PATH}
        breaksw
    case --help:
    case -h:
        echo ""
        echo "Usage: mpivars.sh [i_mpi_library_kind]"
        echo ""
        echo "i_mpi_library_kind can be one of the following:"
        echo "      debug           "
        echo "      debug_mt        "
        echo "      release         "
        echo "      release_mt      "
        echo ""
        echo "If the arguments to the sourced script are ignored (consult docs"
        echo "for your shell) the alternative way to specify target is environment"
        echo "variable I_MPI_LIBRARY_KIND to pass"
        echo "i_mpi_library_kind  to the script."
        echo ""
        breaksw
endsw
