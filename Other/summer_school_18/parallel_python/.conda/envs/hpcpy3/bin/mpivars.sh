#!/bin/sh
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

I_MPI_ROOT=/bb/scinet/course/ss2018/2_ds/4_parallelpython/.conda/envs/hpcpy3; export I_MPI_ROOT

print_help()
{
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
}

if [ -z "${PATH}" ]
then
    PATH="${I_MPI_ROOT}/bin"; export PATH
else
    PATH="${I_MPI_ROOT}/bin:${PATH}"; export PATH
fi

if [ -z "${CLASSPATH}" ]
then
    CLASSPATH="${I_MPI_ROOT}/lib/mpi.jar"; export CLASSPATH
else
    CLASSPATH="${I_MPI_ROOT}/lib/mpi.jar:${CLASSPATH}"; export CLASSPATH
fi

if [ -z "${LD_LIBRARY_PATH}" ]
then
    LD_LIBRARY_PATH="${I_MPI_ROOT}/lib:${I_MPI_ROOT}/mic/lib"; export LD_LIBRARY_PATH
else
    LD_LIBRARY_PATH="${I_MPI_ROOT}/lib:${I_MPI_ROOT}/mic/lib:${LD_LIBRARY_PATH}"; export LD_LIBRARY_PATH
fi

if [ -z "${MANPATH}" ]
then
    if [ `uname -m` = "k1om" ]
    then
        MANPATH="${I_MPI_ROOT}/man"; export MANPATH
    else
        MANPATH="${I_MPI_ROOT}/man":`manpath 2>/dev/null`; export MANPATH
    fi
else
    MANPATH="${I_MPI_ROOT}/man:${MANPATH}"; export MANPATH
fi

library_kind=""
if [ $# -ne 0 ]
then
    library_kind=$1
else
    library_kind=$I_MPI_LIBRARY_KIND
fi
case "$library_kind" in
    debug|debug_mt|release|release_mt)
        LD_LIBRARY_PATH="${I_MPI_ROOT}/lib/${library_kind}:${I_MPI_ROOT}/mic/lib/${library_kind}:${LD_LIBRARY_PATH}"; export LD_LIBRARY_PATH
    ;;
    -h|--help)
        print_help
    ;;
esac
