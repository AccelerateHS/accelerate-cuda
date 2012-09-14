AC_INIT([accelerate-cuda], [0.12.1.2], [accelerate-haskell@googlegroups.com])
AC_CONFIG_SRCDIR([Data/Array/Accelerate/CUDA.hs])
AC_CONFIG_FILES([accelerate-cuda.buildinfo cubits/accelerate_cuda_shape.h])

# This doesn't pick up a user specified '--with-compiler' flag
#
AC_PROG_CXX(nvcc)
AC_PATH_PROG(GHC, ghc)

AC_DEFUN([AC_SIZEOF_HS], [
    AC_MSG_CHECKING(size of [$1])

    sizeof_hs_[$1]=`$GHC -w -ignore-dot-ghci -e "Foreign.sizeOf (undefined::$1)"`
    AC_MSG_RESULT($sizeof_hs_[$1])

    case $sizeof_hs_[$1] in
        4) [$2]=[$3]32 ;;
        8) [$2]=[$3]64 ;;
    esac

    def="-DSIZEOF_HS$(echo [$1] | tr [[:lower:]] [[:upper:]])=$sizeof_hs_[$1]"
    cpp_flags="$cpp_flags $def"
    ghc_flags="$ghc_flags -optP$def"
])

AC_SIZEOF_HS(Int,  type_hs_int,  Int)
AC_SIZEOF_HS(Char, type_hs_char, Word)

AC_SUBST([type_hs_int])
AC_SUBST([type_hs_char])
AC_SUBST([ghc_flags])
AC_SUBST([cpp_flags])
AC_OUTPUT

