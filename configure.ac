AC_INIT([accelerate-cuda], [0.15.0.0], [accelerate-haskell@googlegroups.com])
AC_CONFIG_SRCDIR([Data/Array/Accelerate/CUDA.hs])
AC_CONFIG_FILES([accelerate-cuda.buildinfo])

AC_ARG_WITH([compiler], [Haskell compiler], [GHC=$withval],  [AC_PATH_PROG(GHC, ghc)])
AC_ARG_WITH([nvcc],     [CUDA compiler],    [NVCC=$withval], [AC_PROG_CXX(nvcc)])
AC_ARG_WITH([gcc],      [C compiler],       [CC=$withval])

AC_DEFUN([AC_HTYPE], [
    AC_MSG_CHECKING(size of [$2])

    sizeof_hs_[$2]=`$GHC -w -ignore-dot-ghci -e "putStr . show $ Foreign.sizeOf (undefined::$1)"`
    AC_MSG_RESULT($sizeof_hs_[$2])

    case $sizeof_hs_[$2] in
        4) [$5]=[$3]32 ;;
        8) [$5]=[$3]64 ;;
    esac

    # The type name
    def="-D[$4]=$[$5]"
    cpp_flags="$cpp_flags $def"
    ghc_flags="$ghc_flags -optP$def"

    # And size
    def="-DSIZEOF_[$4]=$sizeof_hs_[$2]"
    cpp_flags="$cpp_flags $def"
    ghc_flags="$ghc_flags -optP$def"
])

AC_HTYPE(Data.Int.Int,     Int,  Int,  HTYPE_INT,  type_hs_int)
AC_HTYPE(Data.Word.Word,   Word, Word, HTYPE_WORD, type_hs_word)
AC_HTYPE(Foreign.C.CLong,  Int,  Int,  HTYPE_LONG, type_hs_long)
AC_HTYPE(Foreign.C.CULong, Word, Word, HTYPE_UNSIGNED_LONG, type_hs_unsigned_long)

AC_SUBST([ghc_flags])
AC_SUBST([cpp_flags])
AC_OUTPUT

