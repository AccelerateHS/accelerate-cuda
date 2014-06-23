{-# LANGUAGE GADTs               #-}
-- |
-- Module      : Data.Array.Accelerate.Array.Slice
--
-- Maintainer  : Frederik Meisner Madsen <fmma@diku.dk>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Array.Slice (

  transferDesc, blocksOf,

  TransferDesc(..)

) where

import Control.Arrow                                    ( first )
import Data.List                                        ( groupBy, elemIndex )
import Data.Array.Accelerate.Array.Representation       ( SliceIndex(..) )


-- Slicing algorithm.

-- Figure out how to transfer a multi-dimensional slice from a
-- multi-dimensional array of the same or higher dimensionality to a
-- contiguous memory region. The algorithm uses as few data transfers
-- as possible, where a linear memory copy with a stride consitutes
-- one transfer (cudaMemcpy2d). Each transfer is described by four
-- numbers: A start, a stride, a number of basic blocks and block
-- size.
--

-- Memory transfer(s) description suitable for (multiple calls to)
-- cudaMemcpu2D. The number of calls necessary is given by the length
-- of the starting offsets. The unit is basic elements of the target
-- array.
data TransferDesc = 
  TransferDesc { starts    :: [Int] -- Starting offsets.
               , stride    :: Int   -- Stride.
               , nblocks   :: Int   -- Number of blocks.
               , blocksize :: Int } -- Elements per block.
  deriving Show

-- Get the basic blocks of the given transfer descriptions, described
-- as offset and length. This function can be used to do a memory copy
-- with only contiguous data transfers, if a strided transfer is not
-- available.
blocksOf :: TransferDesc -> [(Int, Int, Int)]
blocksOf tdesc =
  [ ( start + i * stride tdesc
    , i * blocksize tdesc
    , blocksize tdesc) 
  | start <- starts tdesc
  , i <- [0..nblocks tdesc-1]]

-- A slice index in one dimension, annotated with information used to
-- catogorize how the dimension is transferred.
type SliceIR = ( TransferType -- Transfer type annotation.
               , Int )        -- Size of this dimension.

data TransferType =
            Strided -- Transfer the entire dimension. Each element
                    -- will be transfered in a different block, but in
                    -- a single transfer.
            
          | Contiguous -- Transfer the entire dimension in one
                       -- contiguous data transfer. Each element will
                       -- be transfered in the same block in the same
                       -- transfer.

          | Fixed Int -- Transfer a specific element in this
                      -- dimension. The argument is the element index.

          | FixedAll -- Transfer the entire dimension. Each element
                     -- will be transfered in a seperate data
                     -- transfer.

-- Convert a slice index to internal representation used by this
-- algorithm. Initially, all dimensions are represented as strided
-- (transfer entire dimension) or fixed (transfer specific element).
toIR :: SliceIndex slix sl co dim 
     -> slix
     -> dim
     -> [SliceIR]
toIR SliceNil        ()       ()      = []
toIR (SliceAll   si) (sl, ()) (sh, n) = (Strided, n):toIR si sl sh
toIR (SliceFixed si) (sl, i ) (sh, n) = (Fixed i, n):toIR si sl sh
{-
toIR :: SliceIndex slix sl co dim 
     -> slix
     -> dim
     -> [SliceIR]
toIR slix sl dim =
  f slix sl dim []
  where
    f :: SliceIndex slix' sl' co' dim' 
      -> slix'
      -> dim'
      -> [SliceIR]
      -> [SliceIR]
    f SliceNil        ()       ()      res = res
    f (SliceAll   si) (sl, ()) (sh, n) res = f si sl sh ((Strided, n):res)
    f (SliceFixed si) (sl, i ) (sh, n) res = f si sl sh ((Fixed i, n):res)
-}


-- Promote strided slice indices to contiguous slice indices when
-- possible (= all strided innermost dimensions).
strideToContiguous :: [SliceIR] -> [SliceIR]
strideToContiguous slirs =
  let (strided, rest) = span (isStrided . fst) slirs
      conti           = map (first promote) strided
  in conti ++ rest
  where
    isStrided Strided = True
    isStrided _ = False
    
    promote Strided = Contiguous
    promote x = x

-- Find largest group (= next to each other) of strided slice
-- indices. Promote all other strided groups to fixed groups, which
-- entails multiple data transfers in these dimensions. This is
-- necessary step, since it is impossible to describe the memory
-- transfer with the given transfer description type in some
-- situations (namely when there is a gap between strided
-- dimensions). This step chooses the split that results in fewest
-- possible data transfers by selecting the largest stride pivot.
selectStridePivot :: [SliceIR] -> [SliceIR]
selectStridePivot slirs = 
  let -- Group the slice dimensions in groups of same transfer types.
      groups = groupBy sameTransferType slirs

      -- Calculate the sizes of the strided groups, ignore the rest.
      sizes  = map (product . map stridedSize) groups

      -- Find the largest strided group.
      Just imax = elemIndex (maximum sizes) sizes

      -- Promote the remaining strided groups to fixed groups using
      -- multiple data transfers.
      (groups1, pivotGroup:groups2) = splitAt imax groups

      groups' = map (map (first promote)) groups1 ++ pivotGroup:map (map (first promote)) groups2

  in concat groups'
  where
    sameTransferType :: SliceIR -> SliceIR -> Bool
    sameTransferType x y = fst x ~= fst y
      where         
        (~=) :: TransferType -> TransferType -> Bool
        Contiguous ~= Contiguous = True
        Strided    ~= Strided    = True
        Fixed _    ~= Fixed _    = True
        FixedAll   ~= FixedAll   = True
        _          ~= _          = False

    promote :: TransferType -> TransferType
    promote Strided = FixedAll
    promote x = x

    stridedSize :: SliceIR -> Int
    stridedSize (Strided, n) = n
    stridedSize _ = 0

-- Compute a description of the transfers necessary to copy the
-- specified slice.
transferDesc :: SliceIndex slix sl co dim 
             -> slix -- Slice index.
             -> dim  -- Full shape.
             -> TransferDesc
transferDesc slix sl dim = 
  let slirs  = selectStridePivot $ strideToContiguous $ toIR slix sl dim
      tdesc0 = TransferDesc [0] 1 1 1
      size0  = 1
  in f slirs size0 tdesc0
  where
    f :: [SliceIR] -> Int -> TransferDesc -> TransferDesc
    f slirs m tdesc =
      case slirs of
        [] -> tdesc
        
        (ttyp, n):slirs' -> f slirs' (n * m) $
          case ttyp of
            Strided ->
              TransferDesc
                { starts = starts tdesc
                , stride = if stride tdesc == 1 then m else stride tdesc
                , nblocks = n * nblocks tdesc
                , blocksize = blocksize tdesc }

            Contiguous ->
              TransferDesc
                { starts = starts tdesc
                , stride = stride tdesc
                , nblocks = nblocks tdesc
                , blocksize = n * blocksize tdesc }

            Fixed i ->
              TransferDesc
                { starts = [start + i * m | start <- starts tdesc]
                , stride = stride tdesc
                , nblocks = nblocks tdesc
                , blocksize = blocksize tdesc }

            FixedAll ->
              TransferDesc
                { starts = [start + i * m | start <- starts tdesc, i <- [0..n-1]]
                , stride = stride tdesc
                , nblocks = nblocks tdesc
                , blocksize = blocksize tdesc }
