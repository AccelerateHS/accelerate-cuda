{-# LANGUAGE BangPatterns #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.FullList
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--
-- Non-empty lists of key/value pairs. The lists are strict in the key and lazy
-- in the values. We assume that keys only occur once.
--

module Data.Array.Accelerate.CUDA.FullList (

  FullList(..),
  List(..),

  singleton,
  cons,
  size,
  lookup

) where

import Prelude                  hiding ( lookup )


data FullList k v = FL !k v !(List k v)
data List k v     = Nil | Cons !k v !(List k v)

infixr 5 `Cons`

instance (Eq k, Eq v) => Eq (FullList k v) where
  (FL k1 v1 xs) == (FL k2 v2 ys)      = k1 == k2 && v1 == v2 && xs == ys
  (FL k1 v1 xs) /= (FL k2 v2 ys)      = k1 /= k2 || v1 /= v2 || xs /= ys

instance (Eq k, Eq v) => Eq (List k v) where
  (Cons k1 v1 xs) == (Cons k2 v2 ys) = k1 == k2 && v1 == v2 && xs == ys
  Nil == Nil = True
  _   == _   = False

  (Cons k1 v1 xs) /= (Cons k2 v2 ys) = k1 /= k2 || v1 /= v2 || xs /= ys
  Nil /= Nil = False
  _   /= _   = True


-- List-like operations
--
infixr 5 `cons`
cons :: k -> v -> FullList k v -> FullList k v
cons k v (FL k' v' xs) = FL k v (Cons k' v' xs)

singleton :: k -> v -> FullList k v
singleton k v = FL k v Nil

size :: FullList k v -> Int
size (FL _ _ xs) = 1 + sizeL xs

sizeL :: List k v -> Int
sizeL Nil           = 0
sizeL (Cons _ _ xs) = 1 + sizeL xs

lookup :: Eq k => k -> FullList k v -> Maybe v
lookup key (FL k v xs)
  | key == k    = Just v
  | otherwise   = lookupL key xs
{-# INLINABLE lookup #-}

lookupL :: Eq k => k -> List k v -> Maybe v
lookupL !key = go
  where
    go Nil              = Nothing
    go (Cons k v xs)
      | key == k        = Just v
      | otherwise       = go xs
{-# INLINABLE lookupL #-}

