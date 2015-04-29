{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Persistent
-- Copyright   : [2008..2014] Manuel M T Chakravarty, Gabriele Keller
--               [2009..2014] Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Persistent (

  KernelTable, KernelKey, KernelEntry(..),
  new, lookup, insert, persist,

  module_finalizer,

) where

-- friends
import Data.Array.Accelerate.Error
import Data.Array.Accelerate.FullList                   ( FullList )
import Data.Array.Accelerate.Lifetime                   ( Lifetime, withLifetime, newLifetime )
import Data.Array.Accelerate.CUDA.Context
import qualified Data.Array.Accelerate.CUDA.Debug       as D
import qualified Data.Array.Accelerate.FullList         as FL

-- libraries
import Numeric
import Control.Applicative
import Control.Concurrent
import Control.Exception
import Control.Monad                                    ( when )
import Data.Binary
import Data.Binary.Get
import Data.ByteString                                  ( ByteString )
import Data.ByteString.Internal                         ( w2c )
import Data.Char
import Data.Hashable
import Data.Maybe                                       ( fromMaybe )
import Data.Version
import System.Directory
import System.FilePath
import System.IO
import System.IO.Error
import System.Mem.Weak
import qualified Data.ByteString                        as BS
import qualified Data.ByteString.Lazy                   as BL
import qualified Data.ByteString.Lazy.Internal          as BL
import qualified Data.HashTable.IO                      as HT
import Prelude                                          hiding ( lookup )

import qualified Foreign.CUDA.Driver                    as CUDA

import Paths_accelerate_cuda


instance Hashable CUDA.Compute where
  hashWithSalt salt (CUDA.Compute major minor)
    = salt `hashWithSalt` major `hashWithSalt` minor

instance Binary CUDA.Compute where
  put (CUDA.Compute major minor) = put major >> put minor
  get                            = CUDA.Compute <$> get <*> get


-- Interface -------------------------------------------------------------------
-- ---------                                                                  --

type HashTable key val = HT.BasicHashTable key val

data KernelTable = KT {-# UNPACK #-} !ProgramCache      -- first level, in-memory cache
                      {-# UNPACK #-} !PersistentCache   -- second level, on-disk cache

new :: IO KernelTable
new = do
  message "initialise kernel table"
  cacheDir <- cacheDirectory
  createDirectoryIfMissing True cacheDir
  --
  local         <- newMVar =<< HT.new
  persistent    <- restore (cacheDir </> "persistent.db")
  --
  return        $! KT local persistent


-- Lookup a kernel through the two-level cache system. If the kernel is found in
-- the persistent cache, it is loaded and linked into the current context.
--
lookup :: Context -> KernelTable -> KernelKey -> IO (Maybe KernelEntry)
lookup context (KT !kt_ref !pt_ref) !key = withMVar kt_ref $ \kt -> do
  -- First check the local cache. If we get a hit, this could be:
  --   a) currently compiling
  --   b) compiled, but not linked into the current context
  --   c) compiled & linked
  --
  v1    <- HT.lookup kt key
  case v1 of
    Just _      -> return v1
    Nothing     -> withMVar pt_ref $ \pt -> do

    -- Check the persistent cache. If found, read in the associated object file
    -- and link it into the current context. Also add to the first-level cache.
    --
    -- TLM: maybe we should change KernelObject to hold a possibly empty list,
    --      so we don't have to mess with the CUDA context here.
    --
    v2  <- HT.lookup pt key
    case v2 of
      Nothing   -> return Nothing
      Just ()   -> do
        message "found/persistent"
        cubin   <- (</>) <$> cacheDirectory <*> pure (cacheFilePath key)
        bin     <- BS.readFile cubin
        !mdl    <- CUDA.loadData bin
        lmdl    <- newLifetime mdl
        let obj  = KernelObject bin (FL.singleton (deviceContext context) lmdl)
        addFinalizer lmdl (module_finalizer (weakContext context) key lmdl)
        HT.insert kt key obj
        return  $! Just obj


-- Insert a key/value pair into the first-level cache. This does not add the
-- entry to the persistent database.
--
-- TLM: Also add to the persistent cache, or return a boolean as to whether it
--      exists there already? Would require updating that hash table as new
--      entries are added, which the functions currently do not do.
--
{-# INLINE insert #-}
insert :: KernelTable -> KernelKey -> KernelEntry -> IO ()
insert (KT !kt_ref !_) !key !val = withMVar kt_ref $ \kt -> HT.insert kt key val


-- Unload a kernel module from the specified context
--
module_finalizer :: Weak (Lifetime CUDA.Context) -> KernelKey -> Lifetime CUDA.Module -> IO ()
module_finalizer weak_ctx key lmdl = do
  mc <- deRefWeak weak_ctx
  case mc of
    Nothing     -> D.traceIO D.dump_gc ("gc: finalise module/dead context: " ++ cacheFilePath key)
    Just fctx   -> D.traceIO D.dump_gc ("gc: finalise module: "              ++ cacheFilePath key)
                >> withLifetime fctx (\ctx -> withLifetime lmdl (\mdl ->
                     bracket_ (CUDA.push ctx) CUDA.pop (CUDA.unload mdl)))


-- Local cache -----------------------------------------------------------------
-- -----------                                                                --
--
-- Kernel code that has been generated and linked into the currently running
-- program.

-- An exact association between an accelerate computation and its
-- implementation, which is either a reference to the external compiler (nvcc)
-- or the resulting binary module.
--
-- Note that since we now support running in multiple contexts, we also need to
-- keep track of
--   a) the compute architecture the code was compiled for
--   b) which contexts have linked the code
--
-- We aren't concerned with true (typed) equality of an OpenAcc expression,
-- since we largely want to disregard the array environment; we really only want
-- to assert the type and index of those variables that are accessed by the
-- computation and no more, but we can not do that. Instead, this is keyed to
-- the generated kernel code.
--
type ProgramCache = MVar ( HashTable KernelKey KernelEntry )

type KernelKey    = (CUDA.Compute, ByteString)
data KernelEntry
  -- A currently compiling external process. We record the path of the .cu file
  -- being compiled, and an MVar that will be filled upon completion.
  --
  = CompileProcess !FilePath
                   {-# UNPACK #-} !(MVar ())

  -- The raw compiled data, and the list of contexts that the object has already
  -- been linked into. If we locate this entry in the ProgramCache, it may have
  -- been inserted by an alternate but compatible device context, so just
  -- re-link into the current context.
  --
  | KernelObject {-# UNPACK #-} !ByteString
                 {-# UNPACK #-} !(FullList (Lifetime CUDA.Context)
                                           (Lifetime CUDA.Module))


-- Persistent cache ------------------------------------------------------------
-- ----------------                                                           --
--
-- Stash compiled code into the user's home directory so that they are available
-- across separate runs of the program.
--
-- TLM: We don't have any migration or versioning policy here, so cache files
--      will be kept around indefinitely. This can easily clutter the cache by
--      generating many similar kernels that differ only by, for example, an
--      embedded constant value.
--
--      One way to handle this is to put a maximum size on the cache (either as
--      disk space consumed or number of kernels) and once the maximum size is
--      reached keep only the most recently used files.

type PersistentCache = MVar ( HashTable KernelKey () )


-- The root directory of where the various persistent cache files live; the
-- database and each individual binary object. This is inside a folder at the
-- root of the user's home directory.
--
-- Some platforms may have directories assigned to store cache files; Mac OS X
-- uses ~/Library/Caches, for example. This fact is ignored.
--
cacheDirectory :: IO FilePath
cacheDirectory = do
  home  <- getAppUserDataDirectory "accelerate"
  return $ home </> "accelerate-cuda-" ++ showVersion version </> "cache"


-- A relative path to be appended to (presumably) 'cacheDirectory'.
--
cacheFilePath :: KernelKey -> FilePath
cacheFilePath (cap, key) =
  show cap </> zEncodeString (BS.foldl (flip (showLitChar . w2c)) [] key)

-- stolen from compiler/utils/Encoding.hs
--
type EncodedString = String

zEncodeString :: String -> EncodedString
zEncodeString []       = []
zEncodeString (h:rest) = encode_digit h ++ go rest
  where
    go []     = []
    go (c:cs) = encode_ch c ++ go cs

unencodedChar :: Char -> Bool
unencodedChar 'z' = False
unencodedChar 'Z' = False
unencodedChar c   = isAlphaNum c

encode_digit :: Char -> EncodedString
encode_digit c | isDigit c = encode_as_unicode_char c
               | otherwise = encode_ch c

encode_ch :: Char -> EncodedString
encode_ch c | unencodedChar c = [c]     -- Common case first
encode_ch '('  = "ZL"
encode_ch ')'  = "ZR"
encode_ch '['  = "ZM"
encode_ch ']'  = "ZN"
encode_ch ':'  = "ZC"
encode_ch 'Z'  = "ZZ"
encode_ch 'z'  = "zz"
encode_ch '&'  = "za"
encode_ch '|'  = "zb"
encode_ch '^'  = "zc"
encode_ch '$'  = "zd"
encode_ch '='  = "ze"
encode_ch '>'  = "zg"
encode_ch '#'  = "zh"
encode_ch '.'  = "zi"
encode_ch '<'  = "zl"
encode_ch '-'  = "zm"
encode_ch '!'  = "zn"
encode_ch '+'  = "zp"
encode_ch '\'' = "zq"
encode_ch '\\' = "zr"
encode_ch '/'  = "zs"
encode_ch '*'  = "zt"
encode_ch '_'  = "zu"
encode_ch '%'  = "zv"
encode_ch c    = encode_as_unicode_char c

encode_as_unicode_char :: Char -> EncodedString
encode_as_unicode_char c
  = 'z'
  : if isDigit (head hex_str) then hex_str
                              else '0':hex_str
  where
    hex_str = showHex (ord c) "U"


-- The default Binary instance for lists is (necessarily) spine and value
-- strict for efficiency. For us it is better if we just lazily consume elements
-- and add them directly to the hash table so they can be collected as we go.
--
{-# INLINE getMany #-}
getMany :: Binary a => Int -> Get [a]
getMany n = go n []
  where
    go 0 xs = return xs
    go i xs = do
      x <- get
      go (i-1) (x:xs)


-- Load the entire persistent cache index file. If it does not exist, an empty
-- file is created, so that 'persist' can always append elements.
--
restore :: FilePath -> IO PersistentCache
restore !db = do
  mflush <- D.queryFlag D.flush_cache
  when (fromMaybe False mflush) $ do
    message "deleting persistent cache"
    cacheDir <- cacheDirectory
    removeDirectoryRecursive cacheDir
    createDirectoryIfMissing True cacheDir
  --
  exists <- doesFileExist db
  pt     <- case exists of
    False       -> encodeFile db (0::Int) >> HT.new
    True        -> do
      store         <- BL.readFile db

      -- Just read the start of the input to extract the number of entries
      -- in the persistent kernel table.
      let (n, rest) = setup (runGetIncremental get) store

          setup (Done s _ r)   lbs = (r, BL.Chunk s lbs)
          setup (Partial k)    lbs = setup (k (takeHeadChunk lbs)) (dropHeadChunk lbs)
          setup (Fail _ p msg) _   = $internalError "restore" $ show p ++ ": " ++ msg

          takeHeadChunk (BL.Chunk h _) = Just h
          takeHeadChunk _              = Nothing

          dropHeadChunk (BL.Chunk _ t) = t
          dropHeadChunk _              = BL.empty

      -- Allocate the persistent hash table and populate it with entries decoded
      -- from the index file
      pt            <- HT.newSized n
      let go []      = return ()
          go (!k:xs) = HT.insert pt k () >> go xs

      message $ "persist/restore: " ++ shows n " entries"
      go (runGet (getMany n) rest)
      evaluate pt
  --
  newMVar pt


-- Append a single value to the persistent cache.
--
-- This moves the compiled object file (first argument) to the appropriate
-- location, and updates the database on disk.
--
persist :: KernelTable -> FilePath -> KernelKey -> IO ()
persist (KT !_ !pt_ref) !cubin !key = withMVar pt_ref $ \_ -> do
  cacheDir <- cacheDirectory
  let db        = cacheDir </> "persistent.db"
      cacheFile = cacheDir </> cacheFilePath key
  --
  message $ "persist/save: " ++ cacheFile
  createDirectoryIfMissing True (dropFileName cacheFile)
  renameFile cubin cacheFile
    -- If the temporary and cache directories are on different disks, we must
    -- copy the file instead. Unsupported operation: (Cross-device link)
    --
    `catchIOError` \_ -> do
      copyFile cubin cacheFile
      removeFile cubin
  --
  withBinaryFile db ReadWriteMode $ \h -> do
    -- The file opens with the cursor at the beginning of the file
    --
    n <- runGet (get :: Get Int) `fmap` BL.hGet h 8
    hSeek h AbsoluteSeek 0
    BL.hPut h (encode (n+1))

    -- Append the new entry to the end of file
    --
    hSeek h SeekFromEnd 0
    BL.hPut h (encode key)


-- Debug
-- -----

{-# INLINE message #-}
message :: String -> IO ()
message msg = D.traceIO D.dump_cc ("cc: " ++ msg)

