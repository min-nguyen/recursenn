{-# LANGUAGE
     DeriveFunctor,
     DeriveFoldable,
     DeriveTraversable,
     UndecidableInstances,
     FlexibleInstances,
     ScopedTypeVariables,
     MultiParamTypeClasses,
     FlexibleContexts,
     TypeFamilies,
     GADTs,
     DataKinds,
     KindSignatures,
     RecordWildCards,
     ExistentialQuantification #-}

module Main where

import System.Environment
import Numeric 
import Utils
import Data.Functor     
import Data.Foldable
import Data.Traversable
import Data.List
import Data.List.Split
import Data.Ord
import Text.Read
import Text.Show.Functions
import qualified Vector as V
import Vector (Vector((:-)))
import Debug.Trace
import Convolution
import TestSuite

main = do 
     
     readDataConv
     -- inputFile <- readFile "sine_data"
     -- outputFile <- readFile "sine_outputs"
     -- let inputlines = lines inputFile
     --     outputlines = lines outputFile
     --     input = map read inputlines :: [Double]
     --     output = map read outputlines :: [Double]

     --     nn = runSineNetwork (map (\x ->  replicate 3 x) input) (map (\x -> [x]) output)
     -- print $ show nn
     --print $ show $ runConvolutional


readDataConv = do 
     inputFile <- readFile "conv_data"
     outputFile <- readFile "conv_output"
     let  inputlines = lines inputFile 
          outputlines = lines outputFile
          -- print $ map length ((map2 read (map (splitOn ",") inputlines)) ::  [[Double]])
          input = map (\x -> [x]) $ map (chunksOf 7) ((map2 read (map (splitOn ",") inputlines)) ::  [[Double]])
     --      input = map (chunksOf 5) $ map2 read $ map (splitOn ",") inputlines 
          output = map ((\x -> [[[x]]]) . read)  outputlines

          nn = runConvolutionalV3 input output
     print $ show nn 
     -- print $ input
     -- print $ output
     -- print $ show input

makeDataFC = do 
     args <- getArgs
     content <- readFile (args !! 0)
     let linesOfFiles = lines content
         numbers = map read linesOfFiles
         output  = map sin numbers 
         output' = map ((\x -> x ++ "\n") . (\z -> formatFloatN z 10)) output
     writeFile "sine_outputs" $ concat output'

readShiftRight = do 
     args <- getArgs
     content <- readFile (args !! 0)
     let  linesOfFiles = lines content
          numbers = map read linesOfFiles
          output' = map ((\x -> x ++ "\n") . (\z -> formatFloatN (z/10) 5)) numbers
     writeFile "sequence" $ concat output'