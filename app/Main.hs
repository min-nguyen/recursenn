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
import Data.Ord
import Text.Show.Functions
import qualified Vector as V
import Vector (Vector((:-)))
import Debug.Trace
import FullyConnected2
import TestSuite

main = do 
     readShiftRight
     -- inputFile <- readFile "sine_data"
     -- outputFile <- readFile "sine_outputs"
     -- let inputlines = lines inputFile
     --     outputlines = lines outputFile
     --     input = map read inputlines :: [Double]
     --     output = map read outputlines :: [Double]

     --     nn = runSineNetwork (map (\x ->  replicate 3 x) input) (map (\x -> [x]) output)
     -- print $ show nn
     --print $ show $ runConvolutional


makeData = do 
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