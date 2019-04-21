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
     -- conv_output <- readFile "conv_results/oz_data"
     -- let ilines = lines conv_output :: [String]
     --     slines = map (\y ->  y ++ "\n") (map (head . splitOn ",") ilines)
     -- writeFile "conv_results/oz_labels" $ concat slines

     readDataConv

     -- dna <- readDNA :: IO [[([Double], [Double])]]
     -- print dna
     -- runDNA dna
     -- runRecurrent'
     -- inputFile <- readFile "fullyconnected_results/sine_data_800"
     -- outputFile <- readFile "fullyconnected_results/sine_labels_800"
     -- let inputlines = lines inputFile
     --     outputlines = lines outputFile
     --     input = map read inputlines :: [Double]
     --     output = map read outputlines :: [Double]

     -- fc_network <- runFCNetwork (map (\x ->  replicate 3 x) input) (map (\x -> [x]) output)
     -- print $ show fc_network
     --print $ show $ runConvolutional


readDataConv = do 
     inputFile <- readFile "conv_results/oz_data_300"
     outputFile <- readFile "conv_results/oz_labels_300"
     let  inputlines = lines inputFile 
          outputlines = lines outputFile
          input = map (\x -> [x]) $ map (chunksOf 7) ((map2 read (map (splitOn ",") inputlines)) ::  [[Double]])
          output = map ((\x -> if x == 0 then [[[1]], [[0]]] else [[[0]], [[1]]]) . read)  outputlines
     nn <- runConvolutional input output
     print $ show nn 

makeDataFC = do 
     args <- getArgs
     content <- readFile (args !! 0)
     let linesOfFile = lines content
         numbers = map read linesOfFile
         output  = map sin numbers 
         output' = map ((\x -> x ++ "\n") . (\z -> formatFloatN z 8)) output
     writeFile "fullyconnected_results/sine_labels_1400" $ concat output'

readDNA :: IO [[([Double], [Double])]]
readDNA = do 
     formatDNA
     inputFile <- readFile "rnn_results/dna_300_data"
     let linesOfFile = lines inputFile 
         dna = mapDNA linesOfFile
     return dna
     
formatDNA = do 
     inputFile <- readFile "rnn_results/dna_300"
     let linesOfFile = lines inputFile
         dna_data = map (\x -> x ++ "\n") $ (filter (\x -> length x == 6) linesOfFile) 
     writeFile "rnn_results/dna_300_data" $ concat dna_data

mapDNA :: [String] -> [[([Double], [Double])]]
mapDNA s = 
     map f s
     where f :: String -> [([Double], [Double])]
           f dna = let dna_strand = map (\z -> case z of    'a' -> 0.2
                                                            'c' -> 0.4
                                                            'g' -> 0.6
                                                            't' -> 0.8) dna
                       input = init dna_strand 
                       desired_output = tail dna_strand 
                       res = map (mapT2 (\x -> [x])) $ zip input desired_output
                   in  res

-- formatDataRNN = do 
--      contents <- mapM readFile ["t" ++ show n ++ ".csv" | n <- [1..11]]
--      let linesOfFiles = map lines contents 
--          numbers = map2 read lines

readShiftRight = do 
     args <- getArgs
     content <- readFile (args !! 0)
     let  linesOfFiles = lines content
          numbers = map read linesOfFiles
          output' = map ((\x -> x ++ "\n") . (\z -> showFullPrecision $ read $ formatFloatN (z/100) 6)) numbers
     writeFile "sequence" $ concat output'

