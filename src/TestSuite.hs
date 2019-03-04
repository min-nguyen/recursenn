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
     FlexibleContexts #-}

module TestSuite where

import Utils
import Data.Functor     
import Data.Foldable
import Data.Traversable
import Data.List.Split
import Data.Ord
import Text.Show.Functions
import qualified Vector as Vector
import Vector (Vector((:-)))
import Debug.Trace
import Data.List (transpose, elemIndex)
import Data.Maybe (fromJust)
import Control.Lens hiding (Index)
import Convolution 

runConvolutionTest :: IO ()
runConvolutionTest = do 
    testPooling
    testFlatten
    -- testUnflatten
    -- testDeltaFullyConnected
--     map3 snd (pool 1 2 ( map3 (\x -> ((0,0), x))  [[[ 0.20,  0.60,  0.70, 0.30], 
--                                                     [-0.1 ,  0.5,   0.25, 0.50], 
--                                                     [ 0.75, -0.50, -0.80, 0.40], 
--                                                     [-0.1 ,  0.5,   0.25, 0.50]],
--                                                    [[-0.35,  0.30,  0.80,  0.00], 
--                                                     [ 0.2 ,  0.2,   0.00,  1.00], 
--                                                     [-0.10, -0.40, -0.10, -0.40], 
--                                                     [-0.1 ,  0.5,   0.25,  0.5]],
--                                                    [[ 0.25,  0.25, -0.25, -0.25], 
--                                                     [ 0.5 ,  0.8,   0.12, -0.12], 
--                                                     [ 0.34, -0.34, -0.90,  0.65], 
--                                                     [-0.1 ,  0.5,   0.25,  0.5]]]))

--                                          [[[0.6, 0.7, 0.5], [0.75, 0.5, 0.5], [0.75, 0.25, 0.5]],
--                                           [[0.3, 0.8, 1.0], [0.2, 0.2, 1.0], [0.5, 0.5, 0.5]],
--                                           [[0.8, 0.8, 0.23], [0.8, 0.8, 0.65], [0.5, 0.5, 0.65]]]

testPooling :: IO ()
testPooling = do 
    let result = map3 snd (map (pool 1 2) (( map3 (\x -> ((0,0), x))  [[[ 0.20,  0.60,  0.70, 0.30], 
                                                                    [-0.1 ,  0.5,   0.25, 0.50], 
                                                                    [ 0.75, -0.50, -0.80, 0.40], 
                                                                    [-0.1 ,  0.5,   0.25, 0.50]],
                                                                [[-0.35,  0.30,  0.80,  0.00], 
                                                                    [ 0.2 ,  0.2,   0.00,  1.00], 
                                                                    [-0.10, -0.40, -0.10, -0.40], 
                                                                    [-0.1 ,  0.5,   0.25,  0.5]],
                                                                [[ 0.25,  0.25, -0.25, -0.25], 
                                                                    [ 0.5 ,  0.8,   0.12, -0.12], 
                                                                    [ 0.34, -0.34, -0.90,  0.65], 
                                                                    [-0.1 ,  0.5,   0.25,  0.5]]]) :: Image))
        expectedAnswer =    [[[0.6, 0.7, 0.7], [0.75, 0.5, 0.5], [0.75, 0.5, 0.5]],
                             [[0.3, 0.8, 1.0], [0.2, 0.2, 1.0], [0.5, 0.5, 0.5]],
                             [[0.8, 0.8, 0.12], [0.8, 0.8, 0.65], [0.5, 0.5, 0.65]]]

    print (if result == expectedAnswer then "Pooling Test Correct" else "Pooling Test Failed")

testFlatten :: IO ()
testFlatten = do 
    let expectedAnswer = [[[((0,0), 0.6)]], [[((0,1), 0.7)]], [[((0,2), 0.7)]], [[((1,0), 0.75)]], [[((1,1), 0.5)]], [[((1,2), 0.5)]], [[((2,0), 0.75)]], [[((2,1), 0.5)]], [[((2,2), 0.5)]],
                            [[((3,0), 0.3)]], [[((3,1), 0.8)]], [[((3,2), 1.0)]], [[((4,0), 0.2)]], [[((4,1), 0.2)]], [[((4,2), 1.0)]], [[((5,0), 0.5)]], [[((5,1), 0.5)]], [[((5,2), 0.5)]],
                            [[((6,0), 0.8)]], [[((6,1), 0.8)]], [[((6,2), 0.12)]], [[((7,0), 0.8)]], [[((7,1), 0.8)]], [[((7,2), 0.65)]], [[((8,0), 0.5)]], [[((8,1), 0.5)]], [[((8,2), 0.65)]]]

        image =         [[[((0,0), 0.6), ((0,1), 0.7), ((0,2), 0.7)], [((1,0), 0.75), ((1,1), 0.5), ((1,2), 0.5)], [((2,0), 0.75), ((2,1), 0.5), ((2,2), 0.5)]],
                        [[((3,0), 0.3), ((3,1), 0.8), ((3,2), 1.0)], [((4,0), 0.2), ((4,1), 0.2), ((4,2), 1.0)], [((5,0), 0.5), ((5,1), 0.5), ((5,2), 0.5)]],
                        [[((6,0), 0.8), ((6,1), 0.8), ((6,2), 0.12)], [((7,0), 0.8), ((7,1), 0.8), ((7,2), 0.65)], [((8,0), 0.5), ((8,1), 0.5), ((8,2), 0.65)]]]

    print (if flattenImage image == expectedAnswer then "Flatten Test Correct" else "Flatten Test Failed")

-- testUnflatten :: IO ()
-- testUnflatten = do 
--     let image = [[[((0,0), 0.6)]], [[((0,1), 0.7)]], [[((0,2), 0.7)]], [[((1,0), 0.75)]], [[((1,1), 0.5)]], [[((1,2), 0.5)]], [[((2,0), 0.75)]], [[((2,1), 0.5)]], [[((2,2), 0.5)]],
--                             [[((3,0), 0.3)]], [[((3,1), 0.8)]], [[((3,2), 1.0)]], [[((4,0), 0.2)]], [[((4,1), 0.2)]], [[((4,2), 1.0)]], [[((5,0), 0.5)]], [[((5,1), 0.5)]], [[((5,2), 0.5)]],
--                             [[((6,0), 0.8)]], [[((6,1), 0.8)]], [[((6,2), 0.12)]], [[((7,0), 0.8)]], [[((7,1), 0.8)]], [[((7,2), 0.65)]], [[((8,0), 0.5)]], [[((8,1), 0.5)]], [[((8,2), 0.65)]]]

--         expectedAnswer =         [[[((0,0), 0.6), ((0,1), 0.7), ((0,2), 0.7)], [((1,0), 0.75), ((1,1), 0.5), ((1,2), 0.5)], [((2,0), 0.75), ((2,1), 0.5), ((2,2), 0.5)]],
--                         [[((3,0), 0.3), ((3,1), 0.8), ((3,2), 1.0)], [((4,0), 0.2), ((4,1), 0.2), ((4,2), 1.0)], [((5,0), 0.5), ((5,1), 0.5), ((5,2), 0.5)]],
--                         [[((6,0), 0.8), ((6,1), 0.8), ((6,2), 0.12)], [((7,0), 0.8), ((7,1), 0.8), ((7,2), 0.65)], [((8,0), 0.5), ((8,1), 0.5), ((8,2), 0.65)]]]

--     print (if ((unflattenImage (image) (3, 3 ,3) ) == expectedAnswer) then "Unflatten Test Correct" else "Unflatten Test Failed")

-- testDeltaFullyConnected :: IO ()
-- testDeltaFullyConnected = do 
--     let outputimage = [[[((0,0), 0.6)]], [[((0,1), 0.7)]], [[((0,2), 0.7)]], [[((1,0), 0.75)]], [[((1,1), 0.5)]], [[((1,2), 0.5)]]]
--         desiredOutput = [[[0.2]],[[0.1]],[[0.7]], [[0.25]], [[0.0]], [[0.63]]]
--         expectedAnswer = [[[[ 0.5 * (0.6 - 0.2), 0.5 * (0.7 - 0.1) ]], [[0.5 * (0.7 - 0.7), 0.5 * (0.75 - 0.25)]] , [[0.5 * (0.5 - 0.0), 0.5 * (0.5 - 0.63)]]]]
    
--     print (if compDeltaFullyConnected outputimage desiredOutput (1, 2, 3) == expectedAnswer then "Delta Fully Connected Correct" else "Delta Fully Connected Failed")