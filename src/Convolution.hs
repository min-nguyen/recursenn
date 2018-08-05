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
     RecordWildCards #-}

module Convolution where

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
---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |                        Convolutional NN                        | | ---
  --- ‾------------------------------------------------------------------‾---


data CNNLayer k where
    InputLayer              :: CNNLayer k
    ConvolutionalLayer      :: [Filter] -> [Biases] -> k -> CNNLayer k
    ReluLayer               :: k -> CNNLayer k 
    PoolingLayer            :: Stride -> SpatialExtent -> k -> CNNLayer k 
    FullyConnectedLayer     :: k -> CNNLayer k
    deriving (Functor, Show)

type Filter             = [[[Double]]]       
type Image              = [[[(Int, Double)]]]       
type ImageStack         = [Image]
type Stride             = Int
type SpatialExtent      = Int
type Biases             = [Double]
type Deltas             = [[[Double]]]
type DesiredOutput      = [[[Double]]]
data BackPropData       = BackPropData {
                                    imageStack      :: ImageStack,
                                    outerDeltas     :: [Deltas],
                                    outerFilters    :: [Filter],
                                    desiredOutput   :: DesiredOutput
                                }

---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |                    Forward & Back Propagation                  | | ---
  --- ‾------------------------------------------------------------------‾---

flatten :: Fractional a => [[(Int, a)]] -> SpatialExtent -> Stride -> [[(Int, a)]]
flatten image spatialExtent stride =
    let splitVertical image' stackArray =   
                                if length image' < spatialExtent 
                                then stackArray
                                else (splitHorizontal image' (take spatialExtent image') stackArray)
        splitHorizontal image'' imageChunk stack' = case () of 
                                _ | length (head imageChunk) < spatialExtent -> (splitVertical (drop stride image'') stack')
                                _ | otherwise -> let new_stack = (stack' ++ (concat $ map (take spatialExtent) $ take spatialExtent imageChunk))
                                                 in  splitHorizontal image'' (map (drop stride) imageChunk) new_stack
    in chunksOf (spatialExtent*spatialExtent) (splitVertical image [])

convolute2D :: Fractional a => [[a]] -> [[(Int, a)]] -> Stride -> [[(Int, a)]]
convolute2D filter image stride
    = let flat_image = flatten image (length filter) stride
      in  chunksOf (length filter) $ zip [0 ..] $ map (sum . (zipWith (*) (concat filter)) . (map snd)) flat_image

convolute3D :: Fractional a => [[[a]]] -> [[[(Int, a)]]] -> Stride -> [[[(Int, a)]]]
convolute3D filter image stride
    =  [  convolute2D filter2d image2d stride |  (image2d, filter2d) <- (zip image filter)]

forward :: Fractional a => [[[a]]] -> [[[(Int, a)]]] -> Stride -> [[(Int, a)]]
forward filter image stride 
    = let n     = length filter
          bias  = 1.0
      in  map ((zip [0 ..]) . (map (bias + ))) $ foldr eleaddm (fillMatrix n n 0.0) (map3 snd $ convolute3D filter image stride)

pool :: (Ord a, Fractional a) =>  Stride -> SpatialExtent -> [[(Int, a)]] -> [[(Int, a)]]
pool stride spatialExtent image = 
    let flat_image = flatten image spatialExtent stride
        image_nums = map2 snd flat_image
    in  chunksOf ((quot (length (head image) - spatialExtent) stride) + 1) $ 
            map (\xs -> let x = (maximum xs)
                        in  (fromJust $ elemIndex x xs, x) ) image_nums

---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |                          Alg & Coalg                           | | ---
  --- ‾------------------------------------------------------------------‾---

alg :: CNNLayer (Fix CNNLayer, (ImageStack -> ImageStack) ) -> (Fix CNNLayer, (ImageStack -> ImageStack))
alg (ConvolutionalLayer filters biases (innerLayer, forwardPass))
        = (Fx (ConvolutionalLayer filters biases innerLayer), (\imageStacks -> 
            let inputVolume = (head imageStacks) 
                stride = 1
            in  (([forward filter inputVolume 1 | filter <- filters]):imageStacks)) . forwardPass)
alg (PoolingLayer stride spatialExtent (innerLayer, forwardPass))
        = (Fx (PoolingLayer stride spatialExtent innerLayer), 
                (\imageStack -> 
                    ((map (pool stride spatialExtent) (head imageStack)):imageStack)) . forwardPass)
alg (ReluLayer (innerLayer, forwardPass))
        = (Fx (ReluLayer innerLayer), 
                (\imageStacks -> ((map3 (\x -> (0, abs $ snd x)) (head imageStacks)):imageStacks) ) . forwardPass)
alg (InputLayer) 
        = (Fx InputLayer, id)
alg (FullyConnectedLayer (innerLayer, forwardPass)) 
        = (Fx (FullyConnectedLayer innerLayer), id)

-- missing activation function

-- coalg :: (Fix CNNLayer, BackPropData) -> CNNLayer (Fix CNNLayer, BackPropData )
-- coalg (Fx (FullyConnectedLayer innerLayer), BackPropData imageStack outerDeltas outerFilters desiredOutput)
--         =   let actualOutput = (head imageStack)
--                 deltas       = [ [ [map (0.5 *) (zipWith (-) a d)]  
--                                         |  (a, d) <- (zip actOutput2d desOutput2d) ]    
--                                             |  (actOutput2d, desOutput2d) <- (zip actualOutput desiredOutput)  ]
--             in  FullyConnectedLayer (Fx (FullyConnectedLayer innerLayer), BackPropData (tail imageStack) deltas outerFilters desiredOutput)
-- coalg (Fx (ConvolutionalLayer filters biases innerLayer), BackPropData imageStack outerDeltas outerFilters desiredOutput)
--         =   let input           = head (tail imageStack)
--                 learningRate    = 0.1
--                 deltaW          = [ convolute3D outerDelta (transpose3D input) 1 
--                                             |  outerDelta <- outerDeltas ] :: [Deltas]
--                 deltaX          = [ convolute3D (transpose3D filter) outerDelta 1
--                                             |  (outerDelta, filter) <- zip outerDeltas filters ] :: [Deltas]
--                 newFilters      = [ zipWith elesubm filter (map3 (learningRate *) delta_w) 
--                                             | (filter, delta_w) <- (zip filters deltaW) ] 
--             in  ConvolutionalLayer newFilters biases (innerLayer, BackPropData (tail imageStack) deltaX newFilters desiredOutput)
-- coalg (Fx (PoolingLayer stride spatialExtent innerLayer), BackPropData imageStack outerDeltas outerFilters desiredOutput)
--         =   let input           = head (tail imageStack)
--                 deltaX          = [ convolute3D outerDelta (transpose3D input)1
--                                     |  outerDelta <- outerDeltas ] :: [Deltas]
--             in  (PoolingLayer stride spatialExtent (innerLayer, BackPropData (tail imageStack) deltaX outerFilters desiredOutput) )
-- coalg (Fx (ReluLayer innerLayer), BackPropData imageStack outerDeltas outerFilters desiredOutput)
--         =   let input           = head (tail imageStack)
--                 deltaX          = [ convolute3D outerDelta (transpose3D input)1
--                                     |  outerDelta <- outerDeltas ] :: [Deltas]
--             in  (ReluLayer (innerLayer,  BackPropData (tail imageStack) deltaX outerFilters desiredOutput) )
-- coalg  (Fx InputLayer, backPropData)
--         =   InputLayer
















-- Each filter is applied to the entire depth of images in the current stack of 2D images, so
-- must have the same depth as the input volume
-- Each filter will produce a separate 2D activation map (output volume)
-- Each output volume by a filter, can be interpreted as an output of a neuron
-- Given a receptive field size of 3x3 and input volume of 16x16x20, every neuron in the conv layer
-- would now have a total of 3*3*20 = 180 connections to the input volume. The connectivity is local
-- in space (3x3) but along the full input depth (20).
-- The depth of the output volume is the number of filters we used
