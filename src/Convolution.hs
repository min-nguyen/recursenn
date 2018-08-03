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

import Lib
import Data.Functor     
import Data.Foldable
import Data.Traversable
import Data.List.Split
import Data.Ord
import Text.Show.Functions
import qualified Vector as Vector
import Vector (Vector((:-)))
import Debug.Trace


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
type Image              = [[[Double]]]       
type ImageStack         = [Image]
type Stride             = Int
type SpatialExtent      = Int
type Biases             = [Double]
type Deltas             = [[[Double]]]
type DesiredOutput      = [[[Double]]]
data BackPropData       = BackPropData {
                                    imageStack      :: ImageStack,
                                    outerDeltas     :: Deltas,
                                    outerFilters    :: [Filter],
                                    desiredOutput   :: DesiredOutput
                                }
---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |                          Alg & Coalg                           | | ---
  --- ‾------------------------------------------------------------------‾---

flatten :: Fractional a => [[a]] -> SpatialExtent -> Stride -> [[a]]
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

convolute2D :: Fractional a => [[a]] -> [[a]] -> Stride -> [[a]]
convolute2D filter image stride
    = let flat_image = flatten image (length filter) stride
      in  chunksOf (length filter) $ map  (sum . (zipWith (*) (concat filter))) flat_image

convolute3D :: Fractional a => [[[a]]] -> [[[a]]] -> Stride -> [[a]]
convolute3D filter image stride
    = let n     = length filter
          bias  = 1.0
      in  map (map (bias + )) $ foldr eleaddm (fillMatrix n n 0.0) [  convolute2D filter2d image2d stride |  (image2d, filter2d) <- (zip image filter)]


pool :: Fractional a =>  Stride -> SpatialExtent -> [[a]] -> [[a]]
pool stride spatialExtent image = 
    let flat_image = flatten image spatialExtent stride
    in  chunksOf ((quot (length (head image) - spatialExtent) stride) - 1) $ map sum flat_image

alg :: CNNLayer (Fix CNNLayer, (ImageStack -> ImageStack) ) -> (Fix CNNLayer, (ImageStack -> ImageStack))
alg (ConvolutionalLayer filters biases (innerLayer, forwardPass))
        = (Fx (ConvolutionalLayer filters biases innerLayer), (\imageStacks -> 
            let inputVolume = (head imageStacks) 
                stride = 1
            in  (([(convolute3D filter) inputVolume 1 | filter <- filters]):imageStacks) ) . forwardPass)
alg (PoolingLayer stride spatialExtent (innerLayer, forwardPass))
        = (Fx (PoolingLayer stride spatialExtent innerLayer), (\imageStack -> ((map (pool stride spatialExtent) (head imageStack)):imageStack) ) . forwardPass  )
alg (ReluLayer (innerLayer, forwardPass))
        = (Fx (ReluLayer innerLayer), (\imageStacks -> ((map3 abs (head imageStacks)):imageStacks) ) . forwardPass)
alg (InputLayer) = (Fx InputLayer, id)




-- coalg :: (Fix CNNLayer, BackPropData) -> CNNLayer (Fix CNNLayer, BackPropData )
-- coalg (Fx (FullyConnectedLayer innerLayer), BackPropData imageStack outerDeltas desiredOutput)
--         =   let actualOutput = (head imageStack)
--                 deltas       = [  [ 0.5 * (zipWith (-))  |  (a, d) <- (zip actOutput2d desOutput2d) ]    |  (actOutput2d, desOutput2d) <- (zip actualOutput desiredOutput)  ]
--             in  FullyConnectedLayer (Fx innerLayer, BackPropData (tail imageStack) deltas desiredOutput)
-- coalg (Fx (ConvolutionalLayer filters biases innerLayer), BackPropData imageStack outerDeltas desiredOutput)
--         =   let deltas = elemul (mvmul (map2 transpose outerFilters) outerDeltas) (map derivActivation inputs)
--                 newFilters = convolute3D deltas (head imageStack) 
--             in  ConvolutionalLayer newFilters biases (innerLayer, BackPropData (tail imageStack) deltas desiredOutput)





















-- Each filter is applied to the entire depth of images in the current stack of 2D images, so
-- must have the same depth as the input volume
-- Each filter will produce a separate 2D activation map (output volume)
-- Each output volume by a filter, can be interpreted as an output of a neuron
-- Given a receptive field size of 3x3 and input volume of 16x16x20, every neuron in the conv layer
-- would now have a total of 3*3*20 = 180 connections to the input volume. The connectivity is local
-- in space (3x3) but along the full input depth (20).
-- The depth of the output volume is the number of filters we used
