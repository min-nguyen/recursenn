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
import Control.Lens hiding (Index)
---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |                        Convolutional NN                        | | ---
  --- ‾------------------------------------------------------------------‾---


data Layer k where
    InputLayer              :: Layer k
    ConvolutionalLayer      :: [Filter] -> [Biases] -> k -> Layer k
    ReluLayer               :: k -> Layer k 
    PoolingLayer            :: Stride -> SpatialExtent -> k -> Layer k 
    FullyConnectedLayer     :: k -> Layer k
    deriving (Functor, Show)

type Filter             = [[[Double]]]       
type Index              = (Int, Int)
type Image2D            = [[(Index, Double)]] 
type Image              = [Image2D]    
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
 --- | |                          Alg & Coalg                           | | ---
  --- ‾------------------------------------------------------------------‾---

alg :: Layer (Fix Layer, (ImageStack -> ImageStack) ) -> (Fix Layer, (ImageStack -> ImageStack))
alg (ConvolutionalLayer filters biases (innerLayer, forwardPass))
        = (Fx (ConvolutionalLayer filters biases innerLayer), (\imageStacks -> 
            let inputVolume = (head imageStacks) 
                stride = 1
            in  (([map2 (\(a,b) -> (a, sigmoid b)) (forward filter inputVolume 1) | filter <- filters]):imageStacks)) . forwardPass)
alg (PoolingLayer stride spatialExtent (innerLayer, forwardPass))
        = (Fx (PoolingLayer stride spatialExtent innerLayer), 
                (\imageStack -> 
                    ((map (pool stride spatialExtent) (head imageStack)):imageStack)) . forwardPass)
alg (ReluLayer (innerLayer, forwardPass))
        = (Fx (ReluLayer innerLayer), 
                (\imageStacks -> ((map3 (\x -> ((0,0), abs $ snd x)) (head imageStacks)):imageStacks) ) . forwardPass)
alg (InputLayer) 
        = (Fx InputLayer, id)
alg (FullyConnectedLayer (innerLayer, forwardPass)) 
        = (Fx (FullyConnectedLayer innerLayer), (\imageStack -> ((flattenImage $ head imageStack) : imageStack) ) . forwardPass )

coalg :: (Fix Layer, BackPropData) -> Layer (Fix Layer, BackPropData)
coalg (Fx (FullyConnectedLayer innerLayer), BackPropData imageStack outerDeltas outerFilters desiredOutput)
        =   let (actualOutput:input:_) = (imageStack)
        
                (m, n, v)   = (length (head $ head input), length (head input), length input)

                deltas       = compDeltaFullyConnected actualOutput desiredOutput (m, n, v)

            in trace ("m: " ++ show deltas)  $  FullyConnectedLayer (innerLayer, BackPropData (tail imageStack) deltas outerFilters desiredOutput)

coalg (Fx (ConvolutionalLayer filters biases innerLayer), BackPropData imageStack outerDeltas outerFilters desiredOutput)
        =   let output          = head imageStack
                input           = head (tail imageStack)
                learningRate    = 0.1

                deltaX          = [ (mmmul3d wTdelta (map3 (sigmoid' . snd) input) )
                                            |  (outerDelta, filter) <- zip outerDeltas filters, 
                                                let wTdelta = (convoluteDeltaX (head outerDelta)  (transpose3D filter) 1)] :: [Deltas]

                deltaW          = [ (convoluteDeltaW (head outerDelta) (map3 (sigmoid' . snd) $ transpose3D input) 1)
                                            |  (outerDelta) <- (outerDeltas)] :: [Deltas]

                newFilters      = [ zipWith elesubm filter (map3 (learningRate *) delta_w) 
                                            | (filter, delta_w) <- (zip filters deltaW) ] 

            in  trace ("Input: " ++ show input ++ "\n Output:" ++ show output ++ "\n Delta:" ++ show deltaX)  $ ConvolutionalLayer newFilters biases (innerLayer, BackPropData (tail imageStack) deltaX newFilters desiredOutput)
     
coalg (Fx (PoolingLayer stride spatialExtent innerLayer), BackPropData imageStack outerDeltas outerFilters desiredOutput)
        =   let input           = head (tail imageStack)
                output          = head imageStack
                delta           = [[unpool (length $ head input2d, length $ input2d) output2d] | (input2d, output2d) <- zip input output  ]
            in  (PoolingLayer stride spatialExtent (innerLayer, BackPropData (tail imageStack) delta outerFilters desiredOutput) )

coalg (Fx (ReluLayer innerLayer), BackPropData imageStack outerDeltas outerFilters desiredOutput)
        =   let input           = head (tail imageStack)
                delta          = [ convolute3D outerDelta (map3 snd $ transpose3D input) 1
                                    |  outerDelta <- outerDeltas ] :: [Deltas]
            in  (ReluLayer (innerLayer,  BackPropData (tail imageStack) delta outerFilters desiredOutput) )
coalg  (Fx InputLayer, backPropData)
        =   InputLayer


train :: Fix Layer -> Image -> DesiredOutput -> Fix Layer 
train neuralnet sample desiredoutput 
    = --trace (show $ head inputStack) $ 
        ana coalg $ (nn, BackPropData inputStack [[[[]]]] [[[[]]]] desiredoutput)
            where 
                (nn, diff_fun)      = cata alg neuralnet
                inputStack          = diff_fun [sample]
        
h = map3 (\x -> ((0,0), x))
pad = convoluteDeltaX (head [[[0.5, -0.5], [-0.5, 0.5]], 
                            [[0.8, 0.8], [-0.8, 0.8]], 
                            [[1.0, -1.0], [1.0, -1.0]]]) (([[[0.2, 0.6, 0.7,0.3],       [-0.1, 0.5, 0.25, 0.5],  [0.75, -0.5, -0.8, 0.4] , [-0.1, 0.5, 0.25, 0.5]],
                                                            [[-0.35, 0.3, 0.8, 0.0],    [0.2, 0.2, 0.0, 1.0],    [-0.1, -0.4, -0.1, -0.4], [-0.1, 0.5, 0.25, 0.5]],
                                                            [[0.25, 0.25, -0.25, -0.25],[0.5, 0.8, 0.12, -0.12], [0.34, -0.34, -0.9, 0.65], [-0.1, 0.5, 0.25, 0.5]]] )) 1

example = Fx (FullyConnectedLayer (Fx $ PoolingLayer 1 2 (Fx $ ConvolutionalLayer [[[[0.5, -0.5], [-0.5, 0.5]], 
                                                                                    [[0.8, 0.8], [-0.8, 0.8]], 
                                                                                    [[1.0, -1.0], [1.0, -1.0]]], 
                                                                                   [[[0.2, -0.1], [0.5, 0.5]], 
                                                                                    [[0.3, -0.8], [-0.1, 0.3]], 
                                                                                    [[0.0, -0.3], [0.3, -0.4]]]] [[0.0], [0.0]] (Fx $ InputLayer))))

runConvolutional = --head $ map3 (map (\(a, f) -> (a, (fromInteger $ round $ f * (10^2)) / (10.0^^2))) )
                                                             train example (h ([[[0.2, 0.6, 0.7,0.3],       [-0.1, 0.5, 0.25, 0.5],  [0.75, -0.5, -0.8, 0.4] , [-0.1, 0.5, 0.25, 0.5]],
                                                                                [[-0.35, 0.3, 0.8, 0.0],    [0.2, 0.2, 0.0, 1.0],    [-0.1, -0.4, -0.1, -0.4], [-0.1, 0.5, 0.25, 0.5]],
                                                                                [[0.25, 0.25, -0.25, -0.25],[0.5, 0.8, 0.12, -0.12], [0.34, -0.34, -0.9, 0.65], [-0.1, 0.5, 0.25, 0.5]]] )) 
                                                                                [[[0.2]], [[0.0]], [[0.3]], [[-0.2]], [[0.2]], [[0.0]], [[0.3]], [[-0.2]]]

---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |                    Forward & Back Propagation                  | | ---
  --- ‾------------------------------------------------------------------‾---

-- verified 
convoluteDims :: SpatialExtent -> [[[a]]] -> Int -> (Int, Int)
convoluteDims spatialExtent image stride =
    let (m0, n0, i0, j0)     = (spatialExtent, spatialExtent, 
                                length $ head image,  length $ head $ head image )
    in  ((quot (i0 - m0) stride) + 1 , (quot (j0 - n0) stride) + 1 )

convoluteDims2D :: SpatialExtent -> [[a]] -> Int -> (Int, Int)
convoluteDims2D spatialExtent image stride =
    let (m0, n0, i0, j0)     = (spatialExtent, spatialExtent, 
                                length image,  length $ head image )
    in  (quot (i0 - m0) stride + 1 , quot (j0 - n0) stride + 1 )  


-- verified
convFlatten_ind :: Image2D -> SpatialExtent -> Stride -> Image2D
convFlatten_ind image spatialExtent stride =
    let splitVertical image' stackArray =   
                                if length image' < spatialExtent 
                                then stackArray
                                else (splitHorizontal image' (take spatialExtent image') stackArray)
        splitHorizontal image'' imageChunk stack' = case () of 
                                _ | length (head imageChunk) < spatialExtent -> (splitVertical (drop stride image'') stack')
                                _ | otherwise -> let new_stack = (stack' ++ (concat $ map (take spatialExtent) $ take spatialExtent imageChunk))
                                                 in  splitHorizontal image'' (map (drop stride) imageChunk) new_stack
    in chunksOf (spatialExtent*spatialExtent) (splitVertical image [])


convFlatten :: [[Double]] -> SpatialExtent -> Stride -> [[Double]]
convFlatten image spatialExtent stride =

    let splitVert imageV stackArray = 
                            if length imageV < spatialExtent 
                            then stackArray
                            else (splitHori imageV (take spatialExtent imageV) stackArray)

        splitHori imageH imageChunk stack = case () of 
            _ | length (head imageChunk) < spatialExtent -> (splitVert (drop stride imageH) stack)
            _ | otherwise ->    let newStack = stack ++ (concat $ map (take spatialExtent) $ take spatialExtent imageChunk)
                                in splitHori imageH (map (drop stride) imageChunk) newStack

    in chunksOf (sqri spatialExtent) (splitVert image [])

-- verified   
convolute2D_ind :: [[Double]] -> Image2D -> Stride -> Image2D
convolute2D_ind filter image stride
    = let (m, n) = convoluteDims2D (length filter) image stride
          flat_image = convFlatten_ind image (length filter) stride
      in  chunksOf (n) $ zip (zip [0 ..] [0 ..]) $ map (sum . zipWith (*) (concat filter) . map snd) flat_image

-- verified
convolute3D_ind :: Filter -> Image -> Stride -> Image
convolute3D_ind filter image stride
    =  [  convolute2D_ind filter2d image2d stride |  (image2d, filter2d) <- (zip image filter)]

convolute2D :: [[Double]] -> [[Double]] -> Stride -> [[Double]]
convolute2D filter image stride
    = let (m, n) = convoluteDims2D (length filter) image stride
          flat_image = convFlatten image (length filter) stride
      in  chunksOf (n) $ map (sum . zipWith (*) (concat filter)) flat_image

-- verified
convolute3D :: Filter -> [[[Double]]] -> Stride -> [[[Double]]]
convolute3D filter image stride
    =  [  convolute2D filter2d image2d stride |  (image2d, filter2d) <- (zip image filter)]

convoluteDeltaW :: [[Double]] -> [[[Double]]] -> Stride -> [[[Double]]]
convoluteDeltaW delta image stride
    =  [  convolute2D delta image2d stride |  image2d  <- image ]


convoluteDeltaX :: [[Double]] -> [[[Double]]] -> Stride -> [[[Double]]]
convoluteDeltaX delta image stride
    =  let (w, h) = (length (head delta) - 1, length delta - 1)
           padding_w = if w == 0 then [] else [ 0 | x <- [1 .. w]] 
           padding_h = if h == 0 then [] else [[ 0 | x <- [1 .. (length (head image) + w + w)]] | y <- [1 .. h]  ]
           padded_image = map (\mat -> if h == 0 then mat else padding_h ++ mat ++ padding_h) . map2 (\row -> padding_w ++ row ++ padding_w) $ image
       in  [  convolute2D delta image2d stride |  image2d <- padded_image ]


-- verified
forward :: Filter -> Image -> Stride -> Image2D
forward filter image stride 
    = let (m, n)             = convoluteDims (length $ head filter) image stride 
          bias  = 1.0
      in  map (( zip (zip [0 ..] [0 ..]) ) . (map (bias + ))) $ foldr eleaddm (fillMatrix m n 0.0) (map3 snd $ convolute3D_ind filter image stride)

-- verified
pool :: Stride -> SpatialExtent -> Image2D -> Image2D
pool stride spatialExtent image = 
    let flat_image = convFlatten_ind image spatialExtent stride
        image_nums = map2 snd flat_image
        (h, w)     = (length $ image, length $ head image)
        (m, n)     = ((quot (h - spatialExtent) stride) + 1 , (quot (w - spatialExtent) stride) + 1 )
        f (x:xs) i =    let max_x = (maximum x)
                            ind = fromJust $ elemIndex max_x x
                            (m', n')    = (quot ind spatialExtent, ind `mod` spatialExtent)
                            (row, col)  = (quot i n, (stride * i) `mod` n)
                        in  (((row + m', col + n'), max_x):(f xs (i + 1)))
        f [] i     = []
    in  trace (show (m, n) )chunksOf ((quot (length (head image) - spatialExtent) stride) + 1) $ 
            f image_nums 0


-- verified
unpool :: (Int, Int) -> Image2D -> [[Double]]
unpool (orig_w, orig_h) image = 
    let zeros              = replicate orig_h $ replicate orig_w 0.0
        set'  ls (y:ys)    = let ((m, n), value) = y 
                             in  set' (replaceElement ls m (replaceElement (ls !! m) n 1.0)) ys
        set'  ls []        = ls
    in  set' zeros (concat image)

flattenImage :: Image -> Image
flattenImage image = [ [[(i, d)]] | (i, d) <- (concat $ concat $ image) ]

unflatten :: [Double] -> (Int, Int, Int) -> [Deltas] 
unflatten flattened_deltas (m, n, v) 
                        = --  let deltas' = concat (concat deltas)
                            trace (show flattened_deltas) $ map (\x -> [x]) $ map (chunksOf m) (chunksOf (m * n) flattened_deltas)


compDeltaFullyConnected :: Image -> [[[Double]]] -> (Int, Int, Int) -> [Deltas]
compDeltaFullyConnected actualOutput desiredOutput (m, n, v) = 
    unflatten  (zipWith (\actOutput desOutput -> 0.5 * ((snd actOutput) - desOutput)) (concat $ concat actualOutput) (concat $ concat desiredOutput)) (m, n, v)






-- Each filter is applied to the entire depth of images in the current stack of 2D images, so
-- must have the same depth as the input volume
-- Each filter will produce a separate 2D activation map (output volume)
-- Each output volume by a filter, can be interpreted as an output of a neuron
-- Given a receptive field size of 3x3 and input volume of 16x16x20, every neuron in the conv layer
-- would now have a total of 3*3*20 = 180 connections to the input volume. The connectivity is local
-- in space (3x3) but along the full input depth (20).
-- The depth of the output volume is the number of filters we used
