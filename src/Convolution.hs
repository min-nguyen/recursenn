{-# LANGUAGE
     DeriveFunctor,
     DeriveFoldable,
     DeriveTraversable,
     TemplateHaskell, RankNTypes, DeriveFoldable,
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
type Position           = (Int, Int)
type Image2D            = [[Double]] 
type Image              = [Image2D]    
type ImageStack         = [Image]
type Stride             = Int
type SpatialExtent      = Int
type Biases             = [Double]
type Deltas             = [[[Double]]]
type DesiredOutput      = [[[Double]]]


data ForwardProp        = ForwardProp {
                                    _image       :: Image,
                                    _positions   :: [[[Position]]]
                                } deriving Show

makeLenses ''ForwardProp

data BackProp           = BackProp {
                                    forwardProps    :: [ForwardProp],
                                    outerDeltas     :: Deltas,
                                    outerFilters    :: [Filter],
                                    desiredOutput   :: DesiredOutput
                                }


---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |                          Alg & Coalg                           | | ---
  --- ‾------------------------------------------------------------------‾---

alg :: Layer (Fix Layer, ([ForwardProp] -> [ForwardProp]) ) -> (Fix Layer, ([ForwardProp] -> [ForwardProp]))
alg (ConvolutionalLayer filters biases (innerLayer, forwardPass))
        = (Fx (ConvolutionalLayer filters biases innerLayer), 
                (\fps -> 
                    let inputImage = (head fps) ^. image
                        stride = 1
                        outputImage = [map2 (sigmoid) (forwardConvolutional filter inputImage 1) | filter <- filters]
                        output = (head fps) & image .~ outputImage
                    in  (output:fps)) . forwardPass)
alg (PoolingLayer stride spatialExtent (innerLayer, forwardPass))
        = (Fx (PoolingLayer stride spatialExtent innerLayer), 
                (\fps -> 
                    let inputImage = (head fps) ^. image
                        pooledinput = (map (pool stride spatialExtent) inputImage)
                        originalPositions = [ originalPositions  | (originalPositions, outputImage) <- pooledinput ]
                        outputImage = [ outputImage        | (originalPositions, outputImage) <- pooledinput ]
                        output = (head fps) & image .~ outputImage
                                            & positions .~ originalPositions
                    in  (output:fps)) . forwardPass)
alg (ReluLayer (innerLayer, forwardPass))
        = (Fx (ReluLayer innerLayer), 
                (\fps -> 
                    let inputImage  = (head fps) ^. image
                        outputImage = (map3 (\x -> if x < 0 then 0 else x) inputImage)
                        output = (head fps) & image .~ outputImage
                    in  (output:fps) ) . forwardPass)
alg (InputLayer) 
        = (Fx InputLayer, id)
alg (FullyConnectedLayer (innerLayer, forwardPass)) 
        = (Fx (FullyConnectedLayer innerLayer), 
                (\fps -> 
                    let inputImage  = (head fps) ^. image 
                        outputImage =  (flattenImage $ inputImage) 
                        output = (head fps) & image .~ outputImage
                    in  (output : fps) ) . forwardPass )

coalg :: (Fix Layer, BackProp) -> Layer (Fix Layer, BackProp)
coalg (Fx (FullyConnectedLayer innerLayer), BackProp fps outerDeltas outerFilters desiredOutput)
        =   let (output:input:_) = fps
                (outputImage, inputImage) = (output ^. image,  input ^. image)
                (m, n, v)   = (length (head $ head inputImage), length (head inputImage), length inputImage)

                deltas       = compDeltaFullyConnected outputImage desiredOutput (m, n, v)
                
            in  FullyConnectedLayer (innerLayer, BackProp (tail fps) deltas outerFilters desiredOutput)

coalg (Fx (ConvolutionalLayer filters biases innerLayer), BackProp fps outerDeltas outerFilters desiredOutput)
        =   let (output:input:_) = fps
                (outputImage, inputImage) = (output ^. image,  input ^. image)
                learningRate    = 0.02

                deltaX          = let wTdelta = [ (convoluteDeltaX (outerDelta)  (transpose3D filter) 1) 
                                                               |  (outerDelta, filter) <- zip outerDeltas filters] 
                                      wTdelta' = foldr (\m1 m2 -> sumMat3D m1 m2) (head wTdelta) (tail wTdelta) 
                                  in  (mmmul3d wTdelta' ((map3 sigmoid') inputImage))  :: Deltas

                deltaW          = [ (convoluteDeltaW (outerDelta) (map3 (sigmoid') $ transpose3D inputImage) 1)
                                            |  (outerDelta) <- (outerDeltas)] :: [Deltas]

                newFilters      = [ zipWith elesubm filter (map3 (learningRate *) delta_w) 
                                            | (filter, delta_w) <- (zip filters deltaW) ] 

            in  ConvolutionalLayer newFilters biases (innerLayer, BackProp (tail fps) deltaX filters desiredOutput)
     
coalg (Fx (PoolingLayer stride spatialExtent innerLayer), BackProp fps outerDeltas outerFilters desiredOutput)
        =   let (output:input:_) = fps
                (outputImage, inputImage) = (output ^. image,  input ^. image)
                deltas          = [unpool (length $ head input2d, length $ input2d) output2d positions2d | (input2d, output2d, positions2d) <- zip3 inputImage outputImage (output ^. positions) ]
            in   (PoolingLayer stride spatialExtent (innerLayer, BackProp (tail fps) deltas outerFilters desiredOutput) )
coalg (Fx (ReluLayer innerLayer), BackProp imageStack outerDeltas outerFilters desiredOutput)
        =   let inputImage      = (head (tail imageStack)) ^. image
                deltas          = (map3 (\x -> if x < 0 then 0 else x) inputImage)
            in  (ReluLayer (innerLayer,  BackProp (tail imageStack) deltas outerFilters desiredOutput) )
coalg  (Fx InputLayer, backProp)
        =   InputLayer


train :: Fix Layer -> ForwardProp -> DesiredOutput -> Fix Layer 
train neuralnet sample desiredoutput 
    = --trace (show $ head inputStack) $ 
        ana coalg $ (nn, BackProp inputStack [[[]]] [[[[]]]] desiredoutput)
            where 
                (nn, diff_fun)      = cata alg neuralnet
                inputStack          = diff_fun [sample]


trains :: Fix Layer -> [Image] -> [DesiredOutput] -> Fix Layer
trains neuralnet samples desiredoutputs  
    = foldr (\(sample, desiredoutput) nn -> 
                  let updatedNetwork = train nn (ForwardProp sample [[[]]]) desiredoutput
                  in  updatedNetwork) neuralnet (zip samples desiredoutputs)

neuralnet :: IO (Fix Layer)                                                                                    
neuralnet = do 
    weights_a <- randMat4D 3 3 4 1
    weights_b <- randMat4D 3 3 4 1
    weights_c <- randMat4D 3 3 4 2
    let nn = Fx (FullyConnectedLayer   (Fx $ ConvolutionalLayer weights_c [[0.0],[0.0]]
                                            (Fx $ ConvolutionalLayer weights_b [[0.0]]
                                                    (Fx $ ConvolutionalLayer weights_a [[0.0]]
                                                        (Fx $ InputLayer)))))
    return nn

runConvolutional :: [Image] -> [DesiredOutput] -> IO (Fix Layer)
runConvolutional inputs desiredoutputs = do
    nn <- neuralnet 
    return $ trains nn inputs desiredoutputs


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
forwardConvolutional :: Filter -> Image -> Stride -> Image2D
forwardConvolutional filter image stride 
    = let (m, n)             = convoluteDims (length $ head filter) image stride 
          bias  = 1.0
      in  map2 (bias +) $ foldr eleaddm (fillMatrix m n 0.0) (convolute3D filter image stride)

-- verified
pool :: Stride -> SpatialExtent -> Image2D -> ([[Position]], Image2D)
pool stride spatialExtent image = 
    let image_nums = convFlatten image spatialExtent stride

        (h, w)     = (length $ image, length $ head image)
        (m, n)     = ((quot (h - spatialExtent) stride) + 1 , (quot (w - spatialExtent) stride) + 1 )
        f (x:xs) i =    let max_x = (maximum x) :: Double
                            ind = fromJust $ elemIndex max_x x
                            (m', n')    = (quot ind spatialExtent, ind `mod` spatialExtent)
                            (row, col)  = (quot i n, (stride * i) `mod` n)
                        in  (((row + m', col + n'), max_x):(f xs (i + 1)))
        f [] i     = []

        (positions', image2d') = unzip $ f image_nums 0 :: ([Position], [Double])
    in  (chunksOf n positions',  chunksOf n image2d')
--((quot (h - spatialExtent) stride) + 1) $ unzip
-- verified
unpool :: (Int, Int) -> Image2D -> [[Position]] -> [[Double]]
unpool (orig_w, orig_h) image positions = 
    let zeros              = replicate orig_h $ replicate orig_w 0.0
        set'  ls (y:ys)    = let ((m, n), value) = y 
                             in  set' (replaceElement ls m (replaceElement (ls !! m) n 1.0)) ys
        set'  ls []        = ls
    in  set' zeros (zip (concat positions) (concat image))

flattenImage :: Image -> Image
flattenImage image = [ [[(i)]] | (i) <- (concat $ concat $ image) ]


compDeltaFullyConnected :: Image -> [[[Double]]] -> (Int, Int, Int) -> Deltas
compDeltaFullyConnected actualOutput desiredOutput (m, n, v) = 
    let --(prob, idx) = maximumBy (comparing fst) (zip (concat $ concat actualOutput) [0..]) 
        (prob, idx) = maximumBy (comparing fst) (zip (concat $ concat desiredOutput) [0..]) 
        prob2 = (concat $ concat actualOutput) !! idx
        error = sum (zipWith (\actOutput desOutput -> 0.5 * (sqr (1 - prob2))) (concat $ concat actualOutput) (concat $ concat desiredOutput))
    in  writeResult --("desired: " ++ show desiredOutput ++ " actual: " ++ show actualOutput)
         ((\z -> showFullPrecision  $ read $ formatFloatN (z/100) 18) error) 
                --(show actualOutput ++ ", " ++ show desiredOutput) $
         unflatten  (zipWith (\actOutput desOutput -> 0.5 * ((desOutput ) - actOutput)) (concat $ concat actualOutput) (concat $ concat desiredOutput)) (m, n, v)
    where   unflatten :: [Double] -> (Int, Int, Int) -> Deltas
            unflatten flattened_deltas (m, n, v) 
                                = map (chunksOf m) (chunksOf (m * n) flattened_deltas)
        





-- Each filter is applied to the entire depth of images in the current stack of 2D images, so
-- must have the same depth as the input volume
-- Each filter will produce a separate 2D activation map (output volume)
-- Each output volume by a filter, can be interpreted as an output of a neuron
-- Given a receptive field size of 3x3 and input volume of 16x16x20, every neuron in the conv layer
-- would now have a total of 3*3*20 = 180 connections to the input volume. The connectivity is local
-- in space (3x3) but along the full input depth (20).
-- The depth of the output volume is the number of filters we used
